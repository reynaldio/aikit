# aikit

Provider-agnostic AI gateways for Go. One public Go module; one package per
modality. Today it ships `aikit/llm`. `aikit/tts` (text-to-speech) and any future modality
are siblings added later — STT is not separate, it rides inside `llm` as a multimodal
completion.

## aikit/llm

A provider-agnostic LLM router: one `Client` interface over Anthropic, Google Gemini, and
OpenAI-compatible backends (DeepSeek, Moonshot, …), with per-profile failover and a
token-cost pricing catalog. Usage *persistence* is the consumer's job — the library emits
token counts on `Response`; wrap the `Client` to record them.

```go
import "github.com/reynaldio/aikit/llm"

c := llm.New(llm.Config{
    AnthropicAPIKey: os.Getenv("ANTHROPIC_API_KEY"),
    Profiles: map[llm.Profile]llm.ModelRef{
        llm.ProfileChat: {Provider: llm.ProviderAnthropic, Model: "claude-haiku-4-5"},
    },
    // Logger is optional; nil = the library logs nothing.
})
resp, err := c.Complete(ctx, llm.Request{
    Task:     llm.TaskChat, // routing key: Task → Profile → (provider, model)
    Messages: []llm.Message{{Role: "user", Content: "halo"}},
})
```

Routing is by `Task` (preferred), `Tier` (legacy coarse knob), or an explicit `Model`
override — there is no `Profile` field on `Request`; profiles are what tasks resolve *to*.

`Complete` returns `llm.ErrNotConfigured` when no provider key is set.

### Refusals are errors, not empty strings

A provider's safety classifiers can decline a request. That arrives as a **successful
200** with an empty or partial body — so a client that only checks `err != nil` would
hand its caller a silent empty answer. `llm` turns it into an error instead:

```go
resp, err := c.Complete(ctx, req)
if errors.Is(err, llm.ErrRefused) {
    var re *llm.RefusalError
    errors.As(err, &re)
    log.Warn("declined", "category", re.Category) // "cyber", "bio", …
}
```

A refusal is routed through the normal failover path, because classifiers differ per
model and a profile's configured fallback is often the model that *will* answer. Set
`Request.NoFallback` to make the refusal final instead. Domains that legitimately trip a
classifier — security tooling against `cyber`, life sciences against `bio` — should
configure that fallback rather than treat the refusal as a defect.

### Effort and structured outputs

```go
resp, err := c.Complete(ctx, llm.Request{
    Task:   llm.TaskReason,
    Effort: llm.EffortXHigh, // low | medium | high | xhigh | max
    // Constrain the reply to a schema — a malformed shape becomes impossible.
    JSONSchema: map[string]any{
        "type":                 "object",
        "properties":           map[string]any{"verdict": map[string]any{"type": "string"}},
        "required":             []string{"verdict"},
        "additionalProperties": false,
    },
    Messages: []llm.Message{{Role: "user", Content: "…"}},
})
```

Both are optional and omitted from the wire when unset, so the model's own defaults
apply. Providers without an equivalent knob ignore them — a caller that *requires* the
schema guarantee should pin `Model` rather than rely on profile routing.

> **`MaxTokens` on thinking models.** Config's default is 1024. On models that think by
> default (Claude Opus 5 and up) `max_tokens` bounds thinking **and** the reply together,
> so a small budget yields a truncated answer. Raise it per-request for those profiles —
> 64000 is a sane floor at `EffortXHigh`.

### Tool calling

`Complete` runs **one** round. Declare tools, run whatever the model asks for,
append the round to the history, and call again — the loop is yours, so an agent
that must log or gate each call can do so.

```go
resp, err := c.Complete(ctx, llm.Request{
    Tools:    []llm.ToolDef{{Name: "get_weather", Description: "Current weather", Schema: schema}},
    Messages: msgs,
})

if resp.StopReason == llm.StopToolUse {
    results := make([]llm.ToolResult, 0, len(resp.ToolCalls))
    for _, call := range resp.ToolCalls {
        out, err := run(call) // your dispatch; call.Input is raw JSON
        results = append(results, llm.ToolResult{
            ToolCallID: call.ID, Content: out, IsError: err != nil,
        })
    }
    msgs = append(msgs,
        llm.Message{Role: "assistant", Content: resp.Text, ToolCalls: resp.ToolCalls},
        llm.Message{Role: "user", ToolResults: results},
    )
}
```

Every result from one round goes in **one** message. Splitting them across
messages is rejected by some providers and silently degrades parallel tool
calling on others.

`ToolCall.ID` is opaque — echo it back and never parse it. Check
`resp.StopReason == llm.StopTruncated`: a turn cut off at the token ceiling
otherwise looks exactly like a finished one.

On Google Gemini, every call in a round must get exactly one result.
`Complete` returns an error if the results don't match the calls
one-to-one — Gemini matches calls to responses by position, so a missing
or unrecognized result would otherwise land on the wrong call instead of
failing loudly.

## Installing

```sh
go get github.com/reynaldio/aikit/llm
```

Public module — a plain `go get` works with no extra configuration (no `GOPRIVATE`, no
credentials). Versioned with SemVer tags; pin a release in your `go.mod` as usual.

For co-development against a local checkout, you can temporarily add a `replace` in the
consumer's `go.mod` pointing at a local `aikit` clone — but the committed dependency is the
published tag, so containerized/CI builds resolve it straight from the module proxy.
