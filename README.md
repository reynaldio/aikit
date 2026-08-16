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

## Installing

```sh
go get github.com/reynaldio/aikit/llm
```

Public module — a plain `go get` works with no extra configuration (no `GOPRIVATE`, no
credentials). Versioned with SemVer tags; pin a release in your `go.mod` as usual.

For co-development against a local checkout, you can temporarily add a `replace` in the
consumer's `go.mod` pointing at a local `aikit` clone — but the committed dependency is the
published tag, so containerized/CI builds resolve it straight from the module proxy.
