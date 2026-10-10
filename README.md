# aikit

Provider-agnostic AI gateways for Go. One public Go module; one package per
modality. Today it ships `aikit/llm`, `aikit/decide` and `aikit/guard`. `aikit/tts` (text-to-speech) and any future modality
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
apply. Providers without an effort knob ignore `Effort`.

`JSONSchema` is handled differently by each provider:

| Provider | Schema |
| --- | --- |
| Anthropic | Enforced (`output_config.format`) |
| Google Gemini | Enforced (`generationConfig.responseFormat`). Keywords Gemini doesn't support, such as `pattern` or `minLength`, are stripped; `const` is sent as a one-value `enum` |
| OpenAI | Enforced (`response_format` json_schema, `strict: true`) on OpenAI's own endpoint, for schemas strict mode can express. Every object is closed, and an optional property is sent as required-but-nullable, so it comes back `null` instead of missing. A schema using `allOf`, `not`, `if`, an open object, or a non-object root isn't sent, and gets the check below |
| DeepSeek, Moonshot, a custom `OpenAIBaseURL` | Not enforced. A ```` ```json ```` fence is unwrapped, and a reply that still isn't JSON is `llm.ErrSchemaViolation` |

- **Failover keeps the schema.** With `JSONSchema` set, failover only moves to a
  provider that enforces that schema, never to one that would ignore it.
- **A cut-off reply is an error.** A truncated reply (`StopTruncated`) with a schema is
  `ErrSchemaViolation`, never partial JSON returned as success. The `Response` still
  carries the text and tokens, for logging and metering.
- **Gemini before 3.x can't combine JSON mode with tools.** On a `gemini-2.*` model,
  `JSONSchema` plus `WebSearch` is an error, and `JSONSchema` plus `Tools` sends the
  tools without JSON mode for that round. Gemini 3 takes both.

> **`MaxTokens` on thinking models.** Config's default is 1024. On models that think by
> default (Claude Opus 5 and up) `max_tokens` bounds thinking **and** the reply together,
> so a small budget yields a truncated answer. Raise it per-request for those profiles —
> 64000 is a sane floor at `EffortXHigh`.
>
> **Large `MaxTokens` on Anthropic.** Anthropic: requests above the SDK's non-streaming
> limit are sent as a stream and assembled; Complete's contract is unchanged. The SDK
> refuses a non-streaming call it estimates at over 10 minutes (about `MaxTokens` > 21333,
> or a model's own cap), so those calls stream instead and return the same `Response`:
> text, usage, tool calls, stop reason and refusals. A stream that ends before
> `message_stop` is an error, never a partial reply. **A streamed call has no SDK request
> timeout**, so pass a `ctx` with a deadline: without one, a stalled stream waits forever.
>
> **Long replies on Google and OpenAI.** These requests are bounded by your `ctx`
> deadline, with no fixed cap underneath. A request whose `ctx` has no deadline gets
> `max(120s, 1h × MaxTokens / 128000)`, or `Config.RequestTimeout` if set. Expiry and
> cancellation surface as `context.DeadlineExceeded` / `context.Canceled`, including
> when a reply stalls partway through its body. Before v0.7.0 a fixed 120 s client
> timeout cut off any longer reply; a caller that relied on that cutoff should now pass
> a deadline.

### Tool calling

`Complete` runs **one** round. Declare tools, run whatever the model asks for,
append the round to the history, and call again — the loop is yours, so an agent
that must log or gate each call can do so.

```go
resp, err := c.Complete(ctx, llm.Request{
    Tools:    []llm.ToolDef{{Name: "get_weather", Description: "Current weather", Schema: schema}},
    Messages: msgs,
})

// Branch on the CALLS, not on the stop reason — a turn that hit the token
// ceiling while emitting calls reports StopTruncated and still has them.
if len(resp.ToolCalls) > 0 {
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
if resp.StopReason == llm.StopTruncated {
    // Separate concern: the turn was cut off at the token ceiling. Whatever it
    // did emit is incomplete — raise MaxTokens rather than record it as done.
}
```

Every result from one round goes in **one** message. Splitting them across
messages is rejected by some providers and silently degrades parallel tool
calling on others.

`ToolCall.ID` is opaque — echo it back and never parse it. `ToolCall.Input` is
always a valid JSON object: a call with no arguments arrives as `{}`, never as
nil, `""` or `null`, so handlers can unmarshal it unconditionally.

`StopReason` is only meaningful when `err == nil`. A transport error returns a
zero `Response`, but a refusal returns a populated one — partial text and usage
alongside its error, so a caller metering cost can still see what was billed —
whose `StopReason` reads `StopEndTurn` rather than the zero value. Branch on
the error first either way. `StopToolUse` and `StopTruncated` are mutually
exclusive, which is why the loop above branches on `len(resp.ToolCalls)` and
checks truncation separately: `StopTruncated` with calls present means the
model was cut off mid-round, and treating it as a finished turn records
incomplete work as done.

**`ToolResult.IsError` on OpenAI-compatible backends.** Anthropic and Gemini
carry the flag structurally (`is_error` / an `isError` key). The OpenAI chat
API's `tool` message has no field for it, so there the flag is rendered into the
content instead, prefixed with `Error: `. The model still sees the failure on all
three providers — but on OpenAI it sees it as text, so don't rely on the
distinction being machine-readable there.

On Google Gemini, every call in a round must get exactly one result.
`Complete` returns an error matching `errors.Is(err, llm.ErrToolResultMismatch)`
if the results don't match the calls one-to-one — Gemini matches calls to
responses by position, so a missing or unrecognized result would otherwise land
on the wrong call instead of failing loudly.

### Streaming

Stream the reply's text as it's generated by setting `Request.OnEvent`. `Complete`
still returns the final `Response` — streaming is read-only, not an alternative to
returning the answer.

```go
resp, err := c.Complete(ctx, llm.Request{
    Messages: msgs,
    OnEvent: func(e llm.StreamEvent) {
        if e.Kind == llm.StreamText {
            w.Write([]byte(e.Text))
            w.Flush()
        }
    },
})
if err != nil { /* handle */ }
// resp.Text is the complete answer (and equals the joined StreamText pieces).
if len(resp.ToolCalls) > 0 { /* run tools */ }
```

Rules for `OnEvent`:

- It is called on the goroutine that called `Complete`, in order, never
  concurrently, and never after `Complete` returns.
- It must be quick and must not block: the stream is read on the same goroutine,
  so a slow callback slows the reply.
- `v1` sends only `StreamText` events. Ignore kinds you do not know; later
  versions may add more.
- Join rule: when `Complete` returns a nil error, the text of all `StreamText`
  events, joined in order, equals `Response.Text` exactly.
- On error, events already sent stay sent, and the response is whatever
  `Complete` returns without streaming. Once text has been sent the router does
  not fail over: a second model's text would be glued onto the first's. An error
  before any text still fails over as usual.

**Fallback.** A provider or a refusal before any text arrives falls back as
usual. An error after text has been sent does not: the user already saw part of
a reply, so text from another model would be wrong.

**`JSONSchema` requests.** The text arrives as one event after the reply has been
checked and the schema is valid, since half a JSON document is not usable. A
provider that cannot stream also sends its whole text as one event.

**Tool rounds.** Tool calls are never streamed; they arrive whole in `resp.ToolCalls`
alongside the rest of the response.

**OpenAI-compatible services.** Usage reporting needs `stream_options.include_usage`
in the request. A service that ignores it reports 0 tokens, and aikit logs a warning:
`llm: stream reply had no usage; tokens reported as 0`.

## aikit/decide

Typed decisions instead of text, backed by TypeSafe's [Jev](https://docs.typesafe.ai/api).
One call evaluates a `State` against a map of questions and returns a calibrated answer
per question: a yes/no probability (`Noul`), a pick from a closed list (`Choice`), or a
position on a rubric (`Score`). Use it for routing, triage and gating, where you want a
probability to threshold rather than prose to parse.

```go
import "github.com/reynaldio/aikit/decide"

d := decide.New(decide.Config{APIKey: os.Getenv("TYPESAFE_API_KEY")}) // Model defaults to "jev-latest"
resp, err := d.Evaluate(ctx, decide.Request{
    State: ticketText, // a string, or any JSON-marshalable value
    Questions: map[string]decide.Question{
        "urgent": decide.Noul("Does this need action today?", "", ""),
        "team":   decide.Choice("Which team owns it?", map[string]string{"billing": "charges, refunds", "technical": "bugs, outages"}),
        "mood":   decide.Score("How upset is the customer?", "Calm", "Frustrated", "Very angry"),
    },
})
if resp.Answers["urgent"].Noul > 0.8 { /* page someone */ }
```

A nil error guarantees an answer for every question. A missing one is an error, because
a zero `Noul` would read as a confident "no". `Score` is probability-weighted and can
land between levels. Non-2xx replies come back as `*decide.APIError`. The client does
not retry, so back off yourself when `Retryable()` is true (429 / 529 / 5xx).
`Evaluate` returns `decide.ErrNotConfigured` when no key is set. Jev's rates are in
`llm.DefaultPrices`, so you can price `InputTokens` with the same `PriceBook`.

## Guardrails (`aikit/guard`)

Prompt-injection fencing, injection scanning, size caps and per-user rate limits,
applied once at the `llm.Client` seam.

**Wrap the client once.** Every call then gets the untrusted-content rule
(`guard.DataRule`) in its system message, and is size-checked and rate-limited by class:

```go
import "github.com/reynaldio/aikit/guard"

ai := guard.Wrap(guard.Options{
    Inner:    llm.New(cfg),
    Classify: func(ctx context.Context) guard.Class { /* from your usage label */ },
    Identity: func(ctx context.Context) (string, bool) { /* the user id; !ok = not limited */ },
    Limits:   guard.DefaultLimits(),
    TopicPolicy:   myScopeText,      // conversational calls only
    RefusalMarker: "⟦app:declined⟧", // stripped from conversational replies
    Audit:         myAuditSink,      // suspected injections; never store the text
    OnEvent:       func(ctx context.Context, e guard.Event) { log.Info(e.Kind) },
})
```

| Class | Rules added | Size caps | Rate limit |
| --- | --- | --- | --- |
| `ClassOneShot` (the default) | `DataRule` | yes | per call, `OneShotPerHour` |
| `ClassConversational` | `DataRule` + `TopicPolicy` (+ marker instruction) | yes | once per turn, per minute and per hour |
| `ClassBackground` | `DataRule` | no | never |

Refusals come back as `guard.ErrRequestTooLarge` / `guard.ErrRateLimited`, before the
model is called. Map them onto your own errors with `errors.Is`. The built-in limiter is
in-process; pass `Options.Limiter` to share limits across replicas.

- **Fence every piece of outside text** with `guard.Fence`: uploaded files, tool results,
  memory, context. The model is told to treat fenced text as data. Fence ids are stable
  for the same content, so fencing doesn't break prompt caching, and fenced text can't
  close its own block.
- **Call `guard.BeginTurn(ctx)` once per user message** that may run several tool rounds.
  The turn counts once against the rate limit, its outcome (allowed or refused) sticks for
  every round, and each suspicious source is audited once.
- **Call `guard.CheckMessage`** at entry points that take typed text, to cap message
  length and attachment count before anything runs.
- **The scanner audits; it never blocks.** `guard.ScanInjection` spots text that tries to
  instruct the model, in English and Indonesian, and is tuned against ordinary contract
  wording. The protection is the fence plus `DataRule`; the audit is for awareness.
  Sources holding the user's own words (`OwnSources`, by default transcript and memory)
  are fenced but not scanned.

With `guard.Wrap`, the refusal marker never reaches `OnEvent` — streamed replies strip it
before sending each piece of text.

## Installing

```sh
go get github.com/reynaldio/aikit/llm
```

Public module — a plain `go get` works with no extra configuration (no `GOPRIVATE`, no
credentials). Versioned with SemVer tags; pin a release in your `go.mod` as usual.

For co-development against a local checkout, you can temporarily add a `replace` in the
consumer's `go.mod` pointing at a local `aikit` clone — but the committed dependency is the
published tag, so containerized/CI builds resolve it straight from the module proxy.
