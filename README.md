# aikit

Provider-agnostic AI gateways for Go. One public Go module; one package per
modality. Today it ships `aikit/llm`, `aikit/decide` and `aikit/guard`. `aikit/tts` (text-to-speech) and any future modality
are siblings added later — STT is not separate, it rides inside `llm` as a multimodal
completion.

## aikit/llm

A provider-agnostic LLM router: one `Client` interface over Anthropic, Google Gemini, and
OpenAI-compatible backends (DeepSeek, Moonshot, …), with per-profile failover and a
token-cost pricing catalog. Usage *persistence* is the consumer's job — the library emits
token counts on `Response`; use `llm.WithUsage`, or wrap the `Client` yourself.

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

### Retries

Before failover, a call is retried on the same model at most twice, for 429,
500, 502, 503, 504, 529 and connection errors. The wait follows `retry-after-ms`
or `Retry-After` when the server sends one. If the server asks for more than
10s, there is no retry and the call fails over at once. Without a header, the
waits are 0.5s and then 1s, a little less with jitter. There is no retry if the
wait would pass the context deadline.

A streamed reply is never retried once it has started, so the streaming
failover rule is unchanged. The policy is fixed with no `Config` setting. Each
retry logs one `Warn` line `llm: retrying` to `Config.Logger`.

**Behavior change:** Anthropic used to wait however long `Retry-After` asked.
A wait over 10s now fails over at once, and Anthropic's backoff now matches the
other providers.

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
zero `Response`. A refusal returns a populated one — partial text and usage
alongside its error — whose `StopReason` reads `StopEndTurn` rather than the
zero value. A stream that ended early after data arrived also returns its text
and token counts with the error (its `StopReason` is ""). Either way a caller
metering cost can still see what was billed. Branch on the error first.
`StopToolUse` and `StopTruncated` are mutually
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

#### Tool loop helper

For simple loops with no specialized logic per round, use `llm.RunTools`:

```go
tools := map[string]llm.ToolFunc{
    "search": func(ctx context.Context, call llm.ToolCall) (string, error) {
        var in struct{ Query string }
        if err := json.Unmarshal(call.Input, &in); err != nil {
            return "", err
        }
        return search(ctx, in.Query)
    },
}
run, err := llm.RunTools(ctx, c, req, tools, llm.ToolLoopOptions{
    MaxRounds: 5,   // default 8
    Parallel:  true, // run calls in a round concurrently
    OnRound: func(round int, resp llm.Response) error {
        log.Info("round", "n", round, "stop_reason", resp.StopReason)
        return nil
    },
})
if errors.Is(err, llm.ErrToolLoopLimit) {
    log.Warn("model wanted more tool calls after MaxRounds")
}
if errors.Is(err, llm.ErrToolCallsTruncated) {
    log.Warn("tool calls cut off at token limit")
}
```

An unknown tool name or a function error goes back to the model as a failed
result (`IsError`), and the loop goes on. With `WithUsage`, each round is
recorded on its own. With `guard.Wrap`, call `guard.BeginTurn(ctx)` first.
A call with `Parallel: true` must be safe to run from multiple goroutines.
The hand-written loop above is still fine for callers that need per-round
gating, logging or conditional branching.

### Streaming

Stream the reply's text as it's generated by setting `Request.OnEvent`. `Complete`
still returns the final `Response` — streaming is read-only, not an alternative to
returning the answer.

```go
resp, err := c.Complete(ctx, llm.Request{
    Messages:   msgs,
    Tools:      []llm.ToolDef{ /* ... */ },
    ToolEvents: llm.ToolEventsStartReady, // optional: send tool start and ready events
    OnEvent:    func(e llm.StreamEvent) {
        // w is the http.ResponseWriter of the request being served.
        switch e.Kind {
        case llm.StreamText:
            w.Write([]byte(e.Text))
            if f, ok := w.(http.Flusher); ok {
                f.Flush()
            }
        case llm.StreamToolStart:
            // The model started writing a call to e.ToolName.
            // Show an indicator keyed by e.ToolCallID, e.g. "Calling search_docs…".
            // Running the tool is your job when Complete returns.
        case llm.StreamToolReady:
            // The model finished writing that call.
        }
    },
})
if err != nil { /* handle */ }
// resp.Text is the complete answer (and equals the joined StreamText pieces).
if len(resp.ToolCalls) > 0 { /* run tools */ }
```

**Tool events** let you show "Calling search_docs…" while the model is still writing a tool call.
Running the tool is your job — the app gets a start event as the model begins the call, and
optionally a ready event when it finishes. Set `Request.ToolEvents` to `llm.ToolEventsStart` for
start only, or `llm.ToolEventsStartReady` for both. The zero value sends no tool events. Tool
events are ignored when `OnEvent` is nil or when `JSONSchema` is set.

**Rules for tool events:**

- For each call, `StreamToolStart` comes before `StreamToolReady`. Each is sent at most once per
  call ID.
- `ToolCallID` equals the ID of the matching `ToolCall` in `Response.ToolCalls`, so the app
  can match them.
- Only calls to the app's own tools (`Request.Tools`) produce events. Built-in tools (web search)
  never do.
- When `Complete` returns, `Response.ToolCalls` is the truth. A call the app saw a start or
  ready for may be missing from it: after a fallback (the call was from the first model), or on
  an error. Clear any indicator whose ID is not in `Response.ToolCalls`.
- Ready means the model finished writing the call. It does not promise the call is usable: a
  reply cut off at the token limit may have sent ready and still return `StopTruncated`. Check
  `Response.StopReason` as usual.
- A tool event may arrive slightly before text that the guard's marker filter is still holding
  back (at most the marker's length). Text order and the join rule are unaffected.

**`OnEvent` rules** (apply to all event kinds):

- It is called on the goroutine that called `Complete`, in order, never concurrently, and never
  after `Complete` returns.
- It must be quick and must not block: the stream is read on the same goroutine, so a slow
  callback slows the reply.
- Ignore kinds you do not know; later versions may add more.
- Join rule: when `Complete` returns a nil error, the text of all `StreamText` events, joined in
  order, equals `Response.Text` exactly.
- On error, events already sent stay sent, and the response is whatever `Complete`
  returns without streaming.

**Fallback.** A provider or a refusal before any text arrives falls back as usual. An error after
text has been sent does not: the user already saw part of a reply, so text from another model
would be wrong. Tool events never block fallback: only text does.

**`JSONSchema` requests.** The text arrives as one event after the reply has been checked and
the schema is valid, since half a JSON document is not usable. A provider that cannot stream
also sends its whole text as one event.

**Tool rounds.** A call's arguments are never streamed. With `ToolEvents` set you get start
(and optionally ready) events while the model writes a call; the calls themselves arrive whole
in `resp.ToolCalls`, which is the truth when `Complete` returns.

**OpenAI-compatible services.** Usage reporting needs `stream_options.include_usage` in the
request. A service that ignores it reports 0 tokens, and aikit logs a warning: `llm: stream
reply had no usage; tokens reported as 0`.

**Stop-billing:** A stream that stops or breaks after data has arrived returns its token
counts with the error. Counts the provider had not sent yet are guessed using about 4
characters per token, and `Response.UsageEstimated` reports this. Before any data arrives,
the counts are zero as before. **Behavior change** from v0.9.0: before, the reply was zero.

### Recording usage

Wrap the client with `llm.WithUsage` to record token usage and cost for each call:

```go
c := llm.New(cfg)
var db *sql.DB // your database

// Wrap the client once.
c = llm.WithUsage(c, nil, func(ctx context.Context, u llm.Usage) { // nil = llm.DefaultPrices
    // ctx may be cancelled. Use context.WithoutCancel to save the record anyway.
    ctx = context.WithoutCancel(ctx)
    err := saveUsage(ctx, db, u)
    if err != nil {
        log.Error("usage record failed", "err", err)
    }
})

resp, err := c.Complete(ctx, req) // usage is recorded before this returns
```

`WithUsage` reports every call with tokens — successes, schema violations,
refusals, and streams stopped early all carry InputTokens, OutputTokens and cost.
The `book` argument prices the tokens; nil uses `DefaultPrices`. `record` runs
on the calling goroutine before `Complete` returns, but your ctx may already be
cancelled (the user pressed Stop), so use `context.WithoutCancel(ctx)` to save.

**Known gap:** if the primary model in your profile fails and the fallback answers,
only the fallback's tokens are reported. Tokens the primary already charged (from a
refusal or an early stream break) are not reported; this reports too little, but
never charges twice.

### Testing with llmtest

Package `llmtest` is a scripted fake `llm.Client` for tests. Queue up replies and
inspect the requests that were made:

```go
import "github.com/reynaldio/aikit/llm/llmtest"

c := llmtest.New(
    llmtest.Reply{Response: llm.Response{ToolCalls: []llm.ToolCall{{
        ID: "1", Name: "search", Input: json.RawMessage(`{"q":"foo"}`)}}},
    llmtest.Reply{Response: llm.Response{Text: "answer"}},
)
// ... call your code with c ...
reqs := c.Requests() // one entry per Complete call, in order
```

With `Request.OnEvent` set, the fake sends events from the replies to OnEvent,
matching how a real provider streams. `Reply.Events` lets you script exact events;
without it, the fake builds them from the Response. It is safe for concurrent use.

### Embeddings

Embeddings turn text into vectors for semantic search, clustering and comparison.
Use `NewEmbedder` with your `Config` to build an embedder; call `Embed` to get vectors.

```go
import "github.com/reynaldio/aikit/llm"

cfg := llm.Config{
    OpenAIAPIKey: os.Getenv("OPENAI_API_KEY"),
    EmbedModel:   llm.ModelRef{Provider: llm.ProviderOpenAI, Model: "text-embedding-3-small"},
    // or: llm.ModelRef{Provider: llm.ProviderGoogle, Model: "gemini-embedding-001"}
}
embedder := llm.NewEmbedder(cfg)

docs, _ := embedder.Embed(ctx, llm.EmbedRequest{
    Texts: []string{"cat", "dog"}, Kind: llm.EmbedDocument})
q, _ := embedder.Embed(ctx, llm.EmbedRequest{
    Texts: []string{"pet"}, Kind: llm.EmbedQuery})

// Compare vectors with a dot product. Vectors are unit length.
var dotProduct float32
for i, v := range docs.Vectors[0] {
    dotProduct += v * q.Vectors[0][i]
}
```

**No failover:** an error from the provider is returned as-is. **Vectors from different
models are not comparable**, so store `resp.Model` with each vector. When you change
models, re-embed all your data. Chunking happens automatically; on an error you get
no vectors but the billed tokens are still reported in `resp.InputTokens`.

Metering: use `WithEmbedUsage` to record each call; records have `Task == TaskEmbed`
(input tokens only). For testing, `llmtest.NewEmbedder` makes a fake.

Supported providers: OpenAI, OpenAI-compatible (set `OpenAIBaseURL`), and Google Gemini.
Anthropic, DeepSeek and Moonshot return `ErrEmbedUnsupported`. A provider that is
unknown or has no key gets `ErrNotConfigured`.

For search, documents and queries must use the same model, and both should carry a
`Kind` (`EmbedDocument` for the texts you store, `EmbedQuery` for the search text).

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
before sending each piece of text. One known difference from the non-streamed
path: a reply that starts with blank lines and has the marker only in the middle
keeps those leading blanks when streamed.

## Installing

```sh
go get github.com/reynaldio/aikit/llm
```

Public module — a plain `go get` works with no extra configuration (no `GOPRIVATE`, no
credentials). Versioned with SemVer tags; pin a release in your `go.mod` as usual.

For co-development against a local checkout, you can temporarily add a `replace` in the
consumer's `go.mod` pointing at a local `aikit` clone — but the committed dependency is the
published tag, so containerized/CI builds resolve it straight from the module proxy.
