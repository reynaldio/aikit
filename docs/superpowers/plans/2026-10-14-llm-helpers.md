# Helpers and Stop-Billing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a test fake (`llmtest`), a usage wrapper (`llm.WithUsage`) and a tool loop (`llm.RunTools`).
Also make a stopped or broken stream return the tokens used so far, instead of a zero `Response`.

**Architecture:** Three new small files in `llm` (`usage.go`, `tools.go`, `estimate.go`) and one new package
`llm/llmtest`. The stop-billing fix changes the early-end paths of each provider's stream code and adds one
field, `Response.UsageEstimated`. The router changes in one place: it stamps Provider/Model on any failed reply
that has tokens.

**Tech Stack:** Go 1.25, standard library, `github.com/anthropics/anthropic-sdk-go` v1.56.0. No new
dependencies.

**Spec:** [docs/superpowers/specs/2026-10-14-llm-helpers-design.md](../specs/2026-10-14-llm-helpers-design.md).
Read it before your task. Its decisions are final. Test numbers below (`#8`, `#26`, …) are the spec's numbers.

This is sub-project 2 of 4 (retries → helpers → Embed → Batch). Nothing here is for the later ones.

## Global Constraints

- Branch: `feat/llm-helpers` (created from main; this plan is its first commit). Commit once per task. End
  every commit message with a blank line, then
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` on its own line.
- No new module dependencies. No new `Config` fields.
- New exported names are exactly the ones in "Shared names" below. Everything else is package-private.
- Errors returned to callers do not change. `errors.Is(err, context.Canceled)`,
  `errors.Is(err, io.ErrUnexpectedEOF)`, the error text and `shouldFailover` behave as today. Only the
  `Response` that comes with the error changes.
- A partial reply (stop-billing) carries `Text` and token counts only. `ToolCalls` is nil and `StopReason` is
  "".
- The token guess is `ceil(runes / 4)`, from `estimateTokens`. Nothing else guesses tokens.
- Tests never sleep to wait for something. Use channels, or the "quiet server" helpers that already exist.
- Write comments and docs in plain, short English (owner preference).
- Gate commands (run from the repo root):
  - During a task: `go test -race ./llm/... -run '<pattern>'`
  - Before each commit: `go vet ./...`
  - Once, at the end (Task 9): `go vet ./...` and `go test -race ./...`

---

## File map

| File | Change | Responsibility after this plan |
| --- | --- | --- |
| `llm/llmtest/llmtest.go` | new | The fake `Client`: scripted replies, recorded requests, stream events |
| `llm/llmtest/llmtest_test.go` | new | Tests #1–#7 |
| `llm/estimate.go` | new | `estimateTokens`, `estimateRequestTokens` |
| `llm/estimate_test.go` | new | Test #34 and the request-guess table |
| `llm/llm.go` | modify | `Response.UsageEstimated`; `completeOn` stamps any failed reply with tokens; a small `hasUsage` helper |
| `llm/anthropic.go` | modify | `stream`/`send` keep the partial message; `anthropicPartial` builds the partial reply |
| `llm/google.go` | modify | `sendStream` keeps text, thought text, calls and usage; builds the partial reply on early end |
| `llm/openai.go` | modify | `completeStream` builds the partial reply on early end |
| `llm/usage.go` | new | `Usage`, `WithUsage` |
| `llm/usage_test.go` | new | Tests #8–#14 (package `llm_test`, uses `llmtest`) |
| `llm/usage_router_test.go` | new | Test #37 (package `llm`, uses the internal fakes) |
| `llm/tools.go` | new | `ToolFunc`, `ToolLoopOptions`, `ToolRun`, `RunTools`, two errors |
| `llm/tools_test.go` | new | Tests #15–#25 (package `llm_test`, uses `llmtest`) |
| `llm/anthropic_stream_test.go` | modify | Tests #26–#29, #33 (Anthropic); follow-up 6 (Anthropic part) |
| `llm/google_stream_test.go` | modify | Tests #30, #31, #33 (Gemini); follow-ups 4 and 6 (Gemini part) |
| `llm/openai_stream_test.go` | modify | Tests #32, #33 (OpenAI) |
| `llm/stream_test.go` | modify | Tests #35, #36; follow-up 7; `evRec.desc` prints unknown kinds (follow-up 6) |
| `llm/retry_test.go` | modify | Follow-up 8 |
| `README.md` | modify | Four short sections (Task 9) |

## Shared names (exact)

```go
// llm/llm.go — new field on Response, after StopReason:
// UsageEstimated is true when at least one token count is a guess, not the
// provider's number. Only a stream that ended early sets it.
UsageEstimated bool

// llm/llm.go — package-private:
func hasUsage(r Response) bool // any of the four token counts > 0

// llm/estimate.go — package-private:
func estimateTokens(s string) int          // (utf8.RuneCountInString(s) + 3) / 4
func estimateRequestTokens(req Request) int // see Task 2

// llm/usage.go — exported:
type Usage struct {
    Provider         Provider
    Model            string
    Task             Task
    InputTokens      int
    OutputTokens     int
    CachedTokens     int
    CacheWriteTokens int
    CostUSD          float64
    Estimated        bool
    Duration         time.Duration
    Err              error
}
func WithUsage(inner Client, book *PriceBook, record func(ctx context.Context, u Usage)) Client

// llm/tools.go — exported:
type ToolFunc func(ctx context.Context, call ToolCall) (string, error)
type ToolLoopOptions struct {
    MaxRounds int
    Parallel  bool
    OnRound   func(round int, resp Response) error
}
type ToolRun struct {
    Response Response
    Messages []Message
    Rounds   []Response
}
var ErrToolLoopLimit      = errors.New("llm: tool loop reached MaxRounds")
var ErrToolCallsTruncated = errors.New("llm: tool calls were cut off at the token limit")
func RunTools(ctx context.Context, c Client, req Request, tools map[string]ToolFunc, opts ToolLoopOptions) (ToolRun, error)

// llm/llmtest/llmtest.go — exported (package llmtest):
type Reply struct {
    Response llm.Response
    Err      error
    Events   []llm.StreamEvent
}
type Client struct{ /* unexported fields */ }
func New(replies ...Reply) *Client
func (c *Client) Complete(ctx context.Context, req llm.Request) (llm.Response, error)
func (c *Client) Enabled() bool                  // true
func (c *Client) CompareTargets() []llm.ModelRef // nil
func (c *Client) Requests() []llm.Request
func (c *Client) Add(replies ...Reply)
var ErrNoReplies = errors.New("llmtest: no replies left")
```

Package rule: `llmtest` imports `llm`. So any test that imports `llmtest` must be in package `llm_test`, in
its own file. Tests that need internal fakes (`streamFake`, `newStreamRouter`) stay in package `llm`.

## Task order and dependencies

1 → (2 → 3, 4, 5) → 6 → 7 → 8 → 9. Task 1 is independent and goes first, because Tasks 6 and 7 use it.
Tasks 3, 4 and 5 each need Task 2 only, and can run in any order.

---

### Task 1: `llmtest` fake client

Low risk. Model: Sonnet. Review: the controller reads the diff.

**Files:**
- Create: `llm/llmtest/llmtest.go`, `llm/llmtest/llmtest_test.go`

**Interfaces:**
- Consumes: `llm.Client`, `llm.Request`, `llm.Response`, `llm.StreamEvent` and the `Stream*`/`ToolEvents*`
  constants (existing).
- Produces: everything under `llmtest` in "Shared names". Tasks 6 and 7 build their tests on it.

**What to build** (spec part 1; rule order is in the spec)

- A package doc comment with a 6–10 line example: a tool-call reply, then a text reply, then checking
  `Requests()`.
- `Client` holds a `sync.Mutex`, the reply queue and the recorded requests. `var _ llm.Client = (*Client)(nil)`.
- `Complete`: lock, record `req`, check `ctx.Err()` (return it, no reply used), pop the next reply or return
  `ErrNoReplies`, unlock, then send events, then return. Events are sent **after** unlocking.
- `Requests()` returns a copy of the slice (a new slice; the `Request` values are copied as values).
- Event generation:
  - `Reply.Events` set: send them exactly. Nothing else.
  - `Err != nil`: send nothing.
  - `req.JSONSchema` set: one `StreamText` with the whole text, if not empty. No tool events.
  - Otherwise: text in up to 3 pieces at rune boundaries (`splitText(s string) []string`: an empty string
    gives no pieces; 1–2 runes give one piece each; longer text gives 3 pieces of about equal rune count).
    Then, for each `ToolCall` in order: `StreamToolStart` if `req.ToolEvents` is `ToolEventsStart` or
    `ToolEventsStartReady`; then `StreamToolReady` if it is `ToolEventsStartReady`. Each event has
    `ToolCallID: call.ID, ToolName: call.Name`.

**Tests** (`package llmtest`)

- #1 three replies in order; a fourth call returns `ErrNoReplies`.
- #2 `Requests()` order; appending to or changing the returned slice does not change the next `Requests()`.
- #3 cancelled ctx: `ctx.Err()`, and the reply is still there for the next call.
- #4 the joined text pieces equal `Response.Text` for "", "a", "héllo wörld" (multi-byte); tool events for
  each `ToolEvents` value (none, start, start_ready, and "bogus" = none).
- #5 `JSONSchema` set: exactly one text event. `Err` set: no events.
- #6 `Reply.Events` sent exactly, also with `Err` set.
- #7 10 goroutines, 10 replies, under `-race`: each reply returned exactly once.
- `Add` appends after the existing replies.

**Steps**
- [ ] Write the tests. Run `go test -race ./llm/llmtest/`: they fail (no package yet).
- [ ] Write `llmtest.go`. Run the tests: pass.
- [ ] `go vet ./...`. Commit: `feat(llmtest): scripted fake llm.Client`.

---

### Task 2: `UsageEstimated`, the token guess, router stamping

Money. Model: Sonnet. Review: **Opus** per-task review.

**Files:**
- Create: `llm/estimate.go`, `llm/estimate_test.go`
- Modify: `llm/llm.go` (`Response`, `completeOn` around lines 702–713, new `hasUsage`)
- Modify: `llm/stream_test.go` (tests #35, #36; follow-up 7; `evRec.desc`)

**Interfaces:**
- Produces: `Response.UsageEstimated`, `hasUsage`, `estimateTokens`, `estimateRequestTokens`. Tasks 3–6 use
  them.

**What to build**

`estimateTokens(s)`: `(utf8.RuneCountInString(s) + 3) / 4`. So "" → 0, "abc" → 1, "abcd" → 1, "abcde" → 2.

`estimateRequestTokens(req)`: `estimateTokens` of one string made by joining, with no separator:
1. `req.SystemCacheable`
2. for each message, in order: `Content`; then each `ToolCall`'s `Name` and `string(Input)`; then each
   `ToolResult`'s `Content`
3. for each `ToolDef`: `Name`, `Description`, and `json.Marshal(Schema)` (ignore a marshal error: add nothing)

Images and documents add nothing. Joining before guessing (instead of adding up many small guesses) keeps the
rounding from adding up.

`hasUsage(r)`: `r.InputTokens > 0 || r.OutputTokens > 0 || r.CachedTokens > 0 || r.CacheWriteTokens > 0`.

`completeOn`, exact change: the error branch today stamps only when `errors.Is(err, ErrRefused)`. Change the
condition to `errors.Is(err, ErrRefused) || hasUsage(resp)`. Update its comment: a refusal, a schema
violation and a stream that ended early can all return a reply with billed tokens. (A schema violation is
already stamped on the success path before `checkJSONReply`; leave that alone.) Nothing else in the router
changes.

Follow-up 7 (`llm/stream_test.go`):
- Rename `eq` to `eqStrings` and update all callers (`grep -rn "eq(" llm/*_test.go` to find them; only the
  helper named exactly `eq`).
- Rewrite `TestStreamDoesNotMutateCallersRequest` so it can fail. Give the caller's request a `Messages` slice
  with spare capacity (`make([]Message, 1, 4)`), call `Complete`, then check that:
  - `len(req.Messages)` is still 1, and `req.Messages[:cap(req.Messages)]` has no new elements written past
    index 0 (compare against a zero `Message`);
  - `req.OnEvent` is still the caller's function (call it and see the caller's recorder get the event, as
    today).

`evRec.desc()`: add a `default:` case that appends `"unknown:" + string(ev.Kind)`.

**Tests**
- `estimate_test.go`: #34 as listed above, plus 2 runes of "é" count as 2 runes; `estimateRequestTokens` on
  a request with each field filled once (expect the guess of the joined string, computed in the test with
  `estimateTokens`), and with only an image (expect 0).
- `stream_test.go` #35: a `streamFake` whose `complete` sends one text event, then returns
  `Response{Text: "Hi", InputTokens: 10, OutputTokens: 1, UsageEstimated: true}` with `context.Canceled`.
  Through the router: the error is `context.Canceled`, the reply has Provider/Model of the primary, the
  tokens are kept, and the fallback was never called. (`streamFake` already has `resp` and `err`
  fields; `resp` is returned with `err`.)
- #36: the same, but no text event is sent and the error is a 503 text. The fallback answers. The returned
  reply is the fallback's alone (its tokens, its model; `UsageEstimated` false).
- An extra case: a transport error with a zero reply stays zero (no Provider/Model stamped).

**Steps**
- [ ] Write the tests. Run `go test -race ./llm -run 'Estimate|TestStream'`: the new ones fail.
- [ ] Implement. Run again: pass. Run `go test -race ./llm` (whole package, the rename touches many files).
- [ ] `go vet ./...`. Commit: `feat(llm): UsageEstimated, token guess, stamp billed failures`.

---

### Task 3: Anthropic partial usage

Money. Model: Sonnet. Review: **Opus**.

**Files:**
- Modify: `llm/anthropic.go` (`complete` ~line 181, `send` ~line 203, `stream` ~line 221)
- Modify: `llm/anthropic_stream_test.go`

**Interfaces:**
- Consumes: `estimateTokens`, `Response.UsageEstimated` (Task 2).
- Produces: nothing new for later tasks.

**Exact logic**

Signatures become:

```go
func (a *anthropicProvider) send(ctx, params, onEvent) (msg *anthropic.Message, sawDelta bool, err error)
func (a *anthropicProvider) stream(ctx, params, onEvent) (msg *anthropic.Message, sawDelta bool, err error)
func anthropicPartial(msg *anthropic.Message, sawDelta bool) Response
```

`stream`:
- Keep a counter `n` of events that `Accumulate` took without error, and `sawDelta`, set to true when an
  event with `ev.Type == "message_delta"` was accumulated.
- On **every** error return (the `Accumulate` error, the `ctx.Err()` inside the event loop, `s.Err()` after
  the loop, `ctx.Err()` after the loop, the missing `message_stop`): return `&msg` if `n > 0`, else `nil`,
  together with `sawDelta` and the **same error as today**.
- The success return is unchanged (`&msg, sawDelta, nil`).

`send`: the non-streaming `Messages.New` branch returns `msg, false, err` when `err == nil`, and
`nil, false, err` when `err != nil` (never a partial from the non-streaming path). The stream branches pass
`stream`'s three values through.

`complete`:
```text
msg, sawDelta, err := a.send(...)
if err != nil {
    if msg != nil { return anthropicPartial(msg, sawDelta), err }
    return Response{}, err
}
return anthropicResponse(model, msg)
```

`anthropicPartial`, in this order:
1. `r := anthropicUsage(msg.Usage)`. Input, cache read and cache write come from `message_start` and are real.
2. `r.Text` = the text of every text block, joined (the same loop as `anthropicResponse`).
3. If `!sawDelta`: the output count from `message_start` is a placeholder. Set
   `r.OutputTokens = estimateTokens(all)` where `all` joins, for every block in `msg.Content`:
   `block.Text + block.Thinking + string(block.Input)`. Set `r.UsageEstimated = true`.
4. If `sawDelta`: keep `msg.Usage.OutputTokens` (the delta's real count). `UsageEstimated` stays false.
5. `ToolCalls` nil, `StopReason` "".

A partial message that holds a refusal stop reason never reaches this path: a refusal arrives with
`message_delta` and `message_stop`, so `stream` returns success.

**Tests** (use the existing helpers `anthropicServer`, `msgStart`, `blockStart`, `blockDelta`, `msgEnd`,
`quietServer`)
- #26 cancel from `OnEvent` after the second text delta. `msg_start` usage has input 100, cache_read 20,
  cache_creation 5, output 1. Expect `errors.Is(err, context.Canceled)`, input 100, cached 20, cache write 5,
  output = `estimateTokens` of the two deltas joined (not 1), `UsageEstimated` true, `Text` = the two deltas.
- #27 a `message_delta` with `output_tokens: 42` arrives, then the stream is cancelled before
  `message_stop`. Expect output 42, `UsageEstimated` false.
- #28 the server sends `message_start` and one delta, then holds the connection; the test cancels ctx from
  another goroutine after the first `OnEvent` call (signal through a channel). This is the `s.Err()` path.
  Expect the same kind of partial usage as #26.
- #29 the body ends after `message_start` and a delta, with no `message_stop`. Expect the existing
  `io.ErrUnexpectedEOF` error and partial usage.
- #33 (Anthropic) cancel before any event: zero `Response`.
- A thinking block plus a tool_use block with partial input, then cancel: output is guessed over thinking
  text + input JSON + text.
- Check and update the existing tests that assert a zero `Response` on an early end
  (`TestAnthropicStreamAbruptCloseIsError`, `TestAnthropicStreamMidStreamCancellation`,
  `TestAnthropicStreamEndedEarlyKeepsSentEvents`, `TestAnthropicStreamCancelledFromOnEvent`, and any other
  that fails). Change only the `Response` assertions, never the error assertions.

Follow-up 6, Anthropic part (same file):
- Cancel from `OnEvent` right after a `StreamToolReady` event: the stream stops, the error is
  `context.Canceled`, no further events.
- `max_tokens` hit in the middle of a tool input, with `ToolEvents: ToolEventsStartReady`: a start event, no
  ready event, `StopTruncated`, and the call's `Input` is `{}`.
- In `TestAnthropicStreamToolEventsOneCall` and `...ParallelCalls`, also check each returned call's `Input`
  JSON.

**Steps**
- [ ] Write the new tests. Run `go test -race ./llm -run 'TestAnthropic'`: the new ones fail.
- [ ] Implement. Fix the old assertions as described. Run again: pass.
- [ ] `go vet ./...`. Commit: `feat(llm): Anthropic stream returns partial usage on early end`.

---

### Task 4: Gemini partial usage

Money. Model: Sonnet. Review: **Opus**.

**Files:**
- Modify: `llm/google.go` (`complete` ~line 361, `sendStream` ~line 535–645)
- Modify: `llm/google_stream_test.go`

**Interfaces:**
- Consumes: `estimateTokens`, `estimateRequestTokens`, `Response.UsageEstimated` (Task 2).

**Exact logic**

`sendStream` gets one more parameter, `inputGuess func() int`. `complete` passes
`func() int { return estimateRequestTokens(req) }`. It is called only on the early-end path.

New loop state: `thought strings.Builder` (text of parts with `p.Thought`; today these are skipped, keep
skipping them for events and for `Text`), and `seen bool`, set to true after a payload decodes **and** has
no `ev.Error`.

Every early-end return inside the loop (EOF with ctx ended; EOF with no `finishReason` and no block; read
error; decode error; `ev.Error`; the two `ctx.Err()` checks after `onEvent`) returns
`partial(), res.StatusCode, <same error as today>`, where `partial()` is a closure:

```text
if !seen { return Response{} }
r := Response{Text: text.String()}
if usage != zero {
    r.CachedTokens = usage.CachedContentTokenCount
    r.InputTokens  = usage.PromptTokenCount - usage.CachedContentTokenCount
} else {
    r.InputTokens = inputGuess(); r.UsageEstimated = true
}
if usage.CandidatesTokenCount > 0 {
    r.OutputTokens = usage.CandidatesTokenCount + usage.ThoughtsTokenCount
} else {
    args := for each call in calls: json.Marshal(call.FunctionCall.Args) (skip on error)
    r.OutputTokens = estimateTokens(text.String() + thought.String() + args joined)
    r.UsageEstimated = true
}
return r
```

The decode error and `ev.Error` on the current payload: `seen` reflects earlier payloads only, so a first
payload that fails gives a zero reply.

`complete`'s thinking-config loop already returns `resp, err` for any status other than 400. A stream's
status is 200, so the partial reply reaches the router. No change there; add a test (below).

**Tests** (use `gemStreamFake`, `gemText`, `gemFinish`, `gemCall`)
- #30 an event with text and `usageMetadata` {prompt 50, cached 10, candidates 7, thoughts 3}, then cancel
  from `OnEvent`. Expect input 40, cached 10, output 10, `UsageEstimated` false.
- #31 two text events with no `usageMetadata`, then cancel. Expect input = `estimateRequestTokens(req)`,
  output = guess of the text, `UsageEstimated` true.
- An event with only `promptTokenCount`, a thought part and a text part, then the body ends with no
  `finishReason`: input real, output guessed over text + thought, `UsageEstimated` true, error unchanged.
- #33 (Gemini) cancel before any event: zero `Response`. An `ev.Error` as the first payload: zero
  `Response`.
- With a "flash" model (the thinking-config loop): a partial reply still comes back.
- Update existing tests that assert a zero `Response` on an early end, `Response` assertions only.

Follow-up 4: add `mu sync.Mutex` to `gemStreamFake`. Lock it around every append in the handler and around
every read of `paths`, `alts` and `bodies` in tests (add small getter methods if that is simpler).

Follow-up 6, Gemini part:
- Cancel from `OnEvent` on the `StreamToolReady` event: stops, `context.Canceled`, no more events.
- A refusal with function calls (`finishReason: SAFETY` after a call): the returned tool call IDs are the ones
  sent in events (the ID-overwrite path at the end of `sendStream`).
- In the existing tool tests, also check each call's `Input` JSON.

**Steps**
- [ ] Write the tests. Run `go test -race ./llm -run 'TestGemini'`: the new ones fail.
- [ ] Implement. Run again: pass.
- [ ] `go vet ./...`. Commit: `feat(llm): Gemini stream returns partial usage on early end`.

---

### Task 5: OpenAI-compatible partial usage

Money. Model: Sonnet. Review: **Opus**.

**Files:**
- Modify: `llm/openai.go` (`complete`, `completeStream` ~lines 433–550)
- Modify: `llm/openai_stream_test.go`

**Interfaces:**
- Consumes: `estimateTokens`, `estimateRequestTokens`, `Response.UsageEstimated` (Task 2).

**Exact logic**

`completeStream` gets one more parameter, `inputGuess func() int`. `complete` passes
`func() int { return estimateRequestTokens(req) }`.

New state: `seen bool`, set to true after a chunk decodes **and** has no `chunk.Error`.

Every early-end return in the loop (EOF with ctx ended; EOF before `[DONE]`; read error; decode error;
`chunk.Error`; each `ctx.Err()` check after `onEvent`) returns `partial(), <same error as today>`:

```text
if !seen { return Response{} }
r := Response{Text: text.String()}
if usage != nil {
    r.CachedTokens = usage.PromptTokensDetails.CachedTokens
    r.InputTokens  = usage.PromptTokens - r.CachedTokens
    r.OutputTokens = usage.CompletionTokens
    return r
}
args := every call's Function.Arguments, in index order, joined
r.InputTokens  = inputGuess()
r.OutputTokens = estimateTokens(text.String() + refusal.String() + args)
r.UsageEstimated = true
return r
```

**Tests** (use `oaiStreamFake`, `oaiTextChunk`, `oaiToolChunk`, `eofAfterCancelBody`)
- #32 two text chunks, cancel from `OnEvent`. Expect input = `estimateRequestTokens(req)` of the request
  used, output = guess of the two pieces, `UsageEstimated` true, `Text` = the two pieces.
- A tool chunk with partial arguments, then the body ends before `[DONE]`: output guessed over the
  arguments, `io.ErrUnexpectedEOF` unchanged.
- A usage chunk (prompt 30, cached 10, completion 5) arrives, then the body ends before `[DONE]`: input 20,
  cached 10, output 5, `UsageEstimated` false.
- #33 (OpenAI) cancel before any chunk: zero `Response`. A `chunk.Error` as the first chunk: zero
  `Response`.
- `TestOpenAIStreamEOFAfterCancel` and the other early-end tests: update the `Response` assertions only.

**Steps**
- [ ] Write the tests. Run `go test -race ./llm -run 'TestOpenAI'`: the new ones fail.
- [ ] Implement. Run again: pass.
- [ ] `go vet ./...`. Commit: `feat(llm): OpenAI stream returns partial usage on early end`.

---

### Task 6: `llm.WithUsage`

Money (cost metering). Model: Sonnet. Review: **Opus**.

**Files:**
- Create: `llm/usage.go`, `llm/usage_test.go` (package `llm_test`), `llm/usage_router_test.go`
  (package `llm`)

**Interfaces:**
- Consumes: `hasUsage`, `Response.UsageEstimated` (Task 2); `llmtest` (Task 1); `PriceBook` (existing).
- Produces: `Usage`, `WithUsage`.

**Exact logic** (`Complete` of the wrapper)

```text
start := time.Now()
resp, err := inner.Complete(ctx, req)
if hasUsage(resp) {
    record(ctx, Usage{
        Provider, Model, the four counts: from resp
        Task: req.Task
        CostUSD: book.Cost(resp.Model, resp.InputTokens, resp.OutputTokens, resp.CachedTokens, resp.CacheWriteTokens)
        Estimated: resp.UsageEstimated
        Duration: time.Since(start)
        Err: err
    })
}
return resp, err
```

- `book.Cost` on a nil `*PriceBook` works today (`Rate` checks `b != nil`). Do not add a nil check; add a
  test.
- `WithUsage` panics with `"llm: WithUsage needs a record func"` when `record` is nil. `inner` nil also
  panics (`"llm: WithUsage needs an inner Client"`).
- `Enabled` and `CompareTargets` call `inner`.
- Doc comment, in plain English: when it reports (calls with tokens: success, refusal, schema violation,
  stopped stream); `record` runs on the calling goroutine before `Complete` returns; the `ctx` may already
  be cancelled, so use `context.WithoutCancel(ctx)` to save; the known gap (a failed primary's tokens are
  not reported when a fallback answers); either order with `guard.Wrap` works.

**Tests**
- `usage_test.go` (package `llm_test`, `llmtest` as the inner client): #8 (with
  `llm.NewPriceBook(nil, map[string]llm.Rate{"m": {Input: 1, Output: 2}})`, 1M input and 1M output tokens →
  cost 3.0), #9, #10, #11, #12 (a model from `DefaultPrices`, e.g. `claude-haiku-4-5`), #13 (an `OnEvent`
  on the request still gets the fake's events), #14.
- `Duration` is ≥ 0 and the record is made before `Complete` returns (the test reads its slice right after
  the call, with no wait).
- `usage_router_test.go` (package `llm`): #37 a router over a `streamFake` that sends text, then returns a
  partial reply with `UsageEstimated: true` and `context.Canceled`. `WithUsage` over it records once, with
  `Estimated` true, the primary's Provider/Model, and `errors.Is(u.Err, context.Canceled)`.

**Steps**
- [ ] Write the tests. Run `go test -race ./llm -run 'Usage'`: fail.
- [ ] Implement. Run again: pass.
- [ ] `go vet ./...`. Commit: `feat(llm): WithUsage reports tokens and cost per call`.

---

### Task 7: `llm.RunTools`

Concurrency (the `Parallel` option). Model: Sonnet. Review: **Opus**.

**Files:**
- Create: `llm/tools.go`, `llm/tools_test.go` (package `llm_test`)

**Interfaces:**
- Consumes: `hasUsage` (Task 2), `llmtest` (Task 1).
- Produces: `ToolFunc`, `ToolLoopOptions`, `ToolRun`, `RunTools`, `ErrToolLoopLimit`,
  `ErrToolCallsTruncated`.

**Exact logic**

The loop order is spec part 3, steps 1–2.9. Implement it in that order, with the step numbers in code
comments. Points that are easy to get wrong:

- History: `msgs := append([]Message(nil), req.Messages...)`. Each round sends a request copy with
  `Messages: msgs`. Appending to `msgs` never touches the caller's backing array.
- `MaxRounds <= 0` means 8. The check is `round == maxRounds` **after** the reply was received and found to
  have calls (step 2.8). So `MaxRounds: 2` makes exactly 2 `Complete` calls.
- Step 2.3: add the reply to `Rounds` and set `Response` when `err == nil || hasUsage(resp)`.
- Step 2.7 (`ErrToolCallsTruncated`) comes before 2.8 (`ErrToolLoopLimit`).
- The returned `ToolRun.Messages` is `msgs` at the moment of return. On a normal end it does **not** include
  the final reply as an assistant turn (the caller has it in `Response`; the history is "ready to continue"
  with a new user turn after appending that reply themselves). State this in the doc comment, with one line
  of example: `run.Messages = append(run.Messages, llm.Message{Role: "assistant", Content: run.Response.Text})`.

Running the calls, exact:

```text
results := make([]ToolResult, len(calls))
runOne := func(i int) {
    call := calls[i]
    results[i] = ToolResult{ToolCallID: call.ID}
    fn, ok := tools[call.Name]
    if !ok { results[i].Content = "unknown tool: " + call.Name; results[i].IsError = true; return }
    out, err := fn(ctx, call)
    if err != nil { results[i].Content = err.Error(); results[i].IsError = true; return }
    results[i].Content = out
}
if !opts.Parallel || len(calls) == 1 {
    for i := range calls { runOne(i) }
} else {
    var wg sync.WaitGroup
    for i := range calls { wg.Add(1); go func() { defer wg.Done(); runOne(i) }() }
    wg.Wait()
}
if ctx.Err() != nil { return run, ctx.Err() }   // results not appended
```

Each goroutine writes only its own `results[i]`, so no lock is needed. Do not stop early when one call
fails, and do not cancel the others. A panic in `fn` is not recovered.

Doc comment: the loop in a few lines; `Parallel` needs goroutine-safe functions; `OnEvent` gets text from
every round (the join rule holds per round); call `guard.BeginTurn(ctx)` first when using `guard.Wrap`; with
`WithUsage` every round is reported.

**Tests** (`llmtest` as the client; build tool-call replies with `llm.Response{ToolCalls: ...,
StopReason: llm.StopToolUse}`)
- #15–#22, #24, #25 as in the spec.
- #16 also checks the second request's `Messages` exactly: the input, the assistant turn (text + both
  calls), the user turn with both results in call order.
- #23 two calls with `Parallel: true`. Each function sends on `started` and then waits on `release`. The test
  waits for both `started` signals (with a 5s `select` timeout that fails the test), then closes `release`.
  This proves both run at once with no sleeps. The results are in call order.
- Without `Parallel`, the same two functions must run one after another: the first function's `started` is
  received and the second has not started before `release` is closed (a non-blocking check on `started`).
- `MaxRounds: 0` with 9 tool rounds scripted: `ErrToolLoopLimit` after 8 calls.
- A reply with calls and no tools map entry for any: all results are `IsError` and the loop goes on.

**Steps**
- [ ] Write the tests. Run `go test -race ./llm -run 'RunTools'`: fail.
- [ ] Implement. Run again with `-count=20` for the parallel tests: pass.
- [ ] `go vet ./...`. Commit: `feat(llm): RunTools tool loop`.

---

### Task 8: Retry test timeout (follow-up 8)

Tiny. Model: Haiku, or the controller does it inline. No review.

**Files:**
- Modify: `llm/retry_test.go` (`TestRetryDeadlineTooClose`, the "backoff past deadline" case)

**What to do**
- Change the case's timeout from `100 * time.Millisecond` to `time.Second`. The fake `now` (10ms before the
  deadline) already makes the check fixed, so nothing else changes.
- Run `go test -race ./llm -run TestRetryDeadlineTooClose -count=5`: pass.
- Commit: `test(llm): retry deadline test uses a 1s timeout`.

---

### Task 9: README, memory note, final gates

Docs. Model: Haiku. No per-task review (the final review covers it).

**Files:**
- Modify: `README.md`

**What to write** (plain English, lines wrapped at about 100 characters, like the rest of the README)

1. Under `### Streaming`, at its end: a short paragraph. A stream that is stopped or breaks after data
   arrived returns its token counts with the error. Counts the provider had not sent yet are guessed (about 4
   characters per token), and `Response.UsageEstimated` says so. Before any data arrives, the reply is zero
   as before. **Behavior change** since v0.9.0.
2. A new `### Recording usage` section after `### Streaming`: a 10–15 line `llm.WithUsage` example that saves
   with `context.WithoutCancel(ctx)`; what gets reported; the known fallback gap. Replace the sentence in the
   intro "wrap the `Client` to record them" with "use `llm.WithUsage`, or wrap the `Client` yourself".
3. Under `### Tool calling`, at its end: a `#### Tool loop helper` part with a short `llm.RunTools` example,
   the rules (tool error → error result, unknown tool, truncated round → `ErrToolCallsTruncated`, the round
   limit → `ErrToolLoopLimit`, `Parallel`), and a note that the hand-written loop above is still fine for
   callers that need full control.
4. A new `### Testing with llmtest` section (before `## aikit/decide`): a short example and the event rules
   in two or three lines.

Then:
- [ ] Full gates: `go vet ./...` and `go test -race ./...`. Both must pass. Paste the summary lines into the
  task report.
- [ ] Commit: `docs: helpers and stop-billing in the llm README`.

After merge (controller, not the implementer): update the memory note `streaming-deferred-followups` (remove
items 4, 6, 7, 8) and `next-subprojects-order` (helpers merged; next is Embed). Releasing **v0.10.0** (tag and
push) is the owner's call: ask first.

---

## Review guidance (for the controller)

- Task 1: low risk. The controller reads the diff.
- Task 2: Opus per-task review. Point the reviewer at `completeOn`'s stamping condition, `estimateRequestTokens`'
  field list, and that #36 proves no fallback usage is merged.
- Tasks 3–5: Opus per-task review each. Point the reviewer at: every early-end return carries the partial
  reply; the error value is unchanged; `seen`/`n` is the only gate for "data arrived"; the
  `message_start` placeholder output is never reported as real; only `Response` assertions changed in old
  tests.
- Task 6: Opus per-task review (cost). Check that a zero reply is never recorded and that `record` runs
  before `Complete` returns.
- Task 7: Opus per-task review (concurrency). Check `results[i]` writes, no early cancel, the round-limit
  order, and that the caller's slice is untouched.
- Task 8: none. Task 9: none.
- Small fix rounds: the controller reads the fix diff (no new reviewer).
- Final whole-branch review: Opus, once.

## Spec coverage check

| Spec item | Task |
| --- | --- |
| Part 1: `llmtest` (replies, `Requests`, `ErrNoReplies`, events, `Reply.Events`, goroutine-safe), tests #1–#7 | 1 |
| Part 2: `WithUsage`, `Usage`, report only with tokens, nil book, panics, doc notes, tests #8–#14, #37 | 6 |
| Part 3: `RunTools` loop order, call running, `Parallel`, errors, doc notes, tests #15–#25 | 7 |
| Part 4: `Response.UsageEstimated`, `estimateTokens`, request guess, #34 | 2 |
| Part 4: router stamps any failed reply with tokens; fallback rules unchanged, #35, #36 | 2 |
| Part 4: Anthropic table row, #26–#29, #33 | 3 |
| Part 4: Gemini table row, #30, #31, #33 | 4 |
| Part 4: OpenAI-compatible table row, #32, #33 | 5 |
| Part 4: partial reply holds Text + counts, ToolCalls nil, StopReason "" | 3, 4, 5 (Global Constraints) |
| Part 4: release note, v0.10.0 | 9 (README); tag is the owner's call |
| Follow-up 4 (gemStreamFake mutex) | 4 |
| Follow-up 6 (tool-event test gaps, `evRec.desc`) | 2 (`desc`), 3, 4 |
| Follow-up 7 (mutation test, `eq` rename) | 2 |
| Follow-up 8 (retry test timeout) | 8 |
| README sections | 9 |
| Gates | 9 |
