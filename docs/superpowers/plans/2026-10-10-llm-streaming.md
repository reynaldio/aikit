# LLM Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream a reply's text to the caller piece by piece through a new `Request.OnEvent` callback, on all three providers, with the router's fallback and `guard.Wrap`'s refusal-marker strip still correct.

**Architecture:** `OnEvent` is an optional field on `llm.Request`; `Client` and `Complete` are unchanged. The router wraps the callback in a tracker that knows whether any text has reached the caller, and uses that to decide whether fallback is still allowed. Each provider streams text to the callback and builds the same `Response` it builds today. `guard` puts a marker filter between the router and the caller for conversational calls.

**Tech Stack:** Go 1.25, `github.com/anthropics/anthropic-sdk-go` v1.56.0 (already a dependency), `net/http` + a small hand-written server-sent events reader for OpenAI-compatible and Gemini. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-10-10-llm-streaming-design.md](../specs/2026-10-10-llm-streaming-design.md). Read it before your task; this plan does not repeat its reasoning.

## Global Constraints

- Branch: `feat/llm-streaming` (already checked out). Commit per task. End every commit message with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- No new module dependencies. `go.mod` stays as it is.
- `OnEvent == nil` must behave exactly as today. Every existing test passes unchanged, except for the
  mechanical `newOpenAI` signature update in Task 3.
- `OnEvent` is called on the goroutine that called `Complete`, in order, never concurrently, never after
  `Complete` returns. **No provider or filter may start a goroutine to call it.**
- Providers never send an event with empty `Text`.
- **Join rule:** when `Complete` returns `err == nil`, the `Text` of all `StreamText` events, joined in
  order, equals `resp.Text` exactly. This holds with and without `guard.Wrap`.
- On error, events already sent stay sent; the returned `Response` is what today's code returns (refusals
  carry partial text and usage).
- Log lines, exact text:
  - router: `llm: stream failed after text was sent; not falling back`
  - OpenAI-compatible: `llm: stream reply had no usage; tokens reported as 0`
- Timeouts do not change: the caller's `ctx` deadline, else `withDefaultDeadline`. A stream that stalls ends in
  `context.DeadlineExceeded` / `context.Canceled`, never a partial success.
- Match the surrounding code's comment density and idiom (doc comments explain *why*; tests are table-free
  unless the neighbouring tests use tables).

## Decisions

From the spec (owner answers, not open for change):

| Question | Decision |
| --- | --- |
| First user | Chat UI with tool rounds |
| Providers in v1 | Anthropic, OpenAI-compatible, Gemini |
| API shape | `Request.OnEvent func(StreamEvent)`; no new method |
| Callback type | `func(StreamEvent)` with a `Kind`; only `StreamText` in v1 |
| `JSONSchema` set, or provider can't stream | Normal call; on success send the whole text as one event |
| Error after text was sent | No fallback; return the error |

Clarifications made while planning (from reading the code; none changes the spec's behaviour):

1. **"Has a schema" means `len(req.JSONSchema) > 0`**, not `!= nil`. The router (`canServe`,
   `checkJSONReply`) already treats an empty map as "no schema"; streaming uses the same test so the two
   never disagree.
2. **"Provider can't stream" is an internal capability check.** A provider streams only if it implements
   `streams() bool` and returns true. Each provider gets the method in the task that adds its streaming.
   Until then the router's rule 7 covers it. The test `fakeProvider` never implements it.
3. **Rule 7 lives in `completeOn`**, not `Complete`, because it applies per call: a primary that streams can
   fall back to one that doesn't, and the reverse.
4. **The explicit `req.Model` path streams too**, through the same tracker. It never falls back today, so
   rules 5–6 don't apply to it and it logs nothing.
5. **OpenAI provider gets a logger** for the no-usage warning: `newOpenAI` takes a `*slog.Logger` as a fourth
   parameter. `New` resolves its logger before building providers. A nil logger means no log.
6. **Gemini `thought: true` parts are left out of `resp.Text` on both paths.** aikit never asks for thought
   text (`includeThoughts` is never set), so today's non-streamed text doesn't change in practice. Leaving
   them out on both paths keeps the streamed and non-streamed text the same.
7. **A Gemini stream with no `finishReason` is an error, except a prompt-level block.** A request blocked
   before generation has no candidates, only `promptFeedback.blockReason`. That must still become
   `ErrRefused`, as it does today.
8. **Marker filter assumption:** the marker does not overlap itself, and nested constructions (a marker
   assembled from pieces around another marker) are out of scope. `StripMarker` is single-pass, so nobody
   relies on those cases today. Filter tests use natural replies.
9. **Robustness beyond the spec, in the marker filter:** it also holds back an incomplete UTF-8 sequence at the
   end of the ready text. Real providers deliver valid UTF-8 pieces, but this lets the split-at-every-position
   tests split at every *byte* without the filter ever emitting half a rune.

## File map

| File | Responsibility | Task |
| --- | --- | --- |
| `llm/llm.go` | `StreamKind`, `StreamEvent`, `Request.OnEvent` + godoc; `streamer`, `canStream`, `streamTracker`; router rules 1–7 | 1 |
| `llm/stream_test.go` (new) | Router streaming tests and the `streamFake` provider | 1 |
| `llm/anthropic.go` | `OnEvent` forces the stream path; text deltas to `OnEvent`; `streams()` | 2 |
| `llm/anthropic_stream_test.go` | Anthropic streaming tests (reuse existing fake server and SSE helpers) | 2 |
| `llm/sse.go` (new) | Server-sent events reader | 3 |
| `llm/sse_test.go` (new) | SSE reader tests | 3 |
| `llm/openai.go` | Streaming request, chunk parsing, tool-call joining, usage, shared response builder; `streams()`; logger | 3 |
| `llm/openai_stream_test.go` (new) | OpenAI-compatible streaming tests | 3 |
| `llm/timeout_test.go`, `llm/schema_test.go` | Update `newOpenAI` call sites (add `nil` logger) | 3 |
| `llm/google.go` | `geminiPart.Thought`; `streamGenerateContent` path; shared response builder; `streams()` | 4 |
| `llm/google_stream_test.go` (new) | Gemini streaming tests | 4 |
| `guard/stream.go` (new) | Marker filter | 5 |
| `guard/stream_test.go` (new) | Filter unit tests | 5 |
| `guard/guard.go` | Wrap `OnEvent` with the filter for conversational calls | 5 |
| `guard/guard_test.go` | `fakeLLM` emits pieces; Wrap-level streaming tests | 5 |
| `README.md` | New "Streaming" section | 6 |

## Shared interfaces

Every task's implementer sees only their own task. These are the exact names they share.

**Public (`llm/llm.go`, Task 1):**

```go
type StreamKind string

const StreamText StreamKind = "text"

type StreamEvent struct {
    Kind StreamKind
    Text string // the new piece of text (for StreamText)
}

// on Request:
OnEvent func(StreamEvent) // nil = no streaming
```

**Internal (`llm/llm.go`, Task 1):**

```go
// streamer is implemented by a provider that can send text to Request.OnEvent as it arrives.
type streamer interface{ streams() bool }

func canStream(p provider) bool // p implements streamer and streams() returns true

type streamTracker struct {
    next func(StreamEvent) // the caller's OnEvent
    sent bool              // a StreamText event with non-empty Text has been passed on
}
func (t *streamTracker) emit(ev StreamEvent) // passes every event on; sets sent on non-empty StreamText

func (r *router) complete(ctx context.Context, req Request, t *streamTracker) (Response, error)
// today's Complete body, plus rule 6; t is nil when not streaming
```

**Provider contract (Tasks 2–4):** when `req.OnEvent != nil` and the provider was given it (the router only
passes it to providers where `canStream` is true), the provider:

- calls `req.OnEvent(StreamEvent{Kind: StreamText, Text: piece})` for each non-empty text piece, in order, on the
  calling goroutine;
- returns a `Response` whose `Text` is exactly those pieces joined (on success *and* on refusal);
- returns errors exactly as its non-streamed path would for the same situation (refusal → `*RefusalError` with
  populated `Response`; transport → zero `Response`);
- returns a wrapped `io.ErrUnexpectedEOF` when the body ends before the end signal (unless `ctx` has ended, then
  the context error), so failover classifies it as transient, as on Anthropic today.

**SSE reader (`llm/sse.go`, Task 3; also used by Task 4):**

```go
type sseReader struct{ /* bufio.Scanner over the body */ }
func newSSEReader(r io.Reader) *sseReader
// next returns the next event's data payload. io.EOF at a clean end of body.
// Any other error is the read error, unwrapped (callers pass it through readError).
func (s *sseReader) next() ([]byte, error)
```

**OpenAI (`llm/openai.go`, Task 3):**

```go
func newOpenAI(apiKey, baseURL string, timeout time.Duration, log *slog.Logger) provider
// new oaiRequest fields:
Stream        bool              `json:"stream,omitempty"`
StreamOptions *oaiStreamOptions `json:"stream_options,omitempty"`
type oaiStreamOptions struct{ IncludeUsage bool `json:"include_usage"` }
type oaiUsage struct { // extracted from oaiResponse.Usage, same JSON tags
    PromptTokens, CompletionTokens int; PromptTokensDetails struct{ CachedTokens int }
}
// Shared tail of both paths: token split, tool calls, stop reason, refusal.
func oaiBuildResponse(model string, hasChoice bool, text, refusal, finishReason string,
    calls []oaiToolCall, u oaiUsage) (Response, error)
func (o *openaiProvider) completeStream(ctx context.Context, model string, oaiReq oaiRequest,
    onEvent func(StreamEvent)) (Response, error)
```

**Gemini (`llm/google.go`, Task 4):**

```go
// on geminiPart:
Thought bool `json:"thought,omitempty"`
// Shared tail of both paths: text (non-thought parts), tool calls, stop reason, usage, refusal.
func geminiBuildResponse(model string, out geminiResponse) (Response, error)
func (g *googleProvider) sendStream(ctx context.Context, model string, body geminiRequest,
    onEvent func(StreamEvent)) (Response, int, error) // same contract as send: status for the 400 retry
```

**Anthropic (`llm/anthropic.go`, Task 2):**

```go
func (a *anthropicProvider) send(ctx context.Context, params anthropic.MessageNewParams,
    onEvent func(StreamEvent)) (*anthropic.Message, error)
func (a *anthropicProvider) stream(ctx context.Context, params anthropic.MessageNewParams,
    onEvent func(StreamEvent)) (*anthropic.Message, error)
```

**Guard (`guard/stream.go`, Task 5):**

```go
type markerFilter struct {
    marker   string
    next     func(llm.StreamEvent)
    buf      string          // pending text not yet sent
    blanks   string          // leading blanks held until the first non-blank text
    started  bool            // the first non-blank text has been sent
    declined bool            // a marker was removed
    out      strings.Builder // everything sent, joined
}
func newMarkerFilter(marker string, next func(llm.StreamEvent)) *markerFilter
func (f *markerFilter) push(ev llm.StreamEvent) // used as the inner request's OnEvent
func (f *markerFilter) finish()                 // call once, only when the inner Complete succeeded
func (f *markerFilter) text() string            // f.out.String()
```

## Model and review per task

| Task | Implement | Per-task review |
| --- | --- | --- |
| 1 Router | Sonnet | Sonnet (required: decides whether a user sees a broken reply) |
| 2 Anthropic | Sonnet | Sonnet |
| 3 SSE + OpenAI | Sonnet | Sonnet |
| 4 Gemini | Sonnet | Sonnet |
| 5 Guard filter | Sonnet | **Opus** (guard / security package) |
| 6 README + gates | Haiku | none (final review covers it) |
| Final whole-branch review | — | Opus, once |

Order: 1 first. 2, 3, 4 and 5 each depend only on Task 1 and may run in any order. 4 reuses `sse.go` from 3,
so run 3 before 4. 6 last.

---

### Task 1: API and router rules

**Files:**
- Modify: `llm/llm.go` (types near `Request`; `Complete` → `complete`; `completeOn`)
- Create: `llm/stream_test.go`

**Interfaces:**
- Consumes: nothing new.
- Produces: `StreamKind`, `StreamText`, `StreamEvent`, `Request.OnEvent`, `streamer`, `canStream`,
  `streamTracker`, `router.complete` (see Shared interfaces).

**Godoc for `Request.OnEvent`** must state the five rules from the spec ("Rules for `OnEvent`"): calling
goroutine/order/never after return; quick, never blocks; only `StreamText` in v1, ignore unknown kinds; the join
rule; events stay sent on error. Also: with `JSONSchema` set, the text arrives as one event after the reply is
checked.

**Router logic — exact order. Implement exactly this; do not reorder.**

`Complete(ctx, req)`:

```text
1. if req.OnEvent == nil:
       return r.complete(ctx, req, nil)                 // today's path, byte-for-byte the same behaviour

2. if len(req.JSONSchema) > 0:
       onEvent := req.OnEvent
       req.OnEvent = nil                                 // providers and fallback never see it
       resp, err := r.complete(ctx, req, nil)            // today's path incl. schema-aware fallback
       if err == nil && resp.Text != "":
           onEvent(StreamEvent{Kind: StreamText, Text: resp.Text})   // resp.Text is post-checkJSONReply (unfenced)
       return resp, err                                  // on any error: no event

3. t := &streamTracker{next: req.OnEvent}
   req.OnEvent = t.emit
   return r.complete(ctx, req, t)
```

`streamTracker.emit(ev)`:

```text
if ev.Kind == StreamText && ev.Text != "": t.sent = true
t.next(ev)
```

`complete(ctx, req, t)` is today's `Complete` body with **one** insertion (rule 6), placed right after the primary
call and before `shouldFailover`:

```text
if req.Model != nil && !req.Model.empty():
    return r.completeOn(ctx, *req.Model, req)          // unchanged; no fallback, no log
ref, prof := r.resolve(req)
if ref.empty(): return Response{}, ErrNotConfigured
resp, err := r.completeOn(ctx, ref, req)               // rule 3/4: primary gets the tracker via req.OnEvent
if err == nil: return resp, nil                        // rule 4
if t != nil && t.sent:                                 // rule 6 — NEW
    if r.log != nil:
        r.log.Warn("llm: stream failed after text was sent; not falling back",
            "err", err, "profile", string(prof), "model", string(ref.Provider)+"/"+ref.Model)
    return resp, err                                   // as returned; no fallback
if !shouldFailover(err): return resp, err              // rule 5 from here: today's logic, unchanged
if req.NoFallback: return resp, err
fb := ... (today's configured-fallback / fallbackRef selection, unchanged)
if !fb.empty():
    if resp2, err2 := r.completeOn(ctx, fb, req); err2 == nil:   // same req ⇒ same tracker
        (today's "answered via fallback model" warn log, unchanged)
        return resp2, nil
return resp, err                                       // unchanged: primary's error
```

Note: in rule 5 the tracker carries over to the fallback. `sent == false` at that point, so only the fallback's
text ever reaches the caller.

`completeOn(ctx, ref, req)`, rule 7:

```text
p, ok := r.providers[ref.Provider]; if !ok: return Response{}, ErrNotConfigured   // unchanged
onEvent := req.OnEvent
synth := onEvent != nil && !canStream(p)
if synth: req.OnEvent = nil                            // provider sees a plain request
... maxTokens, p.complete, refusal stamping — unchanged ...
resp, err = checkJSONReply(resp, req)                  // unchanged, last step today
if err == nil && synth && resp.Text != "":
    onEvent(StreamEvent{Kind: StreamText, Text: resp.Text})   // onEvent is t.emit ⇒ sent becomes true
return resp, err
```

On the error branch inside `completeOn` (refusal stamping), no event is sent.

- [ ] **Step 1: Write the failing tests** in `llm/stream_test.go`. Add a test provider:
  `streamFake{pieces []string; err error; resp Response; calls int}` that implements `provider` *and*
  `streamer` (returns true). `complete` sends each piece through `req.OnEvent` (if non-nil), then returns
  `err` with `resp` (with `resp.Text = strings.Join(pieces, "")` unless the test set `resp.Text`), or success
  with the joined text. Reuse `newTestRouter`'s profile layout (chat → google/flash, fallback →
  anthropic/haiku) with `streamFake`s in place of `fakeProvider`s. Record events in a slice.
  Test cases:
  1. **Error before any text → fallback used.** Primary: no pieces, err `"503 overloaded"`. Fallback: pieces
     `"Hel","lo"`. Expect success, events `["Hel","lo"]`, joined == `resp.Text`, `resp.Model == "haiku"`.
  2. **Error after some text → no fallback.** Primary: pieces `"Par"`, err `"503 overloaded"`. Expect that error,
     events `["Par"]`, fallback `calls == 0`. With a `slog` logger writing to a buffer, the buffer contains
     `llm: stream failed after text was sent; not falling back`.
  3. **Refusal midway → no fallback.** Primary: pieces `"I can"`, err `&RefusalError{...}`, resp with
     `OutputTokens: 5`. Expect `errors.Is(err, ErrRefused)`, `resp.Text == "I can"`, `resp.Model == "flash"`
     (stamped), fallback not called.
  4. **Refusal before text → fallback** (rule 5 includes refusals). Primary: no pieces, refusal. Fallback
     answers. Expect success with only the fallback's events.
  5. **Empty-text events don't count as sent.** Primary sends a piece `""` (deliberately, test only) then
     fails with 503. Expect fallback used.
  6. **`JSONSchema` set → exactly one event on success.** The chat primary (google/flash; Google counts as
     schema-enforcing in `supportsJSONSchema`) is a `streamFake` with pieces `"```json\n{\"a\":1}\n```"`. The provider must receive `req.OnEvent == nil`
     (assert inside the fake), so no pieces are sent. Expect exactly one event `{"a":1}` (unfenced) ==
     `resp.Text`.
  7. **`JSONSchema` set, reply not JSON → no event.** Expect `ErrSchemaViolation` and zero events.
  8. **Provider that can't stream → one event.** Plain `fakeProvider` (no `streamer`) as primary. Expect one
     event `"ok:flash"`; the fake received `OnEvent == nil` (add a `lastReq Request` field to `fakeProvider` if
     needed, without changing its existing behaviour).
  9. **Streaming primary falls back to a non-streaming provider.** Primary `streamFake` fails before text,
     fallback is a `fakeProvider` → one event with the fallback's full text.
  10. **Explicit `req.Model` streams and never falls back.** Pieces arrive; on error after text, no fallback and
      no warn log.
  11. **Caller's request is not mutated.** After `Complete`, the caller's `req.OnEvent` is still their own
      function (compare by calling it, or check a counter), not the tracker.
- [ ] **Step 2: Run them to see them fail.** `go test ./llm -run 'Stream' -v` → compile errors
  (`StreamEvent` undefined).
- [ ] **Step 3: Implement** the types, godoc, `streamer`/`canStream`/`streamTracker`, and the router logic above.
- [ ] **Step 4: Run the tests.** `go test ./llm -run 'Stream' -v` → PASS. Then `go test ./llm` → all existing
  tests PASS unchanged (rule 1: `OnEvent == nil` behaves as today).
- [ ] **Step 5: Commit.** `git add llm/llm.go llm/stream_test.go && git commit -m "feat(llm): OnEvent streaming API and router fallback rules"`

---

### Task 2: Anthropic text deltas

**Files:**
- Modify: `llm/anthropic.go` (`complete`, `send`, `stream`; add `streams()`)
- Test: `llm/anthropic_stream_test.go` (reuse `anthropicServer`, `sseEvent`, `claudeSSE`, `blockStart`,
  `blockDelta`, `blockStop`, `msgStart`, `msgEnd`)

**Interfaces:**
- Consumes: `StreamEvent`, `StreamText`, `Request.OnEvent`, `streamer` (Task 1).
- Produces: `func (a *anthropicProvider) streams() bool { return true }`; `send`/`stream` take
  `onEvent func(StreamEvent)` (nil = today's behaviour).

**What to build:**
- `complete` passes `req.OnEvent` to `send`.
- `send`: take the stream path when `onEvent != nil` **or** `CalculateNonStreamingTimeout` errors (today's
  rule). Otherwise `Messages.New` as today.
- `stream`: inside the loop, **after** `msg.Accumulate(ev)` succeeds, if `onEvent != nil` and
  `ev.Type == "content_block_delta"` and `ev.Delta.Type == "text_delta"` and `ev.Delta.Text != ""`, call
  `onEvent(StreamEvent{Kind: StreamText, Text: ev.Delta.Text})`. Nothing else is sent (no `thinking_delta`,
  `input_json_delta`, `citations_delta`, `signature_delta`).
- Keep `repairToolInputs`, the "ended before `message_stop`" error, the stop-on-`message_stop` return, and
  `anthropicResponse` exactly as they are. `anthropicResponse` joins text blocks with no separator, and
  `Accumulate` appends each `text_delta` to its block, so the join rule holds without changes.

- [ ] **Step 1: Write the failing tests:**
  1. **`OnEvent` forces streaming for a small request.** `MaxTokens: 100` with `OnEvent` set → the request body
     has `"stream": true`. (The existing `TestAnthropicSmallRequestIsNotStreamed`, without `OnEvent`, still passes.)
  2. **Text pieces arrive in order; joined == `resp.Text`.** `claudeSSE("end_turn", "", "Hel", "lo ", "world")` →
     events `["Hel","lo ","world"]`, `resp.Text == "Hello world"`.
  3. **Only text is sent.** Reuse the stream from `TestAnthropicStreamedToolUseThinkingAndWebSearch` with
     `OnEvent`: events contain only the text-block deltas; tool call comes back whole in `resp.ToolCalls`; joined ==
     `resp.Text`.
  4. **Two text blocks** (text, tool_use, text) → joined events == `resp.Text` (no separator added).
  5. **Refusal midway** → `ErrRefused`, events == the streamed deltas, `resp.Text` == their join, usage populated.
  6. **Stream ends before `message_stop`** (`abort: true`) → error, events already sent stay, no panic.
  7. **`ctx` cancelled midway.** The server writes `message_start` and one text delta, flushes, then waits for the
     request context to end. `OnEvent` cancels `ctx` on the first event. Expect `errors.Is(err, context.Canceled)`
     and exactly one event.
  8. **Stream goes quiet past the deadline.** Same server, `ctx` with a 200 ms timeout, no cancel in `OnEvent` →
     `errors.Is(err, context.DeadlineExceeded)`.
- [ ] **Step 2: Run them to see them fail.** `go test ./llm -run 'Anthropic' -v` → new tests FAIL (no events /
  `stream` not true).
- [ ] **Step 3: Implement** as described.
- [ ] **Step 4: Run the tests.** `go test ./llm -run 'Anthropic' -v` → PASS; `go test ./llm` → PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(llm): stream Anthropic text to OnEvent"`

---

### Task 3: SSE reader and OpenAI-compatible streaming

**Files:**
- Create: `llm/sse.go`, `llm/sse_test.go`, `llm/openai_stream_test.go`
- Modify: `llm/openai.go`; `llm/llm.go` (`New`: resolve logger first, pass to `newOpenAI`)
- Modify (mechanical): every `newOpenAI(` call in `llm/*_test.go` gets a fourth argument `nil`
  (`grep -n "newOpenAI(" llm/*_test.go`)

**Interfaces:**
- Consumes: Task 1 types; `withDefaultDeadline`, `readError`, `oaiToolCalls`, `oaiStopReason`, `oaiRefusal`
  (existing).
- Produces: `sseReader`/`newSSEReader`/`next` (used by Task 4); `oaiUsage`, `oaiBuildResponse`,
  `completeStream`, `oaiStreamOptions`, `streams()`; new `newOpenAI` signature.

**SSE reader rules:**
- Read line by line (`bufio.Scanner`, buffer grown to 8 MiB max, because one chunk can carry a large
  tool-argument piece). Strip a trailing `\r` (CRLF bodies).
- `data:` line → append the value (strip **one** leading space if present). Several `data:` lines in one event
  are joined with `\n`.
- Blank line → end of event. If it had any `data:` line, return the payload; otherwise keep reading.
- Lines starting with `:` (comments), `event:`, `id:`, `retry:` → ignored.
- End of body with a pending payload → return it, then `io.EOF` on the next call. Clean end → `io.EOF`.
- A read error → returned as is (the caller wraps it with `readError(ctx, ...)` so a deadline reads as
  `context.DeadlineExceeded`).

**OpenAI streaming — what to build:**
- Refactor first, no behaviour change: name the usage struct `oaiUsage`, and move the tail of `complete`
  (cached/input split, `oaiToolCalls`, `oaiStopReason`, refusal check) into `oaiBuildResponse`. Existing tests
  must still pass after this step alone.
- `complete`: when `req.OnEvent != nil`, set `Stream: true`,
  `StreamOptions: &oaiStreamOptions{IncludeUsage: true}` and call `completeStream`. Otherwise today's path.
  `JSONSchema` requests never arrive here with `OnEvent` set (router rule 2), but `completeStream` must not
  depend on that.
- `completeStream`:
  - Same deadline (`withDefaultDeadline`), headers and URL as today.
  - Status ≥ 300 → read the whole body and return the same error as today (`openai: <msg> (status N)`), zero
    `Response`. No events.
  - Read events with `sseReader`. Payload `[DONE]` → end. Otherwise decode into a chunk type:
    `choices[].{index, delta{content, refusal, tool_calls[]{index, id, type, function{name, arguments}}}, finish_reason}`,
    `usage` (pointer, may be null), `error{message}` (pointer).
  - `error` present → return `fmt.Errorf("openai: %s (stream)", msg)`, zero `Response`.
  - Only `choices` with `index == 0` are used.
  - `delta.content` non-empty → append to text, call `onEvent`.
  - `delta.refusal` → append to a refusal buffer (not sent).
  - `delta.tool_calls[]` → per `index`: if `id` non-empty set it, if `function.name` non-empty set it, append
    `function.arguments`. At the end, order by `index` ascending into `[]oaiToolCall` (empty arguments become
    `{}` through `oaiToolCalls` → `normalizeToolInput`).
  - `finish_reason` non-empty → keep the latest.
  - `usage` non-null → keep the latest. Some backends put it on the last choice chunk, OpenAI on a final chunk
    with empty `choices`.
  - Body ends (`io.EOF`) before `[DONE]` → if `ctx.Err() != nil` return
    `readError(ctx, "openai", ctx.Err())`, else
    `fmt.Errorf("openai: stream ended before [DONE]: %w", io.ErrUnexpectedEOF)`. Read error →
    `readError(ctx, "openai", err)`. Zero `Response` in both cases.
  - At `[DONE]`: if no usage arrived, `o.log.Warn("llm: stream reply had no usage; tokens reported as 0", "model", model)`
    (nil-safe) and leave counts 0. Return
    `oaiBuildResponse(model, sawChoice, text, refusal, finishReason, calls, usage)`.
    A refusal therefore carries the streamed partial text and usage, as on the non-streamed path.
- `streams() bool { return true }`.
- `New`: resolve `lg` (today's nil → discard logic) **before** building providers and pass it to all three
  `newOpenAI` calls.

- [ ] **Step 1: Write the failing SSE tests** (`llm/sse_test.go`):
  1. Two events → two payloads, then `io.EOF`.
  2. Multi-line `data:` → joined with `\n`.
  3. `data:x` (no space) and `data: x` both yield `x`; `data:  x` yields ` x`.
  4. CRLF line endings.
  5. Comments, `event:`, `id:` ignored; an event with only `event:` yields nothing.
  6. Final event without a trailing blank line is still returned.
  7. A 1 MiB `data:` line is returned whole.
  8. A reader that fails midway → that error (not `io.EOF`).
- [ ] **Step 2: Run them to see them fail.** `go test ./llm -run 'SSE' -v` → compile error.
- [ ] **Step 3: Implement `llm/sse.go`.** Run `go test ./llm -run 'SSE' -v` → PASS.
- [ ] **Step 4: Refactor `oaiBuildResponse` / `oaiUsage`, update the `newOpenAI` signature and its call sites.**
  Run `go test ./llm` → PASS (no behaviour change).
- [ ] **Step 5: Write the failing OpenAI streaming tests** (`llm/openai_stream_test.go`). Use a fake
  `/chat/completions` server that records the request body and writes a scripted SSE body (flush after each
  event). Test cases:
  1. **Request shape:** with `OnEvent`, the body has `"stream": true` and `"stream_options": {"include_usage": true}`;
     without `OnEvent`, neither key is present.
  2. **Text pieces in order; joined == `resp.Text`;** `resp.StopReason == StopEndTurn` from `finish_reason: "stop"`.
  3. **Tool calls split across pieces come back whole.** Two calls (index 0 and 1), interleaved, arguments split
     over 3 pieces each → two `ToolCall`s in index order, valid JSON input, `StopToolUse`.
  4. **No-argument tool call** (arguments never sent, or `""`) → `Input` is `{}`.
  5. **Usage parsed** from a final `choices: []` chunk: input excludes cached, cached and output set.
  6. **No usage** → counts 0 and the logger buffer contains `llm: stream reply had no usage; tokens reported as 0`.
  7. **Body ends without `[DONE]`** → error, `errors.Is(err, io.ErrUnexpectedEOF)`; `shouldFailover(err)` is true.
  8. **Error event mid-stream** → error containing the message.
  9. **Status 429 before the stream** → `openai: ... (status 429)`, zero events.
  10. **Refusal** (`delta.refusal` pieces + `finish_reason: "stop"`, or `finish_reason: "content_filter"` after
      some content) → `ErrRefused`, `resp.Text` == streamed content, usage populated.
  11. **Stream goes quiet past the deadline** → `context.DeadlineExceeded`.
  12. **`ctx` cancelled in `OnEvent` after the first piece** → `context.Canceled`, exactly one event.
- [ ] **Step 6: Run them to see them fail.** `go test ./llm -run 'OpenAIStream' -v` → FAIL.
- [ ] **Step 7: Implement `completeStream`, the `complete` branch, `streams()`, the `New` logger change.**
- [ ] **Step 8: Run the tests.** `go test ./llm -run 'OpenAIStream|SSE' -v` → PASS; `go test ./llm` → PASS.
- [ ] **Step 9: Commit.** `git commit -m "feat(llm): stream OpenAI-compatible replies over server-sent events"`

---

### Task 4: Gemini streaming

**Files:**
- Modify: `llm/google.go`
- Create: `llm/google_stream_test.go`

**Interfaces:**
- Consumes: Task 1 types; `sseReader` (Task 3); `withDefaultDeadline`, `readError`, `geminiToolCalls`,
  `geminiStopReason`, `geminiRefusalCategory` (existing).
- Produces: `geminiPart.Thought`, `geminiBuildResponse`, `sendStream`, `streams()`.

**What to build:**
- Add `Thought bool \`json:"thought,omitempty"\`` to `geminiPart`.
- Refactor first, no behaviour change except thought parts: move the tail of `send` (text from parts,
  tool calls, stop reason, cached/input/thought-token usage, refusal) into `geminiBuildResponse(model, out)`.
  Text skips parts with `Thought == true` (Decision 6). Existing tests must pass after this step alone.
- `complete`'s retry loop calls `g.sendStream(ctx, model, body, req.OnEvent)` when `req.OnEvent != nil`, else
  `g.send`. The loop, the 400 check and the attempts list don't change. A 400 arrives as an HTTP status before
  any body, so no text has been sent when the next attempt starts.
- `sendStream`:
  - Same deadline and headers as `send`; URL `"%s/models/%s:streamGenerateContent?alt=sse"`.
  - Status ≥ 300 → read the whole body, return the same error as `send` (`gemini: <msg> (status N)`) **with
    the status**, so the 400 retry works. No events.
  - For each SSE payload decode a `geminiResponse`. `error` present → `fmt.Errorf("gemini: %s (stream)", msg)`.
  - If the event has candidates, from `candidates[0]`: for each part, a text part that is not `thought` and
    non-empty → append to text and call `onEvent`. A `functionCall` part → append the part to a collected list
    (they arrive whole). A non-empty `finishReason` → keep the latest.
  - `usageMetadata`: keep the latest event's value where any count is non-zero.
  - `promptFeedback` non-nil → keep it.
  - At end of body (`io.EOF`): if `ctx.Err() != nil` → `readError(ctx, "gemini", ctx.Err())`. If no
    `finishReason` was seen **and** `promptFeedback` carries no block reason in `geminiBlockReasons` →
    `fmt.Errorf("gemini: stream ended without finishReason: %w", io.ErrUnexpectedEOF)` (Decision 7). Read
    error → `readError(ctx, "gemini", err)`.
  - Build a synthetic `geminiResponse`. If any candidate was seen, set one candidate whose parts are
    `[{Text: text}] + collected function-call parts` and whose `FinishReason` is the latest one. Set the kept
    `UsageMetadata` and `PromptFeedback`. Return `geminiBuildResponse(model, synthetic)` with status 200.
    Its text is exactly `text`, so the join rule holds.
- `streams() bool { return true }`.

- [ ] **Step 1: Refactor to `geminiBuildResponse`, add `Thought`.** Run `go test ./llm` → PASS.
- [ ] **Step 2: Write the failing tests** (`llm/google_stream_test.go`). Use a fake server that answers
  `:streamGenerateContent` with a scripted SSE body (flush per event) and records the path, query and bodies.
  Supports a list of leading failure statuses like `geminiServer.fails`.
  1. **Request shape:** with `OnEvent`, path ends `:streamGenerateContent` and query has `alt=sse`; without it,
     `:generateContent` as today.
  2. **Text pieces in order; joined == `resp.Text`;** `StopEndTurn` from `STOP`.
  3. **`thought` parts never sent** and not in `resp.Text`; thought tokens still counted in `OutputTokens`.
  4. **Function calls** across two events → both `ToolCall`s, `StopToolUse`; a call with no args → `{}`.
  5. **Usage** from the last event with counts: input excludes cached, output = candidates + thoughts.
  6. **Stream ends without `finishReason`** → error with `io.ErrUnexpectedEOF`.
  7. **Prompt blocked** (single event, no candidates, `promptFeedback.blockReason: "SAFETY"`) → `ErrRefused`, no
     events.
  8. **Refusal midway** (text events then `finishReason: "SAFETY"`) → `ErrRefused`, `resp.Text` == streamed text.
  9. **Flash 400 retry still works:** model containing `flash`, first attempt 400, second streams → success, 2
     requests, events only from the second.
  10. **Error event mid-stream** → error.
  11. **Stream goes quiet past the deadline** → `context.DeadlineExceeded`.
  12. **`ctx` cancelled in `OnEvent` after the first piece** → `context.Canceled`, exactly one event.
- [ ] **Step 3: Run them to see them fail.** `go test ./llm -run 'GeminiStream' -v` → FAIL.
- [ ] **Step 4: Implement `sendStream`, the branch in `complete`, `streams()`.**
- [ ] **Step 5: Run the tests.** `go test ./llm -run 'GeminiStream' -v` → PASS; `go test ./llm` → PASS.
- [ ] **Step 6: Commit.** `git commit -m "feat(llm): stream Gemini replies via streamGenerateContent"`

---

### Task 5: Guard marker filter

**Files:**
- Create: `guard/stream.go`, `guard/stream_test.go`
- Modify: `guard/guard.go` (`Complete`), `guard/guard_test.go` (`fakeLLM`)

**Interfaces:**
- Consumes: `llm.StreamEvent`, `llm.StreamText`, `llm.Request.OnEvent` (Task 1); `StripMarker` (existing, for
  test comparisons only).
- Produces: `markerFilter`, `newMarkerFilter`, `push`, `finish`, `text` (see Shared interfaces).

**Wiring in `Complete` — exact order.** Everything up to and including `c.scan` is unchanged, so a request blocked
for size or rate limit returns before any event.

```text
class := ...; size check; rate limit; req = withPolicies(...); c.scan(ctx, req)     // unchanged
var f *markerFilter
if class == ClassConversational && c.o.RefusalMarker != "" && req.OnEvent != nil:
    f = newMarkerFilter(c.o.RefusalMarker, req.OnEvent)
    req.OnEvent = f.push                                // req is Complete's own copy
resp, err := c.o.Inner.Complete(ctx, req)
if err != nil: return resp, err                         // filter sends nothing more; resp unchanged
if class == ClassConversational:
    if f != nil:
        f.finish()
        resp.Text = f.text()                            // join rule by construction
        if f.declined: c.event(ctx, EventRefusal, class)
    else if text, declined := StripMarker(resp.Text, c.o.RefusalMarker); declined:   // today, unchanged
        resp.Text = text; c.event(ctx, EventRefusal, class)
return resp, nil
```

One-shot and background calls, and conversational calls with an empty marker, pass `OnEvent` through untouched.

**Filter logic — exact. Implement exactly this.** Let `m = f.marker` and `pat = m + " "`. Blank means the bytes
`' '` and `'\n'` only, matching `StripMarker`'s `TrimLeft(s, " \n")`.

`push(ev)`:

```text
if ev.Kind != llm.StreamText: f.next(ev); return      // unknown kinds pass through untouched
f.buf += ev.Text                                       // step 1

// step 2: remove complete markers
if strings.Contains(f.buf, pat):
    f.buf = strings.ReplaceAll(f.buf, pat, ""); f.declined = true
atEnd := strings.HasSuffix(f.buf, m)
body := f.buf
if atEnd: body = f.buf[:len(f.buf)-len(m)]             // a space may still follow it in the next piece
if strings.Contains(body, m):
    body = strings.ReplaceAll(body, m, ""); f.declined = true
if atEnd: f.buf = body + m else: f.buf = body

// step 3: hold back the longest tail that is a prefix of pat
k := 0
for n := min(len(f.buf), len(pat)); n > 0; n--:
    if strings.HasPrefix(pat, f.buf[len(f.buf)-n:]): k = n; break
ready := f.buf[:len(f.buf)-k]
// rune safety (Decision 9): never end ready in the middle of a UTF-8 sequence
for j := len(ready)-1; j >= 0 && j >= len(ready)-utf8.UTFMax; j--:
    if utf8.RuneStart(ready[j]):
        if !utf8.FullRuneInString(ready[j:]): ready = ready[:j]
        break
f.buf = f.buf[len(ready):]

f.emitReady(ready)                                     // steps 4–5
```

`emitReady(ready)` (steps 4–5):

```text
if !f.started:
    trimmed := strings.TrimLeft(ready, " \n")
    f.blanks += ready[:len(ready)-len(trimmed)]
    ready = trimmed
    if ready == "": return                             // still only blanks: keep holding
    if !f.declined: ready = f.blanks + ready           // no marker before it: blanks are real text
    f.blanks = ""                                      // declined: blanks dropped
    f.started = true
if ready != "": f.send(ready)
```

`send(s)`: `f.next(llm.StreamEvent{Kind: llm.StreamText, Text: s}); f.out.WriteString(s)`.

`finish()` (only after the inner `Complete` succeeded):

```text
if strings.Contains(f.buf, m):
    f.buf = strings.ReplaceAll(strings.ReplaceAll(f.buf, pat, ""), m, ""); f.declined = true
ready := f.buf; f.buf = ""
f.emitReady(ready)
if !f.started && f.blanks != "" && !f.declined: f.send(f.blanks)   // reply was only blanks, no marker
f.blanks = ""
```

Worked checks (each must hold, and each is a test below). `M` is the marker.

| Pieces | Events joined | `StripMarker(full)` |
| --- | --- | --- |
| `"M Sorry"` | `"Sorry"` | `"Sorry"` |
| `"M"`, `" Sorry"` | `"Sorry"` | `"Sorry"` |
| `"  "`, `"M Hi"` | `"Hi"` | `"Hi"` |
| `"  Hi"` | `"  Hi"` | `"  Hi"` (no marker) |
| `"Hello M"` | `"Hello "` | `"Hello "` |
| `"M \nHello"` | `"Hello"` | `"Hello"` |
| `"  "` | `"  "` | `"  "` |
| `"  M"` | `""` | `""` |
| `"\n\nHello "`, `"M x"` | `"\n\nHello x"` | `"Hello x"` (known difference, spec) |

- [ ] **Step 1: Write the failing filter tests** (`guard/stream_test.go`), using `testMarker` (`⟦dossi:declined⟧`,
  multi-byte). Helper `run(pieces ...string) (events []string, text string, declined bool)` builds a filter, pushes
  each piece, calls `finish`. Test cases:
  1. **Split at every byte position across two pieces**, for each of these replies: `M+" Maaf, saya hanya membantu."`,
     `"Jawaban: "+M+" tidak bisa."`, `"Normal answer."`, `"Hello "+M`, `M+"\nHello"`, `"é"+M+" ü"`. For every split:
     no event contains a partial marker (any non-empty proper prefix of `M` as a suffix, or `M` itself); no event
     is invalid UTF-8 (`utf8.ValidString`); joined == `text`; and since none of these start with blanks,
     joined == `StripMarker(full)`. `declined` == `strings.Contains(full, M)`.
  2. **Split at every pair of byte positions (three pieces)** for the first two replies above: same checks.
  3. **Blanks then marker** (`"  "`, `M+" Hi"`) → `"Hi"`. **Blanks then plain text** (`"  "`, `"Hi"`) → `"  Hi"`.
  4. **Marker then space in the next piece** (`M`, `" Hi"`) → `"Hi"` (the space is removed).
  5. **Marker in the middle** (`"Ini "`, `M`, `" jawaban"`) → `"Ini jawaban"`, `declined`.
  6. **Text that starts like the marker but isn't** (`"⟦dossi:"`, `"other⟧ text"`) → delivered in full,
     not `declined`.
  7. **Multi-byte runes next to the marker are never split**: covered by case 1's `"é"+M+" ü"`; also assert no
     event is empty.
  8. **Only blanks** (`"  "`) → `"  "`; **blanks + marker only** (`"  "+M`) → `""`, `declined`.
  9. **Known difference documented:** (`"\n\nHello "`, `M+" x"`) → `"\n\nHello x"`.
  10. **Non-text kind passes through** unchanged (`StreamKind("other")`), and doesn't affect `text()`.
- [ ] **Step 2: Run them to see them fail.** `go test ./guard -run 'Filter|Marker' -v` → compile error.
- [ ] **Step 3: Implement `guard/stream.go`.** Run the filter tests → PASS.
- [ ] **Step 4: Write the failing Wrap tests** (`guard/guard_test.go`). Extend `fakeLLM` with `pieces []string`:
  when `req.OnEvent != nil` and `pieces` is set, send each piece as a `StreamText` event and return
  `Text: strings.Join(pieces, "")`. Otherwise today's behaviour. Add an `err error` field: when set, send the pieces
  and then return the error with the joined text.
  1. **Conversational with marker:** pieces `M[:5]`, `M[5:]+" Maaf"` → events joined `"Maaf"` == `resp.Text`,
     one `EventRefusal`.
  2. **Conversational, no marker in reply** → events == pieces joined == `resp.Text`, no `EventRefusal`.
  3. **Rate-limited** and **too large** → zero events, inner not called.
  4. **One-shot and background** with a marker in the pieces → events pass through untouched (marker included),
     `resp.Text` unchanged.
  5. **Empty `RefusalMarker`** (conversational) → untouched.
  6. **Inner error after pieces** → error returned, `resp` unchanged, no events after the error, no
     `EventRefusal`.
  7. **Non-streamed conversational** (`OnEvent == nil`) → today's `StripMarker` behaviour (existing
     `TestRefusalMarkerStripped` still passes).
- [ ] **Step 5: Run them to see them fail.** `go test ./guard -v` → new tests FAIL.
- [ ] **Step 6: Implement the wiring in `Complete`.** Update its doc comment: for streamed conversational calls the
  marker is removed as text arrives.
- [ ] **Step 7: Run the tests.** `go test ./guard -v` → PASS; `go test -race ./guard` → PASS.
- [ ] **Step 8: Commit.** `git commit -m "feat(guard): strip the refusal marker from streamed replies"`

---

### Task 6: README and gates

**Files:**
- Modify: `README.md` — new `### Streaming` section under `## aikit/llm`, after `### Tool calling`; one sentence in
  `## Guardrails (aikit/guard)` about streamed replies.

**Content of the Streaming section** (prose plus one short example; match the README's tone):
- What it's for: show the reply as it's written. Set `Request.OnEvent`; `Complete` still returns the final
  `Response`.
- A ~12-line example: a chat handler that writes each `StreamText` piece to the client, then checks `err` and
  `resp.ToolCalls`.
- The five rules (same wording as the godoc): goroutine/order/never after return; quick and non-blocking; ignore
  unknown kinds; the join rule; events stay sent on error.
- Fallback: an error before any text falls back as today; an error after text does not (the user already saw
  part of a reply).
- `JSONSchema` requests: one event with the checked text on success, none on error.
- Tool rounds: tool calls are never streamed; they come back whole in `resp.ToolCalls`.
- OpenAI-compatible services: usage needs `stream_options.include_usage`. A service that ignores it reports 0
  tokens, and aikit logs a warning.
- Guard: with `guard.Wrap`, the refusal marker never reaches `OnEvent`. Known small difference: a reply that
  starts with blank lines and has the marker only in the middle keeps its leading blanks when streamed.

- [ ] **Step 1: Write the section.**
- [ ] **Step 2: Run the gates** (full suite, once, before the final review):

  ```sh
  go vet ./...
  go test ./...
  go test -race ./...
  ```

  Expected: all three clean. If any fails, stop and report the output. Don't fix code in this task.
- [ ] **Step 3: Commit.** `git commit -m "docs: streaming in llm and guard"`

---

## Final review (controller)

One Opus whole-branch review against the spec. Check in particular:

- Router rules 1–7 in the exact order of Task 1. Rule 6's check sits before `shouldFailover`. Rule 7 sits in
  `completeOn`.
- Join rule holds on every success path: each provider, the synthesized one-event paths, and guard.
- No goroutine calls `OnEvent`; nothing calls it after `Complete` returns.
- `OnEvent == nil` paths are unchanged (diff of non-streaming code is refactor-only: `oaiBuildResponse`,
  `geminiBuildResponse`, `Thought`).
- Parked minor findings from per-task reviews are fixed in one final wave.

## Spec coverage check

| Spec item | Task |
| --- | --- |
| `StreamKind`, `StreamEvent`, `Request.OnEvent`, rules 1–5 in godoc | 1 |
| Router rules 1–7 | 1 |
| Anthropic: stream when `OnEvent` set; text deltas; keep repair/stop rules | 2 |
| `sse.go` | 3 |
| OpenAI: `stream`, `stream_options`, `[DONE]`, content, tool-call joining, refusal, finish, usage + warning | 3 |
| Gemini: `streamGenerateContent?alt=sse`, text/thought/function parts, usage, finishReason error, Flash retry | 4 |
| Timeouts unchanged; stall → context error | 2, 3, 4 (tests) |
| Guard: blocked → no events; one-shot/background untouched; filter rules 1–5; finish; `EventRefusal` | 5 |
| README "Streaming" | 6 |
| Gates | 6 |
| Router tests (6 from spec) | 1 (cases 1–3, 6–8) |
| Provider tests (8 from spec) | 2, 3, 4 |
| Guard tests (8 from spec) | 5 |
