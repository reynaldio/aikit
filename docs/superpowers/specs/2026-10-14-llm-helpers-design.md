# Helpers for `aikit/llm` (test fake, usage wrapper, tool loop) and stop-billing — design

Date: 2026-10-14. Status: approved in brainstorming, waiting for spec review.

Second of four sub-projects, in this order: retries (merged, d7b378e) → **helpers** → `llm.Embed` →
Batch. Each one gets its own spec, plan and build.

This spec has four parts:

1. `llmtest`: a fake `llm.Client` for tests.
2. `llm.WithUsage`: a wrapper that reports the tokens and cost of each call.
3. `llm.RunTools`: an optional tool loop.
4. Stop-billing: a stopped or broken stream returns the tokens used so far. This comes from the dossio
   handoff `dossio/dev/docs/handoffs/2026-10-14-aikit-cancelled-stream-usage.md`.

Part 4 is here because the usage wrapper is only right once stopped streams report their tokens.

## Why

- **Test fake.** Apps that use `llm.Client` each write their own fake. The fakes in `llm` are internal
  (`fakeProvider`), so nobody outside can use them.
- **Usage wrapper.** The README says "wrap the `Client` to record usage", and every app writes that wrapper
  again, with the same price lookup.
- **Tool loop.** The README shows the tool loop as code to copy. Most callers want the same loop.
- **Stop-billing.** When a user presses Stop, the stream's `ctx` is cancelled. All three providers then
  return a zero `Response`. The provider has already charged for the tokens, but the app sees 0, so a user
  can stop long replies again and again for free.

## Decisions (owner answers)

| Question | Decision |
| --- | --- |
| How a test sets the fake's answers | A list of replies, used in order. The fake records every request |
| Does the fake stream? | Yes. It sends `OnEvent` events by the same rules as the router |
| Which calls the usage wrapper reports | Every call whose reply has tokens: success, refusal, schema violation, stopped stream. A transport error with a zero reply is not reported |
| Tokens from a primary model that failed before the fallback answered | Not reported. Documented as a known gap. The router does not change for this |
| How the tool loop runs several calls from one round | One after another by default. An option runs them at the same time |
| Where the stop-billing fix goes | In this spec, as part 4 |
| Missing token counts on a stopped stream | Guess them at about 4 characters per token. A new flag says the numbers are guessed |
| Can we wait for the real count after Stop? | No. The connection closes and the provider sends nothing more. Reading to the end would make the model write (and bill) the whole reply |
| Old follow-ups to fold in | 4, 6, 7, 8 from the memory note (see "Small follow-ups") |

## Part 1: `llmtest`

New package `github.com/reynaldio/aikit/llm/llmtest`. It imports only `llm`.

```go
type Reply struct {
    Response llm.Response
    Err      error
    Events   []llm.StreamEvent // when set, sent exactly as written instead of the generated events
}

type Client struct { /* mutex, replies, requests */ }

func New(replies ...Reply) *Client
func (c *Client) Complete(ctx context.Context, req llm.Request) (llm.Response, error)
func (c *Client) Enabled() bool                  // true
func (c *Client) CompareTargets() []llm.ModelRef // nil
func (c *Client) Requests() []llm.Request        // a copy, in call order
func (c *Client) Add(replies ...Reply)           // add more replies after New

var ErrNoReplies = errors.New("llmtest: no replies left")
```

Rules for `Complete`, in this order:

1. Record `req` (with `OnEvent` kept, so a test can see it was set).
2. If `ctx` has already ended, return `ctx.Err()` without using a reply.
3. Take the next reply. If none is left, return `ErrNoReplies`.
4. If `req.OnEvent` is set, send events (see below).
5. Return `Reply.Response, Reply.Err`.

Events, when `req.OnEvent` is set:

- If `Reply.Events` is set, send those exactly, and nothing else. This is how a test fakes "text was sent,
  then an error".
- Otherwise, if `req.JSONSchema` is set: send `Response.Text` as one `StreamText` event, only when `Err` is
  nil and the text is not empty. No tool events.
- Otherwise: split `Response.Text` into up to 3 pieces at rune boundaries and send each one as `StreamText`.
  The pieces joined equal `Response.Text` (the join rule). Then, for each `ToolCall`, send
  `StreamToolStart`, then `StreamToolReady`, filtered by `req.ToolEvents` exactly as the router's
  `streamTracker.emit` does. When `Err` is set, send no generated events.
- Events are sent on the calling goroutine, before `Complete` returns.

`Complete` is safe to call from several goroutines (one mutex around the reply list and the request list).
Events are sent outside the lock.

## Part 2: `llm.WithUsage`

New file `llm/usage.go`.

```go
type Usage struct {
    Provider         Provider
    Model            string
    Task             Task          // from the request
    InputTokens      int
    OutputTokens     int
    CachedTokens     int
    CacheWriteTokens int
    CostUSD          float64       // book.Cost(Model, the four counts)
    Estimated        bool          // Response.UsageEstimated (part 4)
    Duration         time.Duration // time spent in Complete
    Err              error         // nil on success
}

func WithUsage(inner Client, book *PriceBook, record func(ctx context.Context, u Usage)) Client
```

- `book` may be nil. Then the built-in `DefaultPrices` are used (`PriceBook.Rate` already handles a nil book).
- `Enabled` and `CompareTargets` pass through to `inner`.
- `Complete` calls `inner.Complete` with the same `ctx` and `req`, and does not touch `OnEvent`.
- After it returns, `record` is called **only if** the reply has tokens: any of the four counts is more than 0.
  This covers success, `ErrRefused`, `ErrSchemaViolation` and a stopped stream (part 4). A zero reply (a
  transport error, `ErrNotConfigured`, a guard refusal) is not reported.
- `record` is called on the calling goroutine, before `Complete` returns. The response and error are returned
  unchanged.
- `record` must not be nil. `WithUsage` panics at build time if it is (a programming error).

The doc comment says:

- The `ctx` passed to `record` may already be cancelled (the user pressed Stop). To save the record, use
  `context.WithoutCancel(ctx)`.
- **Known gap:** when the primary model fails and the fallback answers, only the fallback's tokens are
  reported. If the primary had already charged tokens (a refusal, or a stream that broke before any text was
  sent), those tokens are not reported. This reports too little but never charges twice.
- Order with `guard.Wrap`: either order works. A guard refusal has no tokens, so it is never reported.

## Part 3: `llm.RunTools`

New file `llm/tools.go`.

```go
type ToolFunc func(ctx context.Context, call ToolCall) (string, error)

type ToolLoopOptions struct {
    MaxRounds int  // calls to Complete, at most. 0 means 8
    Parallel  bool // run the calls of one round at the same time
    OnRound   func(round int, resp Response) error // optional. round is 1-based. A non-nil error stops the loop
}

type ToolRun struct {
    Response Response   // the last reply received
    Messages []Message  // req.Messages plus every assistant turn and tool-result turn added
    Rounds   []Response // every reply received, in order
}

var (
    ErrToolLoopLimit      = errors.New("llm: tool loop reached MaxRounds")
    ErrToolCallsTruncated = errors.New("llm: tool calls were cut off at the token limit")
)

func RunTools(ctx context.Context, c Client, req Request, tools map[string]ToolFunc, opts ToolLoopOptions) (ToolRun, error)
```

The loop, in this order:

1. Copy `req.Messages` into a new slice. The caller's slice is never changed.
2. For round = 1, 2, …:
   1. If `ctx` has ended, return the run so far and `ctx.Err()`.
   2. Call `c.Complete(ctx, req)` with `req.Messages` set to the current history. Every other field of
      `req` is sent unchanged each round (`Tools`, `OnEvent`, `CacheHistory`, …).
   3. If the reply has tokens, or the error is nil, add it to `Rounds` and set `Response`.
   4. If `Complete` returned an error, return the run and that error, unchanged.
   5. If `OnRound` is set, call it. If it returns an error, return the run and that error.
   6. If the reply has no `ToolCalls`, return the run and nil. This is the normal end. A truncated reply
      without calls also ends here; the caller checks `StopReason` as with `Complete`.
   7. If the reply has `ToolCalls` and `StopReason == StopTruncated`, return the run and
      `ErrToolCallsTruncated`. The half-written calls are not run.
   8. If round == MaxRounds, return the run and `ErrToolLoopLimit`. The calls of this round are not run.
   9. Run the calls (see below), then append two messages to the history:
      `{Role: "assistant", Content: resp.Text, ToolCalls: resp.ToolCalls}` and
      `{Role: "user", ToolResults: results}`. All results go in that one message.

Running the calls:

- For each call, the result is `ToolResult{ToolCallID: call.ID}` plus:
  - no entry in `tools` for `call.Name`: `Content: "unknown tool: <name>"`, `IsError: true`;
  - the function returns an error: `Content: err.Error()`, `IsError: true`;
  - otherwise `Content` is the returned string.
- Without `Parallel`: the calls run one after another, in the order the model sent them.
- With `Parallel`: one goroutine per call, and the loop waits for all of them. Results stay in the model's
  order. The functions must be safe to run at the same time; the doc says so.
- Each function gets the loop's `ctx`. A panic in a function is not caught.
- After the calls, if `ctx` has ended, return the run and `ctx.Err()`. The results are not appended.

Notes for the doc comment:

- With `OnEvent` set, text from every round goes to the same callback. The join rule holds per round, not
  for the whole run.
- With `guard.Wrap`, call `guard.BeginTurn(ctx)` before `RunTools`, so the whole loop counts as one turn.
- With `WithUsage`, every round is reported on its own.

## Part 4: Stop-billing

### What changes

When a streamed call ends early, the provider returns the error **together with** a `Response` that holds the
tokens used so far. "Ends early" means:

- `ctx` was cancelled or its deadline passed during the stream, or
- the stream broke after at least one event had arrived (read error, or the body ended before the last
  event).

If no event had arrived yet, the result is unchanged: a zero `Response` and the error.

The error itself does not change. `errors.Is(err, context.Canceled)` still works, and so do the router's
`shouldFailover` and every existing error test.

The partial `Response` holds the text received so far and the token counts. `ToolCalls` is nil (each
provider has a different idea of when a call is complete, and nobody should run calls from a failed reply).
`StopReason` is left empty. As with every error, callers must not trust the other fields; only the
token counts and Provider/Model are meant to be read.

### New field

```go
// On Response:
// UsageEstimated is true when at least one token count is a guess, not the provider's
// number. It is only set on a stream that ended early (see the stop-billing rules).
UsageEstimated bool
```

### Where the numbers come from

New file `llm/estimate.go`:

```go
// estimateTokens guesses a token count: runes / 4, rounded up. "" is 0.
func estimateTokens(s string) int
```

| Provider | Input, cache read, cache write | Output |
| --- | --- | --- |
| Anthropic | From `message_start` (`msg.Usage`, already stored by `Accumulate`). Real numbers | If a `message_delta` with usage arrived: its `output_tokens` (real). Otherwise a guess over every block in `msg.Content`: text, thinking, and tool input JSON. `message_start`'s small `output_tokens` is ignored |
| Gemini | From the last `usageMetadata` seen (`promptTokenCount` minus cached, and `cachedContentTokenCount`), the same way as the normal path. If none arrived: a guess over the request (see below) | If that `usageMetadata` has `candidatesTokenCount` > 0: `candidatesTokenCount + thoughtsTokenCount` (real). Otherwise a guess over the text, thought text and function-call args received |
| OpenAI-compatible | If a usage chunk arrived: from it (real). Otherwise a guess over the request | A guess over the text and tool-call arguments received, unless a usage chunk arrived |

The guess over the request (input) counts: `SystemCacheable`, every message's `Content`, every
`ToolCall.Input`, every `ToolResult.Content`, and the JSON of every `ToolDef` (`Name`, `Description`,
`Schema`). Images and documents count as 0. A guessed input sets cache read and cache write to 0.

`UsageEstimated` is true when any of the four numbers is a guess.

### Router

- `completeOn` today stamps `Provider` and `Model` on a failed reply only for `ErrRefused`. It now stamps
  them on **any** failed reply whose token counts are not all 0, so the right price is used.
- Nothing else in the router changes:
  - text already sent: no fallback, return the partial reply (as today);
  - no text sent and `shouldFailover(err)`: fail over, and the partial reply is dropped (the known gap in
    part 2);
  - a fallback's tokens are never added to the first reply's.
- `guard.Wrap` already returns the inner reply unchanged on error. No change there.

### Behavior change (release note)

A streamed call that is stopped or breaks after data arrived now returns its token counts with the error,
instead of a zero `Response`. Counts that the provider had not sent yet are guessed, and
`Response.UsageEstimated` says so. Release as **v0.10.0** (new field, new exported names).

## Small follow-ups folded in

From the memory note `streaming-deferred-followups`, checked on main on 2026-10-14. All still apply.

| # | Item | File | Change |
| --- | --- | --- | --- |
| 4 | `gemStreamFake` appends to its lists from the handler goroutine with no lock | `llm/google_stream_test.go` | Add a mutex around the appends and the reads |
| 6 | Missing tool-event tests | `llm/anthropic_stream_test.go`, `llm/google_stream_test.go` | Add: cancel right after a ready event (Anthropic, Gemini); Anthropic hits `max_tokens` in the middle of a tool input with tool events on; Gemini refusal-return ID overwrite; tool tests check the accumulated `Input`; `evRec.desc()` prints unknown kinds instead of skipping them |
| 7 | `TestStreamDoesNotMutateCallersRequest` cannot fail (the request is passed by value); helper named `eq` | `llm/stream_test.go` | Make the test check the caller's `Messages` slice and `OnEvent` after the call (the shared parts a value copy does not protect). Rename `eq` to `eqStrings` |
| 8 | `TestRetryDeadlineTooClose` "backoff past deadline" uses a real 100ms timeout | `llm/retry_test.go` | Raise it to 1s. The fake `now` already makes the check itself fixed |

Still parked: OpenAI tool pieces without `index` (1), an OpenAI call ID that changes after its start (2), a
tool piece after `finish_reason` (3), the empty leading Gemini text part (5), the guard tracker case, the
`stream_options` risk, the `flash` model in router tests, and Anthropic's retry-count header.

## Files

| File | Responsibility |
| --- | --- |
| `llm/llmtest/llmtest.go` (new) | The fake `Client` (part 1) |
| `llm/llmtest/llmtest_test.go` (new) | Its tests |
| `llm/usage.go` (new) | `Usage`, `WithUsage` (part 2) |
| `llm/usage_test.go` (new) | Its tests. Uses `llmtest` |
| `llm/tools.go` (new) | `ToolFunc`, `ToolLoopOptions`, `ToolRun`, `RunTools`, the two errors (part 3) |
| `llm/tools_test.go` (new) | Its tests. Uses `llmtest` |
| `llm/estimate.go` (new) | `estimateTokens` and the request-input guess (part 4) |
| `llm/llm.go` | `Response.UsageEstimated`; `completeOn` stamps any failed reply with tokens |
| `llm/anthropic.go` | `stream` keeps the partial `msg` on every early-end path (the `ctx.Err()` checks in and after the loop, `s.Err()`, the missing `message_stop`) and `complete` builds the partial reply |
| `llm/google.go` | The stream loop keeps text, calls and the last `usageMetadata` on every early-end path |
| `llm/openai.go` | `completeStream` keeps text, calls and any usage chunk on every early-end path |
| `llm/*_stream_test.go`, `llm/stream_test.go`, `llm/retry_test.go` | New stop tests and follow-ups 4, 6, 7, 8 |
| `README.md` | Short sections: "Testing with llmtest", "Recording usage", "Tool loop helper", and a line under Streaming about stopped streams and `UsageEstimated` |

Import cycle note: `llm` tests that use `llmtest` must be in package `llm_test` (external test package),
because `llmtest` imports `llm`.

## Out of scope

- Reporting the primary's tokens when a fallback answers (the known gap). One `Response` cannot hold two
  models' usage; a per-attempt hook would be its own change.
- Non-streamed calls that are cancelled. The request never got a body, so there is nothing to report.
- Old follow-ups 1, 2, 3, 5 and the others listed as still parked.
- Counting images and documents in the input guess.

## Tests

`llm/llmtest/llmtest_test.go`:

1. Replies come back in order; the fourth call on three replies returns `ErrNoReplies`.
2. `Requests()` returns every request in order, and changing the returned slice does not change the fake.
3. A cancelled `ctx` returns `ctx.Err()` and does not use a reply.
4. With `OnEvent`: the text pieces joined equal `Response.Text`; tool events follow `ToolEvents` (none,
   start only, start and ready).
5. With `JSONSchema`: one text event. With `Err` set: no generated events.
6. `Reply.Events` is sent exactly, even with `Err` set.
7. Called from 10 goroutines under `-race`: every reply is used once.

`llm/usage_test.go`:

8. Success: one record with the right counts, cost from a `PriceBook` override, `Task`, `Err == nil`.
9. `ErrRefused` with tokens: one record, `Err` set.
10. Transport error with a zero reply: no record.
11. `Estimated` copies `Response.UsageEstimated`.
12. Nil book uses `DefaultPrices`.
13. The reply and error are returned unchanged; `OnEvent` still reaches the caller.
14. `Enabled` and `CompareTargets` pass through. Nil `record` panics.

`llm/tools_test.go`:

15. No tool calls: one round, `Messages` equals the input, nil error.
16. Two rounds: the second request holds the assistant turn and ONE user turn with both results, in order.
17. A tool error becomes `IsError: true` with the error text; the loop goes on.
18. An unknown tool becomes `IsError: true` with `unknown tool: <name>`.
19. A truncated round with calls: `ErrToolCallsTruncated`, the tools are not run.
20. `MaxRounds: 2` with calls every round: `ErrToolLoopLimit` after 2 calls to `Complete`.
21. `OnRound` returns an error: the loop stops with that error.
22. A `Complete` error in round 2: the run holds round 1, and the error is returned unchanged.
23. `Parallel`: two slow calls run at the same time (both start before either ends, checked with channels,
    not sleeps); results stay in the model's order.
24. The caller's `req.Messages` is not changed.
25. `ctx` cancelled by a tool: the loop returns `ctx.Err()` and makes no more `Complete` calls.

Stop-billing (provider tests):

26. Anthropic: cancelled after `message_start` and two text deltas → `errors.Is(err, context.Canceled)`;
    input, cache read and cache write equal the `message_start` values; output is guessed from the content,
    not `message_start`'s placeholder; `UsageEstimated == true`.
27. Anthropic: cancelled after a `message_delta` with `output_tokens` → that exact output,
    `UsageEstimated == false`.
28. Anthropic: cancel while `s.Next()` is blocked on a read (the `s.Err()` path) → the same partial usage as
    test 26.
29. Anthropic: the body ends with no `message_stop` → partial usage with the existing EOF error.
30. Gemini: cancelled after an event with `usageMetadata` → those counts.
31. Gemini: cancelled before any `usageMetadata` but after text → input and output guessed,
    `UsageEstimated == true`.
32. OpenAI-compatible: cancelled mid-text → input guessed from the request, output from the text,
    `UsageEstimated == true`.
33. Any provider: cancelled before any event → zero `Response` (unchanged).
34. `estimateTokens`: "" → 0, "abc" → 1, "abcd" → 1, "abcde" → 2, multi-byte runes count as one each.
35. Router: a stream cancelled after text → the partial reply comes back with Provider and Model set, and no
    fallback is called.
36. Router: a stream that breaks before text → failover, and the fallback's reply is returned alone.
37. `WithUsage` over a router with a cancelled stream → one record with `Estimated == true`.

Plus the follow-up tests in "Small follow-ups folded in".

## Gates

```
go vet ./...
go test -race ./...
```
