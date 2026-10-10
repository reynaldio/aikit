# Streaming text in `aikit/llm` — design

Date: 2026-10-10. Status: approved in brainstorming, waiting for spec review.

## Why

The first user is a chat UI that runs tools. Today `Complete` returns the whole reply at
the end, so the user waits 10–20 s with nothing on screen. With streaming, the reply's
text appears piece by piece while the model writes it.

Out of scope for v1: streaming thinking text, "tool call started" events, citations,
filling a JSON form field by field, and a pull-style `Stream` method. The API leaves room
for all of these without breaking changes.

## Decisions (owner answers)

| Question | Decision |
| --- | --- |
| First user | Chat UI with tool rounds |
| Providers in v1 | All three: Anthropic, OpenAI-compatible, Gemini |
| API shape | A callback on `Request` (`OnEvent`), not a new method. Chosen for the long term: one code path for fallback, refusals, truncation and guard |
| Callback type | `func(StreamEvent)` with a `Kind`, not `func(string)`, so new event kinds can be added later |
| `JSONSchema` set, or provider can't stream | Run the normal call; on success send the whole text as **one** event |
| Error after text was sent | No fallback; return the error |

## API

```go
type StreamKind string

const StreamText StreamKind = "text"

type StreamEvent struct {
    Kind StreamKind
    Text string // the new piece of text (for StreamText)
}

// New field on Request:
OnEvent func(StreamEvent) // nil = no streaming; behaviour identical to today
```

`Client` does not change. `Complete` still returns the final `Response` and error.

### Rules for `OnEvent` (document in godoc and README)

1. Called on the goroutine that called `Complete`, in order, never concurrently, never
   after `Complete` returns.
2. Must be quick and must never block; aikit waits for it before reading more of the
   stream.
3. Only `StreamText` is sent in v1. Callers must ignore kinds they don't know.
4. **Join rule:** when `Complete` returns `err == nil`, the `Text` of all `StreamText`
   events joined in order equals `resp.Text`, exactly. This holds with and without
   `guard.Wrap`.
5. When `Complete` returns an error, events already sent stay sent; the `Response` keeps
   whatever partial text it has today (refusals carry partial text and usage).

## Router rules (`llm/llm.go`) — exact order

This is the part that decides whether a user sees a broken reply, so the order is fixed:

1. If `req.OnEvent == nil`: today's path, unchanged.
2. If `req.JSONSchema != nil`: call providers with `OnEvent` removed (today's path,
   including schema-aware fallback). If the final result is a success, send one
   `StreamText` event with `resp.Text` (skip it if the text is empty), then return. On any
   error, send no event.
3. Otherwise wrap `OnEvent` in a tracker that records `sent = true` the first time it
   passes on an event with non-empty text. Pass the wrapped callback to the primary
   provider.
4. Primary returns success: return it.
5. Primary returns an error (including a refusal) and `sent == false`: run today's fallback
   logic unchanged (`NoFallback`, configured fallback, `fallbackRef`), giving the fallback
   the same tracker. Only the fallback's text reaches the caller.
6. Primary returns an error and `sent == true`: return that error and response as they
   are. Do not fall back. Log at warn: `llm: stream failed after text was sent; not
   falling back`.
7. A provider that cannot stream receives a request with `OnEvent` removed and, on
   success, the router sends one event with the full text (same as rule 2). In v1 all
   three providers stream, so this covers `noop` and test fakes.

## Providers

All three follow the same pattern: send text pieces to `OnEvent` as they arrive, collect
everything else, and at the end build the same `Response` as `complete` does today
(same text, tool calls, stop reason, usage, refusal errors, `ErrToolResultMismatch`
checks). The caller never sees half-built tool calls, thinking text, or raw provider
formats.

### Anthropic (`llm/anthropic.go`)

- Use the existing `stream` path whenever `req.OnEvent != nil` (as well as for large
  `MaxTokens`, as today). One streaming code path.
- On each `content_block_delta` with a `text_delta`, call `OnEvent`.
- Keep `repairToolInputs`, the "stream ended before `message_stop` is an error" rule, and
  `anthropicResponse` as they are. `resp.Text` is the text blocks joined with no
  separator, so the join rule holds.

### OpenAI-compatible (`llm/openai.go`)

- When streaming: add `"stream": true` and `"stream_options": {"include_usage": true}`.
- Read server-sent events until `data: [DONE]`. A body that ends without `[DONE]` is an
  error.
- `choices[0].delta.content` → append to text and call `OnEvent`.
- `choices[0].delta.tool_calls[]` arrive in pieces keyed by `index`: the first piece has
  `id` and `function.name`, later pieces add to `function.arguments`. Join per index, then
  build `ToolCall`s with today's rules (empty arguments → `{}`).
- `choices[0].delta.refusal` → collect; with `finish_reason` it goes through today's
  `oaiRefusal` to become `ErrRefused`.
- `finish_reason` → today's `oaiStopReason`.
- Usage comes in the last chunk (`usage`, with empty `choices`). If no usage arrived, log
  a warning (`llm: stream reply had no usage; tokens reported as 0`) and leave the
  counts at 0.

### Gemini (`llm/google.go`)

- When streaming: post to `models/{model}:streamGenerateContent?alt=sse`.
- Each event is a `geminiResponse`. Text parts (not `thought: true`) → append and call
  `OnEvent`. Function-call parts → collect (they arrive whole). Keep the latest
  `usageMetadata`, `finishReason` and safety data.
- At the end build the `Response` with today's helpers (`geminiToolCalls`,
  `geminiStopReason`, `geminiRefusalCategory`). A stream with no `finishReason` is an
  error.
- The Flash thinking-config retry keeps working: a 400 arrives before any text, so the
  next attempt starts with nothing sent.

### Shared

- New `llm/sse.go`: a small reader that yields `data:` payloads from a server-sent events
  body. Used by OpenAI and Gemini.
- Timeouts don't change: the caller's `ctx` deadline or `withDefaultDeadline`. A stream
  that stops sending partway ends in `context.DeadlineExceeded` / `context.Canceled`,
  never a partial success.

## Guard (`guard/guard.go`, new `guard/stream.go`)

Unchanged: size check, rate limit, policies and scan run before the inner call; a blocked
request sends no events; `BeginTurn` counts a turn once; one-shot and background calls
pass `OnEvent` through untouched.

New, for `ClassConversational` with a non-empty `RefusalMarker` and a non-nil `OnEvent`:
wrap `OnEvent` in a marker filter that gives the same result as `StripMarker`.

Filter rules, in order, for each incoming piece:

1. Add the piece to a small pending buffer.
2. Remove every complete `marker + " "` from the buffer, then every complete `marker`
   that is **not** at the very end of the buffer (a space may still follow it in the next
   piece). If any were removed, note `declined = true`.
3. Hold back the longest tail of the buffer that is a prefix of `marker + " "` (this
   includes a complete marker at the end). Split only on rune boundaries. Everything
   before the held tail is ready to send.
4. Start of reply: until the first non-blank character has been sent, hold blank text
   (spaces, new lines) instead of sending it. When the first non-blank text is ready:
   if a marker was removed before it (`declined`), drop the held blanks; otherwise send
   them first. This matches `StripMarker` for the common case (marker at the start).
5. Send the ready text if it is non-empty.

When the inner `Complete` returns success: remove any remaining `marker` from the buffer,
apply step 4, send what is left if non-empty, emit `EventRefusal` if `declined`, and set
`resp.Text` to the joined output of the filter. So for streamed calls the join rule holds
by construction. On error: send nothing more; `resp` is returned unchanged.

Known small difference from `StripMarker`: a reply that starts with blanks and has the
marker only in the middle keeps its leading blanks when streamed (`StripMarker` would
trim them). Non-streamed calls keep using `StripMarker` unchanged.

Tests for the filter: joined events == `resp.Text` always; and == `StripMarker(full)`
for every reply that does not start with blank text.

## File map

| File | Responsibility |
| --- | --- |
| `llm/llm.go` | `StreamKind`, `StreamEvent`, `Request.OnEvent`; router rules above |
| `llm/sse.go` (new) | Server-sent events reader |
| `llm/anthropic.go` | Text deltas to `OnEvent` on the existing stream path |
| `llm/openai.go` | Streaming request, chunk parsing, tool-call joining, usage |
| `llm/google.go` | `streamGenerateContent` request and chunk parsing |
| `guard/stream.go` (new) | Marker filter |
| `guard/guard.go` | Wrap `OnEvent` for conversational calls |
| `README.md` | New "Streaming" section |

## Tests

Router (`llm/llm_test.go`):

- error before any text → fallback used; only the fallback's text reaches `OnEvent`
- error after some text → no fallback; the error is returned
- refusal midway → `ErrRefused`, partial text in `Response`, no fallback
- `JSONSchema` set → exactly one event on success; no event on `ErrSchemaViolation`
- `OnEvent == nil` → all existing tests pass unchanged
- `noop`/fake provider that can't stream → one event with the full text

Providers (fake HTTP servers, one set each):

- text pieces arrive in order; joined == `resp.Text`
- tool calls split across pieces come back whole; no-argument call → `{}`
- usage parsed; OpenAI-compatible stream without usage → 0 and a warning
- stream ends before its end signal → error
- stream goes quiet past the deadline → `context.DeadlineExceeded`
- refusal → `ErrRefused` with partial text
- Gemini: `thought` parts never sent; Flash 400 retry still works
- `ctx` cancelled midway → `context.Canceled`, no more events

Guard (`guard/stream_test.go`, `guard/guard_test.go`):

- marker split at every possible position across two pieces, and across three pieces:
  no partial marker ever reaches `OnEvent`; joined == `resp.Text` == `StripMarker(full)`
- reply starting with blanks then the marker → blanks dropped; blanks then plain text →
  blanks kept
- marker followed by a space split from it → the space is removed
- marker in the middle of a reply → removed, `EventRefusal` sent
- text that starts like the marker but isn't → delivered in full
- multi-byte runes next to the marker are never split
- rate-limited / too large → zero events
- one-shot and background → events pass through untouched

## Gates

```sh
go vet ./...
go test ./...
go test -race ./...
```

## Risks

- **OpenAI-compatible services without `stream_options`.** Some may reject the option or
  ignore it. Ignored → covered by the 0-usage warning. Rejected (400) → a stream that
  never starts; it falls back like any error before text. If this shows up for a real
  service, add a per-provider switch in a later plan.
- **Slow `OnEvent`.** It slows reading the stream and can hit the deadline. Documented
  as a rule; not enforced.
