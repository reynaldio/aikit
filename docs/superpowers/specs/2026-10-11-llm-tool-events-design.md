# Tool events while streaming in `aikit/llm` — design

Date: 2026-10-11. Status: approved in brainstorming, waiting for spec review.

Builds on [2026-10-10-llm-streaming-design.md](2026-10-10-llm-streaming-design.md) (`Request.OnEvent`,
`StreamEvent`, router rules 1–7). Read that first; this spec only describes what changes.

## Why

The first user of streaming is a chat UI that runs tools. While the model writes a tool call, the screen goes
quiet: no text arrives, and the app only learns about the call when `Complete` returns. With tool events, the
app can show "Calling search_docs…" as soon as the model starts the call.

Running the tool happens in the app, not in aikit, so the app already knows when a tool runs and finishes.
These events only cover the time while the model is *writing* the call.

Out of scope: streaming a call's arguments as they are written ("option C" in brainstorming), events for the
providers' built-in tools (web search), and any change to how the app runs tools.

## Decisions (owner answers)

| Question | Decision |
| --- | --- |
| What the app can see | A: "start" only (tool name + call ID). B: "start" and "ready" (the call is fully written). Not C (live arguments) |
| Opt-in | Off unless the app asks. When on, the app chooses A or B. A is the normal "on" value |
| Shape | One field on `Request` with three values (off / A / B), not two booleans |
| Fallback after a tool event | Tool events do **not** block fallback. Only text does, as today. `resp.ToolCalls` is the truth when `Complete` returns |
| Where the option is applied | Providers always report both kinds when streaming; the router filters by the app's choice |

## API

```go
// New event kinds (StreamText is unchanged).
const (
    StreamToolStart StreamKind = "tool_start" // the model started writing a call
    StreamToolReady StreamKind = "tool_ready" // the model finished writing that call
)

// New fields on StreamEvent, set for the two tool kinds; empty for StreamText.
type StreamEvent struct {
    Kind       StreamKind
    Text       string // StreamText: the new piece of text
    ToolCallID string // tool kinds: equals the ToolCall.ID in the final Response
    ToolName   string // tool kinds: the ToolDef.Name being called
}

// ToolEvents chooses which tool events reach OnEvent.
type ToolEvents string

const (
    ToolEventsStart      ToolEvents = "start"       // option A: StreamToolStart only
    ToolEventsStartReady ToolEvents = "start_ready" // option B: StreamToolStart and StreamToolReady
)

// New field on Request:
ToolEvents ToolEvents // "" = no tool events (today's behaviour). Ignored when OnEvent is nil.
```

Any value other than the two constants is treated as off.

### Rules for tool events (add to the `OnEvent` godoc and README)

All five existing `OnEvent` rules still hold. The join rule is about `StreamText` only and is unchanged.

1. For each call, `StreamToolStart` comes before `StreamToolReady`. Each is sent at most once per call ID.
2. `ToolCallID` equals the `ID` of the matching `ToolCall` in `resp.ToolCalls`, so the app can match them.
3. Only calls to the app's own tools (`Request.Tools`) produce events. Built-in tools (web search) never do.
4. When `Complete` returns, `resp.ToolCalls` is the truth. A call the app saw a "start" or "ready" for may be
   missing from it: after a fallback (the call was from the first model), or on an error. The app clears any
   indicator whose ID is not in `resp.ToolCalls`.
5. "Ready" means the model finished writing the call. It does not promise the call is usable: a reply cut off
   at the token limit may have sent "ready" and still return `StopTruncated`. Check `resp.StopReason` as today.
6. A tool event may arrive slightly before text the guard's marker filter is still holding back (at most the
   marker's length). Text order and the join rule are unaffected.

## Router (`llm/llm.go`)

Rules 1–7 of the streaming spec stay in force, in the same order. The changes:

- **Filtering, in the existing tracker.** `streamTracker` knows the request's `ToolEvents`. In `emit`:
  1. `StreamText`: unchanged (non-empty text sets `sent`, then pass on).
  2. `StreamToolStart`: pass on if `ToolEvents` is `ToolEventsStart` or `ToolEventsStartReady`; else drop.
  3. `StreamToolReady`: pass on only if `ToolEvents` is `ToolEventsStartReady`; else drop.
  4. Any other kind: pass on (unchanged).
  Tool events never set `sent`. So fallback rules 5–6 are unchanged: only text blocks fallback.
- **`JSONSchema` (rule 2):** unchanged. `OnEvent` is removed before the providers run, so no tool events are
  sent. The single text event on success stays as it is.
- **Provider that can't stream (rule 7):** on success, after the one text event, send `StreamToolStart` and
  then `StreamToolReady` for each call in `resp.ToolCalls`, in order, through the tracker (which filters them).
  No events on error.

## Providers

Each provider sends both tool kinds whenever it has `OnEvent`, and checks `ctx.Err()` after every `OnEvent`
call (as the text path already does), stopping if the app cancelled.

### Anthropic (`llm/anthropic.go`)

- `content_block_start` whose block is `tool_use` → `StreamToolStart` with the block's `id` and `name`.
- `content_block_stop` for the index of a `tool_use` block → `StreamToolReady` (same ID and name).
- `server_tool_use` and other block types → nothing.

### OpenAI-compatible (`llm/openai.go`)

- Tool-call pieces are keyed by `index`. The first time an index has both a non-empty `id` and a non-empty
  `function.name` → `StreamToolStart`.
- OpenAI has no per-call end signal. When `finish_reason` arrives → `StreamToolReady` for every started call,
  in index order.
- A call that never got both an `id` and a `name` sends no events. It still comes back in `resp.ToolCalls` as
  today.

### Gemini (`llm/google.go`)

- Function calls arrive whole. For each `functionCall` part: create its ID (`geminiCallID()`) when the part
  arrives, send `StreamToolStart` then `StreamToolReady` back to back, and use that same ID for the final
  `ToolCall`. (Today the ID is created only when the response is built, which would not match.)
- The non-streamed path is unchanged.

## Guard (`guard/guard.go`, `guard/stream.go`)

No code change expected: the marker filter already passes non-text kinds straight through, and one-shot and
background calls pass `OnEvent` through untouched. Tests confirm it.

## File map

| File | Responsibility |
| --- | --- |
| `llm/llm.go` | New kinds, `StreamEvent` fields, `ToolEvents`, `Request.ToolEvents`, godoc; tracker filtering; rule-7 tool events |
| `llm/anthropic.go` | Tool start/ready from `content_block_start` / `content_block_stop` |
| `llm/openai.go` | Tool start per index; ready at `finish_reason` |
| `llm/google.go` | ID created on arrival; start + ready per call; final `ToolCall` reuses the ID |
| `guard/guard_test.go` | Pass-through tests (no code change expected) |
| `README.md` | Streaming section: the field, the kinds, the app rules |

## Tests

Router (`llm/stream_test.go`):

- `ToolEvents` unset → a provider's tool events never reach `OnEvent`; text unchanged
- `ToolEventsStart` → only start events; `ToolEventsStartReady` → start and ready
- unknown `ToolEvents` value → treated as off
- primary sends a tool start then fails before text → fallback still used; the app saw the first start event
- provider that can't stream, reply with tool calls → text event, then start (+ready) per call, filtered by mode
- `JSONSchema` set → no tool events
- `OnEvent == nil` → `ToolEvents` ignored; all existing tests pass unchanged

Providers (fake servers, one set each):

- one call and two parallel calls: start before ready per call; IDs and names match `resp.ToolCalls`
- text, then a call, then text: events in stream order; join rule holds for the text
- Anthropic: `server_tool_use` (web search) sends no tool events
- OpenAI: call pieces split across chunks; ready for every call only at `finish_reason`; a call with no name sends nothing
- Gemini: the ID in the events equals the final `ToolCall.ID`
- cancel inside `OnEvent` on a tool event → `context.Canceled`, no more events

Guard (`guard/guard_test.go`):

- conversational with marker: tool events pass through unchanged; text still filtered
- one-shot and background: tool events pass through untouched

## Gates

```sh
go vet ./...
go test ./...
go test -race ./...
```

## Risks

- **OpenAI-compatible backends that send the name late or never.** Covered: no start until both ID and name
  are known; the call still comes back in `resp.ToolCalls`.
- **Stray indicators.** An app that ignores rule 4 can leave a "Calling…" indicator on screen after a fallback.
  Documented in the godoc and README.
