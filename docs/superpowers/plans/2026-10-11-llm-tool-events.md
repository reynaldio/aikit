# Tool Events While Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an app that streams (`Request.OnEvent`) opt in to "tool call started" and, optionally, "tool call ready" events, on all three providers.

**Architecture:** Providers always report both tool kinds when they stream. The router's existing `streamTracker` drops them according to the new `Request.ToolEvents` field, so the option logic lives in one place. Tool events never count as "sent", so fallback is unchanged.

**Tech Stack:** Go 1.25, `github.com/anthropics/anthropic-sdk-go` v1.56.0, the existing hand-written SSE reader (`llm/sse.go`). No new dependencies.

**Spec:** [docs/superpowers/specs/2026-10-11-llm-tool-events-design.md](../specs/2026-10-11-llm-tool-events-design.md). It builds on the streaming spec [2026-10-10-llm-streaming-design.md](../specs/2026-10-10-llm-streaming-design.md). Read the tool-events spec before your task.

## Global Constraints

- Branch: `feat/llm-tool-events` (already checked out). Commit per task. End every commit message with a blank
  line and then `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` on its own line (not on the
  subject line).
- No new module dependencies.
- `ToolEvents` unset, or `OnEvent == nil`, must behave exactly as today. Every existing test passes unchanged.
- Tool events never set the tracker's `sent`. Only non-empty `StreamText` blocks fallback, as today.
- All existing `OnEvent` rules still hold: the calling goroutine, in order, never concurrently, never after
  `Complete` returns. The join rule (joined `StreamText` == `resp.Text` on success) is unchanged.
- Every provider checks `ctx.Err()` after **every** `OnEvent` call, text or tool, and stops if it is set. This is
  the same check the text path already does.
- A tool event's `ToolCallID` equals the `ID` of the matching `ToolCall` in `resp.ToolCalls`, and its `ToolName`
  equals that call's `Name`.
- Per call ID: `StreamToolStart` before `StreamToolReady`, and each at most once.
- Only calls to the app's own tools produce events. Anthropic `server_tool_use` (web search) and Gemini Google
  Search never do.
- Any `ToolEvents` value other than `ToolEventsStart` / `ToolEventsStartReady` is treated as off.
- Match the surrounding code's comment density and idiom.

## Decisions

From the spec (owner answers, not open for change):

| Question | Decision |
| --- | --- |
| What the app can see | A: start only. B: start and ready. Not live arguments |
| Opt-in | Off unless asked. On = choose A (`ToolEventsStart`) or B (`ToolEventsStartReady`) |
| Fallback after a tool event | Not blocked. `resp.ToolCalls` is the truth when `Complete` returns |
| Where the option is applied | Providers always report both kinds; the router filters |

Clarifications made while planning:

1. **Gemini IDs.** Today `geminiToolCalls` creates each ID while the response is built, which is after the events
   went out. The stream path creates the ID when the part arrives and, after `geminiBuildResponse`, overwrites
   `resp.ToolCalls[i].ID` with those IDs. The order matches: the synthetic candidate holds the call parts in
   arrival order, and `geminiToolCalls` keeps that order. The non-streamed path is unchanged.
2. **Anthropic "ready"** is sent on `content_block_stop` for a block whose accumulated type is `tool_use`. Read the
   block from `msg.Content[ev.Index]` *after* `Accumulate`, which is also where its ID and name live.
3. **Folded follow-ups from the last plan** (from memory, `streaming-deferred-followups`), because these tasks
   touch the same code:
   - Task 2: a text block's initial `text` in `content_block_start` is sent as a `StreamText` event when non-empty,
     so the join rule can't break if Anthropic ever starts a block with text.
   - Task 6: move the README comment `// w is the http.ResponseWriter of the request being served.` from the first
     `llm` example (which has no `w`) to the Streaming example.
   The other deferred items stay deferred.

## File map

| File | Responsibility | Task |
| --- | --- | --- |
| `llm/llm.go` | New kinds, `StreamEvent` fields, `ToolEvents` type and constants, `Request.ToolEvents`, godoc; tracker filtering; rule-7 tool events | 1 |
| `llm/stream_test.go` | Router tests; `streamFake` gains scripted events and tool calls | 1 |
| `llm/anthropic.go` | Tool start on `content_block_start`, ready on `content_block_stop`; initial block text | 2 |
| `llm/anthropic_stream_test.go` | Anthropic tool-event tests | 2 |
| `llm/openai.go` | Tool start per index once ID and name are known; ready at `finish_reason` | 3 |
| `llm/openai_stream_test.go` | OpenAI tool-event tests | 3 |
| `llm/google.go` | ID on arrival, start + ready per call, IDs carried into `resp.ToolCalls` | 4 |
| `llm/google_stream_test.go` | Gemini tool-event tests | 4 |
| `guard/guard_test.go` | Pass-through tests (no guard code change) | 5 |
| `README.md` | Streaming section: field, kinds, app rules; move the misplaced `w` comment | 6 |

## Shared interfaces

**Public (`llm/llm.go`, Task 1):**

```go
const (
    StreamToolStart StreamKind = "tool_start"
    StreamToolReady StreamKind = "tool_ready"
)

type StreamEvent struct {
    Kind       StreamKind
    Text       string // StreamText: the new piece of text
    ToolCallID string // tool kinds: equals the ToolCall.ID in the final Response
    ToolName   string // tool kinds: the ToolDef.Name being called
}

type ToolEvents string

const (
    ToolEventsStart      ToolEvents = "start"
    ToolEventsStartReady ToolEvents = "start_ready"
)

// on Request, next to OnEvent:
ToolEvents ToolEvents // "" = no tool events. Ignored when OnEvent is nil.
```

**Internal (`llm/llm.go`, Task 1):**

```go
type streamTracker struct {
    next  func(StreamEvent)
    sent  bool
    tools ToolEvents // the request's ToolEvents
}
```

**Provider contract (Tasks 2–4).** When a provider has `onEvent`, it calls, for each app tool call:

```go
onEvent(StreamEvent{Kind: StreamToolStart, ToolCallID: id, ToolName: name})
onEvent(StreamEvent{Kind: StreamToolReady, ToolCallID: id, ToolName: name})
```

It sends both regardless of `req.ToolEvents`; the router filters. It checks `ctx.Err()` after each call to
`onEvent`. If it is set, it stops and returns the context error, exactly as the text path does in that provider.

## Model and review per task

| Task | Implement | Per-task review |
| --- | --- | --- |
| 1 Router | Sonnet | Sonnet |
| 2 Anthropic | Sonnet | Sonnet |
| 3 OpenAI | Sonnet | Sonnet |
| 4 Gemini | Sonnet | Sonnet |
| 5 Guard tests | Sonnet | none (tests only; final review covers it) |
| 6 README + gates | Haiku | none |
| Final whole-branch review | — | Opus, once |

Order: 1 first. 2, 3 and 4 depend only on 1. 5 after 1. 6 last.

---

### Task 1: API and router filtering

**Files:**
- Modify: `llm/llm.go` (types near `StreamEvent` / `Request`; `streamTracker`; `Complete`; `completeOn`)
- Test: `llm/stream_test.go`

**Interfaces:**
- Consumes: existing `StreamEvent`, `streamTracker`, `canStream`, rule-7 `synth` logic in `completeOn`.
- Produces: everything in "Shared interfaces → Public" and the `tools` field on `streamTracker`.

**Godoc.** On `Request.ToolEvents`: off by default; `ToolEventsStart` = option A, `ToolEventsStartReady` = option
B; ignored without `OnEvent`; any other value = off. On `OnEvent`, add the spec's six tool-event rules ("Rules for
tool events") after the existing five. On the two kinds and the two `StreamEvent` fields: what they carry.

**Router logic — exact.**

`Complete`, rule 3 (the tracker path) becomes:

```text
t := &streamTracker{next: req.OnEvent, tools: req.ToolEvents}
req.OnEvent = t.emit
return r.complete(ctx, req, t)
```

Rules 1 and 2 are unchanged. Rule 2 removes `OnEvent` before the providers run, so a `JSONSchema` request gets no
tool events.

`streamTracker.emit(ev)`:

```text
switch ev.Kind:
case StreamText:
    if ev.Text != "": t.sent = true
case StreamToolStart:
    if t.tools != ToolEventsStart && t.tools != ToolEventsStartReady: return   // drop
case StreamToolReady:
    if t.tools != ToolEventsStartReady: return                                 // drop
}
t.next(ev)
```

Tool kinds never touch `sent`. Unknown kinds pass on, as today.

`completeOn`, rule 7 (provider that can't stream), the success branch becomes:

```text
if err == nil && synth:
    if resp.Text != "": onEvent(StreamEvent{Kind: StreamText, Text: resp.Text})
    for _, tc := range resp.ToolCalls:
        onEvent(StreamEvent{Kind: StreamToolStart, ToolCallID: tc.ID, ToolName: tc.Name})
        onEvent(StreamEvent{Kind: StreamToolReady, ToolCallID: tc.ID, ToolName: tc.Name})
```

`onEvent` here is `t.emit`, which filters by mode. No events on error.

- [ ] **Step 1: Write the failing tests** in `llm/stream_test.go`. Extend `streamFake` with `script []StreamEvent`:
  when set, the fake sends exactly those events through `req.OnEvent` (in place of `pieces`), and its success text is
  the join of the script's `StreamText` texts. Tests set `resp.ToolCalls` through the existing `resp` field. Existing tests must not change behaviour. Cases:
  1. **Unset `ToolEvents`** → a script of `start(c1,search)`, text `"Hi"`, `ready(c1,search)` reaches `OnEvent` as
     only `["Hi"]`.
  2. **`ToolEventsStart`** → `start(c1)` and `"Hi"`, in that order; no ready.
  3. **`ToolEventsStartReady`** → all three, in order, with `ToolCallID`/`ToolName` intact.
  4. **Unknown value** (`ToolEvents("yes")`) → treated as off.
  5. **Tool event then failure before text → fallback still used.** Primary script `start(c1)`, err 503. Fallback
     answers `"ok"`. With `ToolEventsStart`, events are `start(c1)` then `"ok"`, and `resp.ToolCalls` comes from
     the fallback.
  6. **Provider that can't stream, reply with tool calls.** `fakeProvider.reply` has `Text: "Let me check"` and two
     `ToolCalls`. With `ToolEventsStartReady` → `"Let me check"`, `start(1)`, `ready(1)`, `start(2)`, `ready(2)`.
     With `ToolEventsStart` → text, `start(1)`, `start(2)`. With it unset → text only.
  7. **`JSONSchema` set** with `ToolEventsStartReady` → no tool events. The fake sees `OnEvent == nil`.
  8. **`OnEvent == nil`** with `ToolEventsStartReady` → no panic, same result as today.
- [ ] **Step 2: Run them to see them fail.** `go test ./llm -run 'Stream' -v` → compile errors (`StreamToolStart`
  undefined).
- [ ] **Step 3: Implement** the types, godoc, tracker filtering and rule-7 tool events above.
- [ ] **Step 4: Run the tests.** `go test ./llm -run 'Stream' -v` → PASS; `go test ./llm` → PASS.
- [ ] **Step 5: Commit.** `feat(llm): opt-in tool events on the stream`

---

### Task 2: Anthropic tool events

**Files:**
- Modify: `llm/anthropic.go` (`stream`)
- Test: `llm/anthropic_stream_test.go` (reuse `anthropicServer`, `sseEvent`, `msgStart`, `blockStart`,
  `blockDelta`, `blockStop`, `msgEnd`, the hold/quiet server helper)

**Interfaces:**
- Consumes: `StreamToolStart`, `StreamToolReady`, `StreamEvent.ToolCallID/ToolName` (Task 1).
- Produces: nothing new.

**What to build** (inside the existing `stream` loop, after `msg.Accumulate(ev)` succeeds, and only when
`onEvent != nil`):
- `ev.Type == "content_block_start"`:
  - block type `tool_use` → `StreamToolStart` with `ev.ContentBlock.ID` and `ev.ContentBlock.Name`. Check the
    exact SDK field names on `ContentBlockStartEventContentBlockUnion` in v1.56.0.
  - block type `text` with non-empty `ev.ContentBlock.Text` → `StreamText` with that text (folded follow-up,
    Decision 3).
  - any other type (including `server_tool_use`, `thinking`) → nothing.
- `ev.Type == "content_block_stop"` and `msg.Content[ev.Index].Type == "tool_use"` (bounds-checked) →
  `StreamToolReady` with that block's `ID` and `Name`.
- The existing `text_delta` emission stays as is.
- After every `onEvent` call: `if err := ctx.Err(); err != nil { return nil, err }`. Share one small helper for
  this if it keeps the loop readable.
- Ordering with `repairToolInputs` is unchanged: it runs before `Accumulate`, as today.

- [ ] **Step 1: Write the failing tests:**
  1. **One tool call:** stream = text block `"Let me check"`, then a `tool_use` block (`id: toolu_1`,
     `name: search_docs`, input split over two `input_json_delta`), `stop_reason: tool_use`. Events:
     `"Let me check"`, `start(toolu_1, search_docs)`, `ready(toolu_1, search_docs)`. IDs and names equal
     `resp.ToolCalls[0]`; the join rule holds.
  2. **Two parallel calls:** start1, ready1, start2, ready2 in block order; both in `resp.ToolCalls`.
  3. **`server_tool_use` (web search)** block plus its result block → no tool events.
  4. **Initial text in `content_block_start`:** a text block starting with `"Hel"` and a delta `"lo"` → events
     `"Hel"`, `"lo"`; `resp.Text == "Hello"`.
  5. **Cancel inside `OnEvent` on the start event**, with the rest of the stream already flushed in one write →
     `context.Canceled`, exactly one event.
- [ ] **Step 2: Run them to see them fail.** `go test ./llm -run 'Anthropic' -v` → new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.** `go test ./llm -run 'Anthropic' -v` → PASS; `go test ./llm` → PASS.
- [ ] **Step 5: Commit.** `feat(llm): Anthropic tool events on the stream`

---

### Task 3: OpenAI-compatible tool events

**Files:**
- Modify: `llm/openai.go` (`completeStream`)
- Test: `llm/openai_stream_test.go` (reuse its fake server and chunk helpers)

**Interfaces:**
- Consumes: Task 1 kinds and fields.
- Produces: nothing new.

**What to build** (in `completeStream`):
- Keep a set of started indexes. After merging a `tool_calls[]` piece into `calls[tc.Index]`: if that index is not
  started yet and the merged call has a non-empty `ID` **and** a non-empty `Function.Name` → `StreamToolStart`
  with them, and mark the index started.
- When a chunk's `finish_reason` is non-empty, the first time only → `StreamToolReady` for every started index, in
  ascending index order, with the merged ID and name.
- A call that never gets both ID and name sends no events; it still comes back in `resp.ToolCalls` as today.
- After every `onEvent` call, use the existing `ctx.Err()` check pattern from the text path.

- [ ] **Step 1: Write the failing tests:**
  1. **One call, pieces split:** first piece `id` + `name`, then three argument pieces, then `finish_reason:
     tool_calls`, `[DONE]` → `start(call_1, search_docs)` right after the first piece (assert it arrives before
     the second piece is read: record events and the server's write order, or check the event slice length inside
     the callback), then `ready(call_1)` after `finish_reason`. IDs and names equal `resp.ToolCalls`.
  2. **Two interleaved calls (index 0 and 1)** → start0, start1 in first-piece order; ready0, ready1 at
     `finish_reason`.
  3. **Name arrives in a later piece than the ID** → start sent when the name arrives.
  4. **A call with no name ever** → no events for it; it is still in `resp.ToolCalls`.
  5. **Text then a call** → text events, then start, then ready; the join rule holds.
  6. **Cancel inside `OnEvent` on the start event**, with the rest already flushed → `context.Canceled`, exactly one
     event.
- [ ] **Step 2: Run them to see them fail.** `go test ./llm -run 'OpenAIStream' -v` → new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.** `go test ./llm -run 'OpenAIStream' -v` → PASS; `go test ./llm` → PASS.
- [ ] **Step 5: Commit.** `feat(llm): OpenAI-compatible tool events on the stream`

---

### Task 4: Gemini tool events

**Files:**
- Modify: `llm/google.go` (`sendStream`)
- Test: `llm/google_stream_test.go` (reuse its fake server)

**Interfaces:**
- Consumes: Task 1 kinds and fields; existing `geminiCallID()`, `geminiBuildResponse`.
- Produces: nothing new.

**What to build** (in `sendStream`; the non-streamed path is unchanged):
- Keep `ids []string` next to the collected call parts. For each `functionCall` part as it arrives:
  `id := geminiCallID()`, append the part and `id`, then send `StreamToolStart` and then `StreamToolReady` with `id`
  and `p.FunctionCall.Name`, checking `ctx.Err()` after each.
- After `resp, err := geminiBuildResponse(model, synthetic)`: if `len(resp.ToolCalls) == len(ids)`, set
  `resp.ToolCalls[i].ID = ids[i]` for each `i`. Do this on both the success and the refusal return. If the lengths
  differ (it can't, by construction), leave the IDs as built and add a short comment saying why the check is there.

- [ ] **Step 1: Write the failing tests:**
  1. **One call:** event with a `functionCall` part (`name: search_docs`, args) and `finishReason: STOP` → `start`,
     `ready`, back to back. `ToolCallID` is non-empty and equals `resp.ToolCalls[0].ID`; `ToolName` equals its
     `Name`.
  2. **Two calls across two events** → start1, ready1, start2, ready2; both IDs match `resp.ToolCalls` in order; the
     IDs differ from each other.
  3. **Text then a call** → text event(s), then start, ready; the join rule holds.
  4. **Google Search grounding only** (no `functionCall` parts) → no tool events.
  5. **Cancel inside `OnEvent` on the start event**, with the rest already flushed → `context.Canceled`, exactly one
     event.
- [ ] **Step 2: Run them to see them fail.** `go test ./llm -run 'GeminiStream' -v` → new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.** `go test ./llm -run 'GeminiStream' -v` → PASS; `go test ./llm` → PASS.
- [ ] **Step 5: Commit.** `feat(llm): Gemini tool events on the stream`

---

### Task 5: Guard pass-through tests

**Files:**
- Test: `guard/guard_test.go`. **No change to `guard/guard.go` or `guard/stream.go` is expected.** If a test fails,
  stop and report it (BLOCKED) with the failing output; don't change guard code on your own.

**Interfaces:**
- Consumes: Task 1 kinds and fields; existing `fakeLLM` (`pieces`, `err`).

**What to build:** let `fakeLLM` send a scripted list of `llm.StreamEvent`s (mixed text and tool events) when
`OnEvent` is set, in the same way it sends `pieces` today. Keep `pieces` working.

- [ ] **Step 1: Write the tests:**
  1. **Conversational with marker:** script `M[:5]`, `start(c1, search)`, `M[5:]+" Maaf"`, `ready(c1, search)` →
     the caller sees `start(c1)` and `ready(c1)` unchanged. The text events joined equal `"Maaf"` == `resp.Text`.
     One `EventRefusal`.
  2. **Conversational without marker:** tool events and text pass through. Text joined == `resp.Text`.
  3. **One-shot and background:** the script passes through untouched, marker included.
  4. **Inner client that sends only tool events and returns text** (does not stream text) → the existing "inner
     does not stream" branch still strips the marker and sends the text once. Tool events still reach the caller.
- [ ] **Step 2: Run them.** `go test ./guard -v` → PASS (no guard code change expected).
- [ ] **Step 3: Commit.** `test(guard): tool events pass through the marker filter`

---

### Task 6: README and gates

**Files:**
- Modify: `README.md` — the `### Streaming` section under `## aikit/llm`.

**Content:**
- **Tool events:** what they're for (show "Calling search_docs…" while the model writes a call) and that running
  the tool is the app's job.
- **The field:** `Request.ToolEvents`. Off by default; `llm.ToolEventsStart` = start only;
  `llm.ToolEventsStartReady` = start and ready.
- **A short example:** extend the existing streaming example's callback with a `switch e.Kind` that handles
  `StreamToolStart` (show an indicator using `e.ToolName`, keyed by `e.ToolCallID`).
- **The app rules,** same wording as the godoc: start before ready; IDs match `resp.ToolCalls`; only the app's own
  tools; when `Complete` returns, `resp.ToolCalls` is the truth, so clear indicators not in it; "ready" doesn't
  promise the call is usable (check `StopTruncated`); a tool event may arrive slightly before text the guard filter
  is holding back.
- **Folded follow-up:** move `// w is the http.ResponseWriter of the request being served.` from the first `llm`
  example to the Streaming example.

- [ ] **Step 1: Write the section and move the comment.**
- [ ] **Step 2: Run the gates** (full suite, once):

  ```sh
  go vet ./...
  go test ./...
  go test -race ./...
  ```

  Expected: all clean. If any fails, stop and report the output. Don't fix code in this task.
- [ ] **Step 3: Commit.** `docs: tool events in the streaming section`

---

## Final review (controller)

One Opus whole-branch review against the spec. Check in particular:

- The filter in `streamTracker.emit` matches Task 1 exactly. Tool kinds never set `sent`.
- On every provider, IDs and names in events match `resp.ToolCalls`; start comes before ready; each at most once.
- No events for built-in tools. `ctx.Err()` is checked after every `OnEvent` call on all three providers.
- `ToolEvents` unset paths are unchanged (no event reaches the app that didn't before).
- README and godoc agree with behaviour.

## Spec coverage check

| Spec item | Task |
| --- | --- |
| New kinds, `StreamEvent` fields, `ToolEvents`, `Request.ToolEvents`, unknown = off | 1 |
| Tool-event rules 1–6 in godoc | 1 |
| Router filtering in the tracker; tool events don't set `sent` | 1 |
| `JSONSchema`: no tool events | 1 (test 7) |
| Rule 7: tool events after the text event | 1 (test 6) |
| Anthropic start/ready; `server_tool_use` ignored | 2 |
| OpenAI start once ID + name known; ready at `finish_reason`; nameless call sends nothing | 3 |
| Gemini ID on arrival, start + ready back to back, final ID matches | 4 |
| `ctx.Err()` after every `OnEvent` | 2, 3, 4 |
| Guard: pass-through, no code change | 5 |
| README | 6 |
| Gates | 6 |
