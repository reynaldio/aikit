# Design: tool calling in `aikit/llm`

- **Status:** Implemented in v0.3.0
- **Target:** v0.3.0
- **Date:** 2026-08-17

## Why

`llm` today does single-shot completions. `Request` carries messages, effort, a
JSON schema and a web-search flag; `Response` carries text and token counts.
There is no way to declare a tool, receive a call, or answer one.

That was enough for the first consumer, which only ever needed one-shot
completions. The second consumer is an agent platform whose entire architecture
is tools: its agents dispatch scanners, file reports, record validations and
recommend fixes through them, and two of its product invariants — a report
cannot exist without evidence, and no agent may validate its own report — are
enforced inside tool handlers. Without tool calling there is nowhere for those
rules to live.

Function calling exists on all three backends, so it belongs in the seam rather
than in one consumer.

## Types

Five additions. Nothing is removed or renamed, so existing callers are unaffected.

```go
// ToolDef declares a tool the model may call.
type ToolDef struct {
    Name        string
    Description string
    Schema      map[string]any // JSON Schema for the input object
}

// ToolCall is the model asking for one invocation. ID is provider-assigned and
// opaque — echo it back verbatim on the matching ToolResult.
type ToolCall struct {
    ID    string
    Name  string
    Input json.RawMessage
}

// ToolResult answers exactly one ToolCall. IsError reports that the tool failed;
// the model sees the failure and can adapt rather than being told nothing.
type ToolResult struct {
    ToolCallID string
    Content    string
    IsError    bool
}

// StopReason is why generation ended.
type StopReason string

const (
    StopEndTurn   StopReason = "end_turn"
    StopToolUse   StopReason = "tool_use"
    StopTruncated StopReason = "truncated"
)
```

`Input` is `json.RawMessage`, not `map[string]any`: a map decodes every JSON
number to `float64`, so a tool taking an integer id receives a lossy value that
looks fine until it doesn't. Raw JSON lets each handler unmarshal into its own
type and get the compiler's help.

Extensions to existing types, all additive:

| Type | New field | Set on |
|---|---|---|
| `Request` | `Tools []ToolDef` | any request that offers tools |
| `Message` | `ToolCalls []ToolCall` | assistant turns being echoed back |
| `Message` | `ToolResults []ToolResult` | the user turn answering a round |
| `Response` | `ToolCalls []ToolCall` | when the model wants tools |
| `Response` | `StopReason StopReason` | every response |

### Why results nest on a user turn

All results from one round live in a single `Message.ToolResults` slice rather
than one message per result.

Anthropic requires every `tool_result` from a round to arrive in **one** user
message; splitting them across messages does not error — it silently teaches the
model to stop making parallel calls. The nested shape makes that mistake
unrepresentable instead of merely discouraged. Adapters fan out where a provider
wants one message per result.

This matters for the real workload: an agent dispatching several scanners in one
round depends on parallel calls continuing to work.

## Round trip

```go
resp, err := c.Complete(ctx, llm.Request{
    Tools:    []llm.ToolDef{dispatchScan, fileReport},
    Messages: msgs,
})
// resp.StopReason == StopToolUse
// resp.ToolCalls  == [{ID:"tu_01", Name:"dispatch_scan", Input:{…}}, …]

results := make([]llm.ToolResult, 0, len(resp.ToolCalls))
for _, call := range resp.ToolCalls {
    out, err := run(call)                      // caller's dispatch
    results = append(results, llm.ToolResult{
        ToolCallID: call.ID,
        Content:    out,
        IsError:    err != nil,
    })
}

msgs = append(msgs,
    llm.Message{Role: "assistant", ToolCalls: resp.ToolCalls},
    llm.Message{Role: "user", ToolResults: results},
)
// …call Complete again
```

## Provider mapping

| Concept | Anthropic | OpenAI | Google |
|---|---|---|---|
| Declare | `ToolParam{Name, Description, InputSchema}` wrapped in `ToolUnionParam{OfTool:}` | `tools[].function{name, description, parameters}` | `tools[].functionDeclarations[]` |
| Receive | `ToolUseBlock{ID, Name, Input}` | `choices[].message.tool_calls[]` | `candidates[].content.parts[].functionCall` |
| Answer | `ToolResultBlockParam{ToolUseID, Content, IsError}` in one user turn | one `role:"tool"` message per result | `parts[].functionResponse` in one user turn |
| Stop reason | `Message.StopReason` | `choices[].finish_reason` | `candidates[].finishReason` |

Only the Anthropic column is verified against the SDK in the module cache; it
uses the official client. The other two are hand-rolled JSON against the live
APIs, so their exact field names must be confirmed against current provider docs
during implementation rather than trusted from this table.

Two existing structures absorb most of the change: `geminiTool` already exists
(carrying only `google_search` today) and gains `functionDeclarations`, and
`geminiResponse.Candidates[].Content.Parts` already models parts, so function
calls arrive there naturally.

**Neither the OpenAI nor the Gemini response struct currently parses a finish
reason at all.** `StopReason` therefore adds a field to each, not just a mapping.

### The Google asymmetry

Gemini has no tool-call ID. It matches a `functionCall` to its `functionResponse`
by name and position — ambiguous in exactly the case that matters, two parallel
calls to the *same* tool.

The resolution rests on a property all three APIs share: **they are stateless, so
an ID only has to be consistent within a single request.** aikit therefore treats
IDs as opaque strings and passes them through. Anthropic and OpenAI use their own
natively.

The Google adapter synthesizes an ID per call when it parses a `functionCall`
part. On the way back it resolves each `ToolResult` by finding the `ToolCall`
with a matching `ID` in the **preceding assistant turn** — that entry supplies
the function name, and its index in that turn's `ToolCalls` supplies the
position. Both facts come from the message history, never from the ID itself.

**The synthesized ID's format is deliberately unspecified, and no code may parse
it.** Resolution is by equality against the assistant turn, so any unique string
works — which is precisely what lets a foreign ID survive a failover.

That lookup is also what makes a **mid-loop cross-provider failover** work: after
a failover the history carries the previous provider's IDs, and since nothing
parses them, a foreign ID is still just a consistent string.

## Out of scope

**No loop.** `Complete` does one round. The caller executes tools and calls
again. This is deliberate: the consumer driving this design keeps an append-only
ledger row per tool call and must observe every round, and it already owns a
bounded-rounds loop by its own conventions. A loop inside `llm` would hide the
rounds it needs and duplicate the one it has.

**No `ToolChoice`.** All three providers default to automatic selection, which is
what an agent loop wants. Forcing no-tools for a final summarising round is
already possible by omitting `Tools` on that call. Additive later.

**No `strict` tool schemas.** Anthropic and OpenAI can guarantee tool inputs
validate; Google cannot. It saves a retry, not a correctness property — the
consumer's invariants are enforced server-side in the handler regardless.
Additive later.

## Failure modes

| Situation | Behaviour |
|---|---|
| Model wants tools | `StopToolUse`, `ToolCalls` populated, `Text` may also be set — providers can emit prose alongside a call. Echo both back on the assistant turn: `Message{Role:"assistant", Content: resp.Text, ToolCalls: resp.ToolCalls}` |
| Turn cut off | `StopTruncated`. A caller treating this as `StopEndTurn` records incomplete work as finished |
| Tool failed | Caller sets `IsError` on the result; the model sees it and adapts |
| Refusal mid-loop | Unchanged from v0.2.0 — `ErrRefused`, failover to the profile's fallback, history intact |
| Provider returns a call for an undeclared tool | Passed through as-is. The caller dispatches and is the only party that knows its own tool set; inventing an error here would second-guess it |

## Testing

Extends the existing `fakeProvider` harness in `llm_test.go`.

- Round trip per adapter: a `ToolDef` serialises, a provider call deserialises
  into `ToolCall`, and a `ToolResult` serialises back.
- **Parallel calls to the same tool name through the Google adapter** — the case
  its lack of IDs makes ambiguous, and the reason the lookup exists.
- `StopTruncated` surfaces rather than reading as a completed turn.
- Failover mid-conversation with tool history carrying a foreign ID format.
- Existing tests must pass untouched; every change here is additive.

## Compatibility

Additive only — new fields on existing structs, new types, no signature changes.
Callers that never set `Tools` see no behavioural change, so the existing
consumer is unaffected and needs no coordinated release.
