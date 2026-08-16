# Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provider-agnostic tool (function) calling to `aikit/llm` across all three backends, so a caller can declare tools, receive calls, and answer them.

**Architecture:** Five new types in `llm.go` plus additive fields on `Request`, `Message` and `Response`. Each provider file gains **pure mapping functions** — declarations out, calls in, results out — which are what the tests exercise; no test makes a network call. `Complete` stays single-round: the caller runs the loop.

**Tech Stack:** Go 1.25, `github.com/anthropics/anthropic-sdk-go` v1.56.0 (Anthropic), hand-rolled JSON over `net/http` (Google, OpenAI).

## Global Constraints

- **Additive only.** No existing type, field, function signature, or behaviour changes. Every pre-existing test in `llm/llm_test.go` must pass untouched after every task.
- **The design spec is `docs/tool-calling.md`.** Where this plan and the spec disagree, the spec wins — fix the plan.
- **Tool-call IDs are opaque.** No code may parse, or depend on the format of, a `ToolCall.ID`.
- **No network in tests.** Test the pure mapping functions; the existing `fakeProvider` covers router behaviour.
- Run `go build ./... && go vet ./... && go test ./...` before every commit.
- Verified SDK signatures (v1.56.0) — use exactly these:
  - `anthropic.ToolUnionParam{OfTool: *anthropic.ToolParam}`
  - `anthropic.ToolParam{Name string; Description param.Opt[string]; InputSchema ToolInputSchemaParam}`
  - `anthropic.ToolInputSchemaParam{Properties any; Required []string}`
  - `anthropic.ToolUseBlock{ID string; Name string; Input json.RawMessage}`
  - `anthropic.NewToolUseBlock(id string, input any, name string) ContentBlockParamUnion`
  - `anthropic.NewToolResultBlock(toolUseID string, content string, isError bool) ContentBlockParamUnion`
  - `anthropic.StopReasonToolUse` / `StopReasonMaxTokens` / `StopReasonEndTurn`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `llm/llm.go` | Public types; router | Add 4 types + 1 enum; add fields to `Request`/`Message`/`Response` |
| `llm/anthropic.go` | Anthropic adapter | Add 4 pure mapping funcs; wire into `complete` |
| `llm/openai.go` | OpenAI adapter | Add wire structs + 3 mapping funcs; wire into `complete` |
| `llm/google.go` | Gemini adapter | Add wire structs + 3 mapping funcs incl. ID synthesis |
| `llm/llm_test.go` | All tests | Add per-task tests |
| `README.md` | Docs | Add a tool-calling section |

`llm/noop.go` needs **no change** — it returns `ErrNotConfigured` before touching a request.

---

### Task 1: Core types and the Anthropic adapter

**Files:**
- Modify: `llm/llm.go` (types + `Request`/`Message`/`Response` fields)
- Modify: `llm/anthropic.go` (mapping funcs + `complete`)
- Test: `llm/llm_test.go`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `ToolDef{Name string, Description string, Schema map[string]any}`; `ToolCall{ID string, Name string, Input json.RawMessage}`; `ToolResult{ToolCallID string, Content string, IsError bool}`; `StopReason string` with constants `StopEndTurn`/`StopToolUse`/`StopTruncated`; `Request.Tools []ToolDef`; `Message.ToolCalls []ToolCall`; `Message.ToolResults []ToolResult`; `Response.ToolCalls []ToolCall`; `Response.StopReason StopReason`. Tasks 2 and 3 map these onto their own providers.

- [ ] **Step 1: Write the failing test**

Append to `llm/llm_test.go`:

```go
func TestAnthropicToolDeclarationMapping(t *testing.T) {
	defs := []ToolDef{{
		Name:        "file_report",
		Description: "File a vulnerability report",
		Schema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"title": map[string]any{"type": "string"}},
			"required":   []any{"title"}, // []any: what JSON decoding yields
		},
	}}
	got := anthropicTools(defs)
	if len(got) != 1 || got[0].OfTool == nil {
		t.Fatalf("expected one tool in the union, got %#v", got)
	}
	tp := got[0].OfTool
	if tp.Name != "file_report" {
		t.Fatalf("name = %q", tp.Name)
	}
	if tp.InputSchema.Properties == nil {
		t.Fatal("properties must be lifted out of the schema")
	}
	if len(tp.InputSchema.Required) != 1 || tp.InputSchema.Required[0] != "title" {
		t.Fatalf("required = %#v; []any must be coerced to []string", tp.InputSchema.Required)
	}
}

func TestAnthropicStopReasonMapping(t *testing.T) {
	// max_tokens must NOT read as a completed turn — that is the whole reason
	// StopReason exists.
	cases := map[anthropic.StopReason]StopReason{
		anthropic.StopReasonToolUse:   StopToolUse,
		anthropic.StopReasonMaxTokens: StopTruncated,
		anthropic.StopReasonEndTurn:   StopEndTurn,
	}
	for in, want := range cases {
		if got := anthropicStopReason(in); got != want {
			t.Fatalf("%s → %s, want %s", in, got, want)
		}
	}
}

func TestAnthropicToolResultsMapping(t *testing.T) {
	blocks := anthropicToolResults([]ToolResult{
		{ToolCallID: "tu_01", Content: "ok"},
		{ToolCallID: "tu_02", Content: "boom", IsError: true},
	})
	if len(blocks) != 2 {
		t.Fatalf("expected one block per result, got %d", len(blocks))
	}
}
```

Add `"github.com/anthropics/anthropic-sdk-go"` to the test file's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm && go test ./... -run TestAnthropic -v`
Expected: FAIL — `undefined: anthropicTools`, `undefined: ToolDef`, `undefined: anthropicStopReason`, `undefined: anthropicToolResults`.

- [ ] **Step 3: Add the types to `llm/llm.go`**

Add `"encoding/json"` to the import block. Insert after the `Effort` constants:

```go
// ToolDef declares a tool the model may call. Schema is the JSON Schema of the
// input object — the whole schema, which each adapter reshapes as its provider
// requires.
type ToolDef struct {
	Name        string
	Description string
	Schema      map[string]any
}

// ToolCall is the model asking for one invocation. ID is provider-assigned and
// OPAQUE: echo it back verbatim on the matching ToolResult and never parse it.
type ToolCall struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// ToolResult answers exactly one ToolCall. IsError reports that the tool failed,
// so the model can adapt rather than being told nothing.
type ToolResult struct {
	ToolCallID string
	Content    string
	IsError    bool
}

// StopReason is why generation ended. Truncation matters: a turn cut off at the
// token ceiling otherwise reads exactly like a finished one.
type StopReason string

const (
	StopEndTurn   StopReason = "end_turn"
	StopToolUse   StopReason = "tool_use"
	StopTruncated StopReason = "truncated"
)
```

In `Request`, after `JSONSchema`:

```go
	// Tools the model may call this turn. Complete does ONE round: a reply with
	// ToolCalls means the caller should run them, append an assistant turn
	// echoing the calls plus a user turn carrying every result, and call again.
	Tools []ToolDef
```

In `Message`, after `Documents`:

```go
	// ToolCalls belong to an assistant turn being echoed back.
	ToolCalls []ToolCall
	// ToolResults belong to the user turn answering a round. Every result from
	// one round goes in ONE message: providers that want them together reject a
	// split, and Anthropic silently stops making parallel calls instead.
	ToolResults []ToolResult
```

In `Response`, after `CachedTokens`:

```go
	// ToolCalls is non-empty when the model wants tools run.
	ToolCalls []ToolCall
	// StopReason is why generation ended. Treating StopTruncated as StopEndTurn
	// records incomplete work as finished.
	StopReason StopReason
```

- [ ] **Step 4: Add the mapping functions to `llm/anthropic.go`**

Add `"encoding/json"` to the imports. Append:

```go
// anthropicInputSchema lifts properties/required out of a JSON Schema, which is
// how the SDK models a tool's input. `required` arrives as []string from a Go
// literal but []any once it has been through JSON, so both are accepted.
func anthropicInputSchema(schema map[string]any) anthropic.ToolInputSchemaParam {
	out := anthropic.ToolInputSchemaParam{}
	if props, ok := schema["properties"]; ok {
		out.Properties = props
	}
	switch req := schema["required"].(type) {
	case []string:
		out.Required = req
	case []any:
		for _, r := range req {
			if s, ok := r.(string); ok {
				out.Required = append(out.Required, s)
			}
		}
	}
	return out
}

// anthropicTools maps tool declarations onto the SDK's tool union.
func anthropicTools(defs []ToolDef) []anthropic.ToolUnionParam {
	if len(defs) == 0 {
		return nil
	}
	out := make([]anthropic.ToolUnionParam, 0, len(defs))
	for _, d := range defs {
		t := anthropic.ToolParam{
			Name:        d.Name,
			InputSchema: anthropicInputSchema(d.Schema),
		}
		if d.Description != "" {
			t.Description = anthropic.String(d.Description)
		}
		out = append(out, anthropic.ToolUnionParam{OfTool: &t})
	}
	return out
}

// anthropicToolResults renders one round's results as content blocks. The caller
// puts them all in a single user turn.
func anthropicToolResults(results []ToolResult) []anthropic.ContentBlockParamUnion {
	out := make([]anthropic.ContentBlockParamUnion, 0, len(results))
	for _, r := range results {
		out = append(out, anthropic.NewToolResultBlock(r.ToolCallID, r.Content, r.IsError))
	}
	return out
}

// anthropicToolCalls pulls tool_use blocks out of a reply.
func anthropicToolCalls(content []anthropic.ContentBlockUnion) []ToolCall {
	var out []ToolCall
	for _, block := range content {
		if tu, ok := block.AsAny().(anthropic.ToolUseBlock); ok {
			out = append(out, ToolCall{ID: tu.ID, Name: tu.Name, Input: tu.Input})
		}
	}
	return out
}

// anthropicStopReason maps the SDK's stop reason onto ours. Refusal is handled
// as an error before this is reached, so it has no case here.
func anthropicStopReason(sr anthropic.StopReason) StopReason {
	switch sr {
	case anthropic.StopReasonToolUse:
		return StopToolUse
	case anthropic.StopReasonMaxTokens:
		return StopTruncated
	default:
		return StopEndTurn
	}
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd llm && go test ./... -run TestAnthropic -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Wire the mappings into `complete` and `splitSystemAndMessages`**

In `splitSystemAndMessages`, replace the `case "assistant":` and `default:` arms:

```go
		case "assistant":
			blocks := []anthropic.ContentBlockParamUnion{}
			if m.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, tc.Input, tc.Name))
			}
			if len(blocks) > 0 {
				msgs = append(msgs, anthropic.NewAssistantMessage(blocks...))
			}
		default: // "user"
			blocks := userBlocks(m)
			blocks = append(blocks, anthropicToolResults(m.ToolResults)...)
			if len(blocks) > 0 {
				msgs = append(msgs, anthropic.NewUserMessage(blocks...))
			}
```

In `complete`, after the `params.OutputConfig` block:

```go
	if tools := anthropicTools(req.Tools); len(tools) > 0 {
		params.Tools = append(params.Tools, tools...)
	}
```

(`append`, not assignment — `req.WebSearch` may already have populated `params.Tools`.)

Then extend the `out` literal:

```go
	out := Response{
		Text:         text,
		InputTokens:  int(resp.Usage.InputTokens),
		OutputTokens: int(resp.Usage.OutputTokens),
		CachedTokens: int(resp.Usage.CacheReadInputTokens),
		ToolCalls:    anthropicToolCalls(resp.Content),
		StopReason:   anthropicStopReason(resp.StopReason),
	}
```

- [ ] **Step 7: Add the failover-with-tool-history test**

Append to `llm/llm_test.go`:

```go
func TestFailoverCarriesForeignToolCallIDs(t *testing.T) {
	// After a mid-loop failover the history holds the PREVIOUS provider's ID
	// format. Nothing may parse it — it only has to stay internally consistent.
	g := &fakeProvider{fail: map[string]error{"flash": errors.New("gemini: overloaded (status 503)")}}
	a := &fakeProvider{}
	r := newTestRouter(a, g)
	_, err := r.Complete(context.Background(), Request{
		Task:  TaskChat,
		Tools: []ToolDef{{Name: "dispatch_scan", Schema: map[string]any{"type": "object"}}},
		Messages: []Message{
			{Role: "user", Content: "scan it"},
			{Role: "assistant", ToolCalls: []ToolCall{{ID: "toolu_01FOREIGN", Name: "dispatch_scan"}}},
			{Role: "user", ToolResults: []ToolResult{{ToolCallID: "toolu_01FOREIGN", Content: "done"}}},
		},
	})
	if err != nil {
		t.Fatalf("a foreign tool-call ID must survive failover, got: %v", err)
	}
	if len(a.calls) != 1 || a.calls[0] != "haiku" {
		t.Fatalf("expected the fallback to answer, got %v", a.calls)
	}
}
```

- [ ] **Step 8: Verify everything passes**

Run: `cd .. && go build ./... && go vet ./... && go test ./...`
Expected: PASS, including every pre-existing test unchanged.

- [ ] **Step 9: Commit**

```bash
git add llm/llm.go llm/anthropic.go llm/llm_test.go
git commit -m "Add tool calling types and the Anthropic adapter"
```

---

### Task 2: OpenAI adapter

**Files:**
- Modify: `llm/openai.go`
- Test: `llm/llm_test.go`

**Interfaces:**
- Consumes: `ToolDef`, `ToolCall`, `ToolResult`, `StopReason` and the constants from Task 1.
- Produces: `oaiToolCalls([]oaiToolCall) []ToolCall`, `oaiStopReason(string) StopReason`, and the `oaiTool` / `oaiToolCall` wire structs. Nothing later depends on these.

> **Verify first.** The wire field names below are from the Chat Completions
> shape, not from a vendored SDK. Confirm them against current OpenAI docs
> before implementing; if they differ, the struct tags change and the mapping
> logic does not.

- [ ] **Step 1: Write the failing test**

```go
func TestOpenAIToolCallMapping(t *testing.T) {
	calls := oaiToolCalls([]oaiToolCall{
		{ID: "call_1", Type: "function", Function: oaiToolCallFunc{
			Name: "dispatch_scan", Arguments: `{"host":"a.example"}`,
		}},
	})
	if len(calls) != 1 {
		t.Fatalf("expected one call, got %d", len(calls))
	}
	if calls[0].ID != "call_1" || calls[0].Name != "dispatch_scan" {
		t.Fatalf("got %#v", calls[0])
	}
	var in struct {
		Host string `json:"host"`
	}
	if err := json.Unmarshal(calls[0].Input, &in); err != nil || in.Host != "a.example" {
		t.Fatalf("arguments must land in Input as raw JSON: %v / %#v", err, in)
	}
}

func TestOpenAIStopReasonMapping(t *testing.T) {
	cases := map[string]StopReason{
		"tool_calls": StopToolUse,
		"length":     StopTruncated,
		"stop":       StopEndTurn,
		"":           StopEndTurn,
	}
	for in, want := range cases {
		if got := oaiStopReason(in); got != want {
			t.Fatalf("%q → %s, want %s", in, got, want)
		}
	}
}
```

Add `"encoding/json"` to the test imports if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm && go test ./... -run TestOpenAI -v`
Expected: FAIL — `undefined: oaiToolCalls`, `undefined: oaiToolCall`, `undefined: oaiStopReason`.

- [ ] **Step 3: Add the wire structs and mappings to `llm/openai.go`**

```go
type oaiToolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON encoded as a string
}

type oaiToolCall struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Function oaiToolCallFunc `json:"function"`
}

type oaiToolFunc struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type oaiTool struct {
	Type     string      `json:"type"` // always "function"
	Function oaiToolFunc `json:"function"`
}

// oaiTools maps declarations onto the wire shape.
func oaiTools(defs []ToolDef) []oaiTool {
	if len(defs) == 0 {
		return nil
	}
	out := make([]oaiTool, 0, len(defs))
	for _, d := range defs {
		out = append(out, oaiTool{
			Type: "function",
			Function: oaiToolFunc{
				Name: d.Name, Description: d.Description, Parameters: d.Schema,
			},
		})
	}
	return out
}

// oaiToolCalls maps a reply's tool calls. Arguments arrive as a JSON *string*,
// so the bytes go straight into Input.
func oaiToolCalls(in []oaiToolCall) []ToolCall {
	var out []ToolCall
	for _, c := range in {
		out = append(out, ToolCall{
			ID: c.ID, Name: c.Function.Name, Input: json.RawMessage(c.Function.Arguments),
		})
	}
	return out
}

// oaiStopReason maps finish_reason. An unknown or absent value reads as a
// completed turn rather than inventing a failure.
func oaiStopReason(fr string) StopReason {
	switch fr {
	case "tool_calls":
		return StopToolUse
	case "length":
		return StopTruncated
	default:
		return StopEndTurn
	}
}
```

Add `"encoding/json"` to the file's imports.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd llm && go test ./... -run TestOpenAI -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Extend the request/response structs and `complete`**

On `oaiMessage`, add:

```go
	ToolCalls  []oaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
```

On `oaiRequest`, add:

```go
	Tools []oaiTool `json:"tools,omitempty"`
```

On `oaiResponse`'s anonymous `Choices` element, add `FinishReason string \`json:"finish_reason"\`` beside `Message`, and add `ToolCalls []oaiToolCall \`json:"tool_calls"\`` inside its `Message` struct.

In `complete`, when building messages, after the existing per-message handling add — for an assistant turn — `ToolCalls: m.ToolCalls` mapped back to `[]oaiToolCall`, and for each `ToolResult` in a user turn append a separate message (this is the fan-out the spec describes):

```go
		for _, tr := range m.ToolResults {
			msgs = append(msgs, oaiMessage{
				Role: "tool", ToolCallID: tr.ToolCallID, Content: tr.Content,
			})
		}
```

Set `body.Tools = oaiTools(req.Tools)`, and populate the response:

```go
	if len(out.Choices) > 0 {
		text = out.Choices[0].Message.Content
		resp.ToolCalls = oaiToolCalls(out.Choices[0].Message.ToolCalls)
		resp.StopReason = oaiStopReason(out.Choices[0].FinishReason)
	}
```

- [ ] **Step 6: Verify everything passes**

Run: `cd .. && go build ./... && go vet ./... && go test ./...`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add llm/openai.go llm/llm_test.go
git commit -m "Map tool calling onto the OpenAI adapter"
```

---

### Task 3: Google adapter and synthesized IDs

**Files:**
- Modify: `llm/google.go`
- Test: `llm/llm_test.go`

**Interfaces:**
- Consumes: `ToolDef`, `ToolCall`, `ToolResult`, `StopReason` from Task 1.
- Produces: `geminiToolCalls([]geminiPart) []ToolCall`, `geminiToolResponses([]ToolResult, []ToolCall) []geminiPart`, `geminiStopReason(string) StopReason`.

**This is the task the ID design exists for.** Gemini has no tool-call ID and matches calls to responses by name and position, which is ambiguous for two parallel calls to the *same* tool. `geminiToolResponses` therefore takes the **preceding assistant turn's `ToolCalls`** and resolves each result through it.

- [ ] **Step 1: Write the failing test**

```go
func TestGeminiParallelCallsToSameToolResolve(t *testing.T) {
	// Gemini has no call IDs. Two calls to the SAME tool in one round is the
	// case its name-and-position matching cannot disambiguate on its own.
	parts := []geminiPart{
		{FunctionCall: &geminiFunctionCall{Name: "dispatch_scan", Args: map[string]any{"host": "a.example"}}},
		{FunctionCall: &geminiFunctionCall{Name: "dispatch_scan", Args: map[string]any{"host": "b.example"}}},
	}
	calls := geminiToolCalls(parts)
	if len(calls) != 2 {
		t.Fatalf("expected two calls, got %d", len(calls))
	}
	if calls[0].ID == calls[1].ID || calls[0].ID == "" {
		t.Fatalf("synthesized IDs must be unique and non-empty: %q / %q", calls[0].ID, calls[1].ID)
	}

	// Answer them OUT OF ORDER — resolution is by ID, never by result order.
	got := geminiToolResponses([]ToolResult{
		{ToolCallID: calls[1].ID, Content: "b done"},
		{ToolCallID: calls[0].ID, Content: "a done"},
	}, calls)
	if len(got) != 2 {
		t.Fatalf("expected two response parts, got %d", len(got))
	}
	// Position must follow the CALL order, not the result order.
	if got[0].FunctionResponse == nil || got[0].FunctionResponse.Response["content"] != "a done" {
		t.Fatalf("part 0 must answer call 0: %#v", got[0].FunctionResponse)
	}
	if got[1].FunctionResponse == nil || got[1].FunctionResponse.Response["content"] != "b done" {
		t.Fatalf("part 1 must answer call 1: %#v", got[1].FunctionResponse)
	}
	if got[0].FunctionResponse.Name != "dispatch_scan" {
		t.Fatalf("name must come from the call, got %q", got[0].FunctionResponse.Name)
	}
}

func TestGeminiStopReasonMapping(t *testing.T) {
	cases := map[string]StopReason{
		"MAX_TOKENS": StopTruncated,
		"STOP":       StopEndTurn,
		"":           StopEndTurn,
	}
	for in, want := range cases {
		if got := geminiStopReason(in); got != want {
			t.Fatalf("%q → %s, want %s", in, got, want)
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd llm && go test ./... -run TestGemini -v`
Expected: FAIL — `undefined: geminiFunctionCall`, `undefined: geminiToolCalls`, `undefined: geminiToolResponses`, `undefined: geminiStopReason`.

- [ ] **Step 3: Add the wire structs and mappings to `llm/google.go`**

```go
type geminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

type geminiFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type geminiFunctionDeclaration struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// geminiFunctionDecls maps declarations. Returns nil for none so the caller can
// leave the tools array off entirely.
func geminiFunctionDecls(defs []ToolDef) []geminiFunctionDeclaration {
	if len(defs) == 0 {
		return nil
	}
	out := make([]geminiFunctionDeclaration, 0, len(defs))
	for _, d := range defs {
		out = append(out, geminiFunctionDeclaration{
			Name: d.Name, Description: d.Description, Parameters: d.Schema,
		})
	}
	return out
}

// geminiToolCalls pulls function calls out of a reply and SYNTHESIZES an ID for
// each, because Gemini does not issue any. The format is deliberately
// unspecified and nothing may parse it — geminiToolResponses resolves purely by
// equality against the call list.
func geminiToolCalls(parts []geminiPart) []ToolCall {
	var out []ToolCall
	for i, p := range parts {
		if p.FunctionCall == nil {
			continue
		}
		args, err := json.Marshal(p.FunctionCall.Args)
		if err != nil {
			args = []byte("{}")
		}
		out = append(out, ToolCall{
			ID:    fmt.Sprintf("gemini-%d-%s", i, p.FunctionCall.Name),
			Name:  p.FunctionCall.Name,
			Input: args,
		})
	}
	return out
}

// geminiToolResponses renders results as functionResponse parts, IN CALL ORDER.
// calls is the preceding assistant turn's ToolCalls: it is the only thing that
// knows a result's function name and position, since the ID carries neither.
// A result naming an unknown call is dropped rather than guessed at.
func geminiToolResponses(results []ToolResult, calls []ToolCall) []geminiPart {
	byID := make(map[string]ToolResult, len(results))
	for _, r := range results {
		byID[r.ToolCallID] = r
	}
	var out []geminiPart
	for _, c := range calls {
		r, ok := byID[c.ID]
		if !ok {
			continue
		}
		out = append(out, geminiPart{FunctionResponse: &geminiFunctionResponse{
			Name:     c.Name,
			Response: map[string]any{"content": r.Content, "isError": r.IsError},
		}})
	}
	return out
}

// geminiStopReason maps finishReason. Unknown or absent reads as a completed
// turn rather than inventing a failure.
func geminiStopReason(fr string) StopReason {
	if fr == "MAX_TOKENS" {
		return StopTruncated
	}
	return StopEndTurn
}
```

Add `"encoding/json"` and `"fmt"` to the imports if absent (`fmt` is already used by `send`).

- [ ] **Step 4: Extend `geminiPart`, `geminiTool` and `geminiResponse`**

On `geminiPart`, add:

```go
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
```

On the existing `geminiTool` (which today carries only `GoogleSearch`), add:

```go
	FunctionDeclarations []geminiFunctionDeclaration `json:"function_declarations,omitempty"`
```

On `geminiResponse`'s `Candidates` element, add `FinishReason string \`json:"finishReason"\`` beside `Content`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd llm && go test ./... -run TestGemini -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Wire into `complete`**

Where contents are built, an assistant turn's `m.ToolCalls` become `functionCall` parts, and a user turn's results become `geminiToolResponses(m.ToolResults, prevCalls)` where `prevCalls` is the `ToolCalls` of the most recent preceding assistant `Message`. Track it while looping:

```go
	var prevCalls []ToolCall
	for _, m := range req.Messages {
		// … existing part building …
		if m.Role == "assistant" {
			for _, tc := range m.ToolCalls {
				var args map[string]any
				_ = json.Unmarshal(tc.Input, &args)
				parts = append(parts, geminiPart{FunctionCall: &geminiFunctionCall{Name: tc.Name, Args: args}})
			}
			prevCalls = m.ToolCalls
		} else {
			parts = append(parts, geminiToolResponses(m.ToolResults, prevCalls)...)
		}
		// … existing append to body.Contents …
	}
	if decls := geminiFunctionDecls(req.Tools); len(decls) > 0 {
		body.Tools = append(body.Tools, geminiTool{FunctionDeclarations: decls})
	}
```

In `send`, populate `resp.ToolCalls = geminiToolCalls(out.Candidates[0].Content.Parts)` and `resp.StopReason = geminiStopReason(out.Candidates[0].FinishReason)` in the branch that already reads `Candidates[0]`.

- [ ] **Step 7: Verify everything passes**

Run: `cd .. && go build ./... && go vet ./... && go test ./...`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add llm/google.go llm/llm_test.go
git commit -m "Map tool calling onto the Gemini adapter"
```

---

### Task 4: Document and release

**Files:**
- Modify: `README.md`
- Modify: `docs/tool-calling.md` (status line only)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: nothing consumed by code.

- [ ] **Step 1: Add a tool-calling section to `README.md`**

Insert after the "Effort and structured outputs" section:

````markdown
### Tool calling

`Complete` runs **one** round. Declare tools, run whatever the model asks for,
append the round to the history, and call again — the loop is yours, so an agent
that must log or gate each call can do so.

```go
resp, err := c.Complete(ctx, llm.Request{
    Tools:    []llm.ToolDef{{Name: "get_weather", Description: "Current weather", Schema: schema}},
    Messages: msgs,
})

if resp.StopReason == llm.StopToolUse {
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
```

Every result from one round goes in **one** message. Splitting them across
messages is rejected by some providers and silently degrades parallel tool
calling on others.

`ToolCall.ID` is opaque — echo it back and never parse it. Check
`resp.StopReason == llm.StopTruncated`: a turn cut off at the token ceiling
otherwise looks exactly like a finished one.
````

- [ ] **Step 2: Flip the spec's status**

In `docs/tool-calling.md`, change `- **Status:** Approved, not yet implemented` to `- **Status:** Implemented in v0.3.0`.

- [ ] **Step 3: Verify the whole suite**

Run: `go build ./... && go vet ./... && go test ./...`
Expected: PASS.

- [ ] **Step 4: Commit and tag**

```bash
git add README.md docs/tool-calling.md
git commit -m "Document tool calling"
git tag -a v0.3.0 -m "v0.3.0 — tool calling across all three providers"
```

> **Do not push the tag without confirming.** The Go module proxy caches a
> version immutably; a mistake means v0.3.1, never a re-cut v0.3.0.

---

## Self-Review

**Spec coverage.** Types → Task 1. Nested-results rationale → Task 1 (`Message` comments) and Task 4 (README). Provider mapping table → Tasks 1–3, one column each. Google asymmetry → Task 3, with the out-of-order test that proves ID-based resolution. Out-of-scope items (no loop, no `ToolChoice`, no `strict`) → nothing implements them, as intended; the no-loop decision is documented in Task 4's README copy. Failure modes → `StopTruncated` mapped in all three adapters; `IsError` carried in all three; refusal untouched from v0.2.0. Testing section → Tasks 1–3. Compatibility → the global constraint that pre-existing tests pass untouched, verified at every task.

**Placeholders.** None: every code step contains the real code, and the two "verify against current docs" notes are scoped to specific struct tags in Task 2 with the consequence stated (tags change, logic does not).

**Type consistency.** `ToolDef`/`ToolCall`/`ToolResult`/`StopReason` field names are identical across all four tasks. `ToolCall.Input` is `json.RawMessage` everywhere. Task 3's `geminiToolResponses(results, calls)` argument order matches its call site in Step 6. `anthropicTools` is used only in Task 1; `oaiTools`/`geminiFunctionDecls` only in their own tasks.

**Known gap, deliberate.** Tasks 2 and 3 wire mappings into `complete` without a test that exercises `complete` end to end, because doing so needs a network call or an HTTP round-trip stub, and neither provider has one today. The pure mappings — where the real logic lives — are tested. Adding an `httptest` harness for both is worth its own task later, and is not in scope here.
