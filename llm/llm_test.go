package llm

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// fakeProvider records which models it was asked for and returns a canned result
// per model ("" key = default). An entry in fail makes that model error.
type fakeProvider struct {
	calls    []string
	fail     map[string]error
	failResp map[string]Response
}

func (f *fakeProvider) complete(_ context.Context, model string, _ int, _ Request) (Response, error) {
	f.calls = append(f.calls, model)
	if err, ok := f.fail[model]; ok {
		// failResp lets a test model a refusal's POPULATED Response (tokens billed
		// alongside the error); an absent entry yields the zero Response, as before.
		return f.failResp[model], err
	}
	return Response{Text: "ok:" + model}, nil
}

func newTestRouter(a, g *fakeProvider) *router {
	return &router{
		providers: map[Provider]provider{
			ProviderAnthropic: a,
			ProviderGoogle:    g,
		},
		profiles: map[Profile]ModelRef{
			ProfileFast:       {Provider: ProviderGoogle, Model: "flash-lite"},
			ProfileChat:       {Provider: ProviderGoogle, Model: "flash"},
			ProfileDeep:       {Provider: ProviderAnthropic, Model: "opus"},
			ProfileVision:     {Provider: ProviderGoogle, Model: "flash"},
			ProfileDocument:   {Provider: ProviderGoogle, Model: "flash"},
			ProfileTranscribe: {Provider: ProviderGoogle, Model: "flash"},
			ProfileSearch:     {Provider: ProviderAnthropic, Model: "haiku"},
		},
		fallbacks: map[Profile]ModelRef{
			ProfileChat: {Provider: ProviderAnthropic, Model: "haiku"},
		},
		maxTokens: 100,
	}
}

func TestResolveRoutesTaskToProfileModel(t *testing.T) {
	r := newTestRouter(&fakeProvider{}, &fakeProvider{})
	cases := []struct {
		task  Task
		model string
	}{
		{TaskChat, "flash"},
		{TaskNudge, "flash"}, // nudge rides the chat profile
		{TaskClassify, "flash-lite"},
		{TaskExtract, "flash-lite"},
		{TaskReason, "opus"},
		{TaskSearch, "haiku"},
		{TaskVision, "flash"},
		{TaskDocument, "flash"},   // documents have their own profile (PDF-native)
		{TaskTranscribe, "flash"}, // speech-to-text has its own profile
	}
	for _, c := range cases {
		ref, _ := r.resolve(Request{Task: c.task})
		if ref.Model != c.model {
			t.Errorf("task %s: got model %q, want %q", c.task, ref.Model, c.model)
		}
	}
}

func TestResolveLegacyTier(t *testing.T) {
	r := newTestRouter(&fakeProvider{}, &fakeProvider{})
	if ref, _ := r.resolve(Request{}); ref.Model != "flash-lite" {
		t.Errorf("default tier (cheap) should hit fast, got %q", ref.Model)
	}
	if ref, _ := r.resolve(Request{Tier: TierPremium}); ref.Model != "opus" {
		t.Errorf("premium tier should hit deep, got %q", ref.Model)
	}
}

func TestResolveDegradesWhenProviderUnconfigured(t *testing.T) {
	// Only Anthropic configured; chat profile points at Google → resolve must degrade
	// to a configured profile instead of failing.
	a := &fakeProvider{}
	r := newTestRouter(a, &fakeProvider{})
	delete(r.providers, ProviderGoogle)
	ref, _ := r.resolve(Request{Task: TaskChat})
	if ref.Provider != ProviderAnthropic {
		t.Fatalf("expected degradation to a configured provider, got %+v", ref)
	}
}

func TestCompleteFailsOverToConfiguredFallback(t *testing.T) {
	g := &fakeProvider{fail: map[string]error{"flash": errors.New("gemini: denied (status 403)")}}
	a := &fakeProvider{}
	r := newTestRouter(a, g)
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat})
	if err != nil {
		t.Fatalf("expected fallback to answer, got error: %v", err)
	}
	// The response must be stamped with the model that ACTUALLY answered — the
	// usage ledger attributes failover days from this.
	if resp.Model != "haiku" || resp.Provider != ProviderAnthropic {
		t.Fatalf("expected haiku/anthropic stamp, got %s/%s", resp.Model, resp.Provider)
	}
	if len(a.calls) != 1 || a.calls[0] != "haiku" {
		t.Fatalf("expected exactly one fallback call to haiku, got %v", a.calls)
	}
}

func TestCompleteExplicitModelNeverFailsOver(t *testing.T) {
	// The admin Compare tool measures THAT model — a silent substitution would lie.
	g := &fakeProvider{fail: map[string]error{"flash": errors.New("gemini: overloaded (status 503)")}}
	a := &fakeProvider{}
	r := newTestRouter(a, g)
	_, err := r.Complete(context.Background(), Request{Model: &ModelRef{Provider: ProviderGoogle, Model: "flash"}})
	if err == nil {
		t.Fatal("expected the explicit-model error to surface")
	}
	if len(a.calls) != 0 {
		t.Fatalf("no fallback may fire for explicit models, but got calls %v", a.calls)
	}
}

func TestCompleteNoFailoverOnContentError(t *testing.T) {
	// A content-shaped error (not throttle/auth/network) would likely fail anywhere —
	// surface it instead of burning a second call.
	g := &fakeProvider{fail: map[string]error{"flash": errors.New("gemini: request contains an invalid argument")}}
	a := &fakeProvider{}
	r := newTestRouter(a, g)
	_, err := r.Complete(context.Background(), Request{Task: TaskChat})
	if err == nil {
		t.Fatal("expected the content error to surface")
	}
	if len(a.calls) != 0 {
		t.Fatalf("content errors must not fail over, but got calls %v", a.calls)
	}
}

func TestRefusalFailsOverToConfiguredFallback(t *testing.T) {
	// A safety refusal is a 200, not a transport failure — but classifiers differ per
	// model, so the profile's fallback is exactly the model that may still answer.
	g := &fakeProvider{fail: map[string]error{
		"flash": &RefusalError{Provider: ProviderGoogle, Model: "flash", Category: "cyber"},
	}}
	a := &fakeProvider{}
	r := newTestRouter(a, g)
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat})
	if err != nil {
		t.Fatalf("expected fallback to answer a refusal, got error: %v", err)
	}
	if resp.Model != "haiku" {
		t.Fatalf("expected the fallback model to answer, got %q", resp.Model)
	}
}

func TestRefusalResponseCarriesModelStamp(t *testing.T) {
	// A refusal returns a POPULATED Response (billed tokens) alongside its error. When
	// NoFallback makes it final, the router must still stamp Provider/Model so a cost
	// ledger can attribute the spend — otherwise the tokens land on no model.
	g := &fakeProvider{
		fail:     map[string]error{"flash": &RefusalError{Provider: ProviderGoogle, Model: "flash", Category: "cyber"}},
		failResp: map[string]Response{"flash": {InputTokens: 100, OutputTokens: 5}},
	}
	r := newTestRouter(&fakeProvider{}, g)
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat, NoFallback: true})
	if !errors.Is(err, ErrRefused) {
		t.Fatalf("expected ErrRefused, got %v", err)
	}
	if resp.Provider != ProviderGoogle || resp.Model != "flash" {
		t.Fatalf("refusal Response must be stamped for cost attribution, got provider=%q model=%q", resp.Provider, resp.Model)
	}
	if resp.InputTokens != 100 || resp.OutputTokens != 5 {
		t.Fatalf("billed tokens must ride along the refusal, got in=%d out=%d", resp.InputTokens, resp.OutputTokens)
	}
}

func TestGeminiRefusalCategory(t *testing.T) {
	// Drives the REAL google response parser: a safety decline is a 200, so detection
	// keys on finishReason (candidate-level) or promptFeedback.blockReason (prompt-level).
	cases := []struct{ name, body, want string }{
		{"candidate safety block", `{"candidates":[{"finishReason":"SAFETY","content":{"parts":[]}}]}`, "SAFETY"},
		{"candidate prohibited content", `{"candidates":[{"finishReason":"PROHIBITED_CONTENT","content":{"parts":[]}}]}`, "PROHIBITED_CONTENT"},
		{"prompt-level block, no candidates", `{"candidates":[],"promptFeedback":{"blockReason":"BLOCKLIST"}}`, "BLOCKLIST"},
		{"normal turn", `{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"hi"}]}}]}`, ""},
		{"recitation is not a refusal", `{"candidates":[{"finishReason":"RECITATION","content":{"parts":[]}}]}`, ""},
		{"truncation is not a refusal", `{"candidates":[{"finishReason":"MAX_TOKENS","content":{"parts":[{"text":"par"}]}}]}`, ""},
	}
	for _, c := range cases {
		var out geminiResponse
		if err := json.Unmarshal([]byte(c.body), &out); err != nil {
			t.Fatalf("%s: unmarshal: %v", c.name, err)
		}
		if got := geminiRefusalCategory(out); got != c.want {
			t.Errorf("%s: geminiRefusalCategory = %q, want %q", c.name, got, c.want)
		}
	}
}

func TestOpenAIRefusal(t *testing.T) {
	// Drives the REAL openai refusal detector: a content_filter finish_reason, or the
	// structured refusal field. DeepSeek/Moonshot emit neither, so they never trigger.
	cases := []struct{ name, finishReason, refusal, wantCat, wantExpl string }{
		{"content filter", "content_filter", "", "content_filter", ""},
		{"content filter with detail", "content_filter", "blocked", "content_filter", "blocked"},
		{"structured refusal", "stop", "I can't help with that.", "refusal", "I can't help with that."},
		{"normal completion", "stop", "", "", ""},
		{"tool call is not a refusal", "tool_calls", "", "", ""},
	}
	for _, c := range cases {
		cat, expl := oaiRefusal(c.finishReason, c.refusal)
		if cat != c.wantCat || expl != c.wantExpl {
			t.Errorf("%s: oaiRefusal = (%q,%q), want (%q,%q)", c.name, cat, expl, c.wantCat, c.wantExpl)
		}
	}
}

func TestRefusalSurfacesAsErrorNotEmptyString(t *testing.T) {
	// The whole point of ErrRefused: without it a refusal returns ("", nil) and the
	// caller silently ships an empty answer.
	refusal := &RefusalError{Provider: ProviderGoogle, Model: "flash", Category: "cyber", Explanation: "declined"}
	g := &fakeProvider{fail: map[string]error{"flash": refusal}}
	a := &fakeProvider{}
	r := newTestRouter(a, g)
	// NoFallback so the refusal is the final word rather than being rescued.
	_, err := r.Complete(context.Background(), Request{Task: TaskChat, NoFallback: true})
	if err == nil {
		t.Fatal("a refusal must surface as an error, never as an empty response")
	}
	if !errors.Is(err, ErrRefused) {
		t.Fatalf("expected errors.Is(err, ErrRefused), got %v", err)
	}
	var re *RefusalError
	if !errors.As(err, &re) || re.Category != "cyber" {
		t.Fatalf("expected the cyber category to survive unwrapping, got %v", err)
	}
	if len(a.calls) != 0 {
		t.Fatalf("NoFallback must suppress the retry, but got calls %v", a.calls)
	}
}

func TestOutputConfigOmittedWhenUnset(t *testing.T) {
	// Sending an empty output_config would override the model's own defaults; the
	// request must carry the field only when the caller asked for something.
	if _, ok := outputConfig(Request{}); ok {
		t.Fatal("expected no output_config when neither effort nor schema is set")
	}
	oc, ok := outputConfig(Request{Effort: EffortXHigh})
	if !ok || string(oc.Effort) != "xhigh" {
		t.Fatalf("expected effort xhigh to map through, got %q (ok=%v)", oc.Effort, ok)
	}
	oc, ok = outputConfig(Request{JSONSchema: map[string]any{"type": "object"}})
	if !ok || oc.Format.Schema == nil {
		t.Fatal("expected a JSON schema to map onto output_config.format")
	}
}

func TestShouldFailover(t *testing.T) {
	yes := []string{
		"gemini: quota exceeded (status 429)",
		"anthropic: overloaded (status 503)",
		"gemini: Your project has been denied access. Please contact support. (status 403)",
		"unauthorized (status 401)",
		"models/gemini-2.5-flash is no longer available to new users (status 404)",
		"Post \"https://x\": connection refused",
		"context deadline exceeded",
		"bad gateway (status 502)",
	}
	no := []string{
		"gemini: request contains an invalid argument",
		"prompt blocked by safety settings",
	}
	for _, s := range yes {
		if !shouldFailover(errors.New(s)) {
			t.Errorf("expected failover for %q", s)
		}
	}
	for _, s := range no {
		if shouldFailover(errors.New(s)) {
			t.Errorf("did NOT expect failover for %q", s)
		}
	}
	if shouldFailover(nil) {
		t.Error("nil error must not fail over")
	}
}

func TestPriceBookResolutionOrder(t *testing.T) {
	fetched := map[string]Rate{"m": {Input: 1, Output: 2, CachedRead: 0.1}}
	overrides := map[string]Rate{"m": {Input: 10, Output: 20, CachedRead: 1}}
	b := NewPriceBook(fetched, overrides)
	if got := b.Rate("m").Input; got != 10 {
		t.Errorf("override should win, got input rate %v", got)
	}
	b2 := NewPriceBook(fetched, nil)
	if got := b2.Rate("m").Input; got != 1 {
		t.Errorf("fetched should win over defaults, got %v", got)
	}
	if got := NewPriceBook(nil, nil).Rate("no-such-model"); got != (Rate{}) {
		t.Errorf("unknown model should resolve to the zero rate, got %+v", got)
	}
}

func TestRateCostMath(t *testing.T) {
	// Rates are USD per MILLION tokens.
	r := Rate{Input: 1, Output: 5, CachedRead: 0.1}
	got := r.Cost(1_000_000, 2_000_000, 500_000)
	want := 1.0 + 10.0 + 0.05
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("cost = %v, want %v", got, want)
	}
	if got := (Rate{}).Cost(1000, 1000, 1000); got != 0 {
		t.Errorf("zero rate must cost 0, got %v", got)
	}
}

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
	got, err := geminiToolResponses([]ToolResult{
		{ToolCallID: calls[1].ID, Content: "b done"},
		{ToolCallID: calls[0].ID, Content: "a done"},
	}, calls)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
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

	// Supplying only the SECOND call's result must not silently land at index 0
	// and get attributed to the FIRST call — that is the exact misattribution
	// this design exists to prevent. It must error instead.
	_, err = geminiToolResponses([]ToolResult{
		{ToolCallID: calls[1].ID, Content: "b done"},
	}, calls)
	if err == nil {
		t.Fatal("expected an error for a call with no matching result, got nil")
	}

	// A result naming an ID that isn't in calls at all must also error, not be
	// dropped silently.
	_, err = geminiToolResponses([]ToolResult{
		{ToolCallID: calls[0].ID, Content: "a done"},
		{ToolCallID: calls[1].ID, Content: "b done"},
		{ToolCallID: "gemini-unknown-call", Content: "mystery"},
	}, calls)
	if err == nil {
		t.Fatal("expected an error for a result referencing an unknown call, got nil")
	}
}

func TestGeminiStopReasonMapping(t *testing.T) {
	cases := []struct {
		fr       string
		hasCalls bool
		want     StopReason
	}{
		{"MAX_TOKENS", false, StopTruncated},
		{"STOP", false, StopEndTurn},
		{"", false, StopEndTurn},
		// Gemini says only "STOP" on a functionCall turn; the calls are the signal.
		{"STOP", true, StopToolUse},
		{"", true, StopToolUse},
		// Truncation WINS over the tool-use override. Anthropic reports
		// StopTruncated for max_tokens-with-tool_use, and reporting StopToolUse
		// here would hide the cut-off behind a round the caller then runs.
		{"MAX_TOKENS", true, StopTruncated},
	}
	for _, c := range cases {
		if got := geminiStopReason(c.fr, c.hasCalls); got != c.want {
			t.Fatalf("(%q, calls=%v) → %s, want %s", c.fr, c.hasCalls, got, c.want)
		}
	}
}

func TestNormalizeToolInput(t *testing.T) {
	// ToolCall.Input is documented as always being a valid JSON object. An EMPTY
	// json.RawMessage is the dangerous one: it makes json.Marshal of the whole
	// request fail, so a single such call poisons every later turn.
	empty := json.RawMessage("{}")
	cases := []struct {
		name string
		in   json.RawMessage
		want json.RawMessage
	}{
		{"nil", nil, empty},
		{"empty", json.RawMessage(""), empty},
		{"blank", json.RawMessage("   "), empty},
		{"null", json.RawMessage("null"), empty},
		{"truncated", json.RawMessage(`{"host":`), empty},
		{"object", json.RawMessage(`{"host":"a.example"}`), json.RawMessage(`{"host":"a.example"}`)},
	}
	for _, c := range cases {
		got := normalizeToolInput(c.in)
		if string(got) != string(c.want) {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
		// Whatever comes out must survive being marshalled as part of a request.
		if _, err := json.Marshal(struct {
			In json.RawMessage `json:"in"`
		}{got}); err != nil {
			t.Errorf("%s: normalized input must always marshal, got %v", c.name, err)
		}
	}
}

func TestToolCallInputNeverEmptyPerProvider(t *testing.T) {
	// An OpenAI-compatible backend answering a no-arg call with `"arguments": ""`.
	calls := oaiToolCalls([]oaiToolCall{
		{ID: "call_1", Type: "function", Function: oaiToolCallFunc{Name: "ping", Arguments: ""}},
	})
	if len(calls) != 1 || string(calls[0].Input) != "{}" {
		t.Fatalf(`openai: empty arguments must normalize to "{}", got %#v`, calls)
	}
	// Gemini omits args entirely for a no-arg call, so Args is a nil map, which
	// marshals to `null` rather than an object.
	gcalls := geminiToolCalls([]geminiPart{{FunctionCall: &geminiFunctionCall{Name: "ping"}}})
	if len(gcalls) != 1 || string(gcalls[0].Input) != "{}" {
		t.Fatalf(`gemini: absent args must normalize to "{}", got %#v`, gcalls)
	}
	// And the outbound path: an assistant turn echoed back with a nil Input must
	// not serialise as `null` or "".
	msgs := oaiMessages(Request{Messages: []Message{
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_1", Name: "ping"}}},
	}})
	if len(msgs) != 1 || len(msgs[0].ToolCalls) != 1 {
		t.Fatalf("expected one assistant message carrying one call, got %#v", msgs)
	}
	if got := msgs[0].ToolCalls[0].Function.Arguments; got != "{}" {
		t.Fatalf(`openai outbound: nil Input must serialise as "{}", got %q`, got)
	}
}

func TestOpenAIToolResultsFollowTheAssistantTurnDirectly(t *testing.T) {
	// OpenAI requires each "tool" message to IMMEDIATELY follow the assistant turn
	// whose tool_calls it answers. The documented shape for answering a round is a
	// user message with ONLY ToolResults — if that empty user turn is emitted, it
	// sits between the two and every second-round call fails.
	msgs := oaiMessages(Request{Messages: []Message{
		{Role: "user", Content: "scan it"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_1", Name: "dispatch_scan", Input: json.RawMessage(`{}`)}}},
		{Role: "user", ToolResults: []ToolResult{{ToolCallID: "call_1", Content: "done"}}},
	}})
	var roles []string
	for _, m := range msgs {
		roles = append(roles, m.Role)
	}
	want := []string{"user", "assistant", "tool"}
	if len(roles) != len(want) {
		t.Fatalf("got roles %v, want %v (an empty user turn must not be emitted)", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("got roles %v, want %v", roles, want)
		}
	}
	// An assistant turn carrying only tool calls (no prose) must still be emitted —
	// suppressing it would orphan the tool message it anchors.
	if len(msgs[1].ToolCalls) != 1 {
		t.Fatalf("the assistant turn must survive with its tool_calls: %#v", msgs[1])
	}
}

func TestOpenAIWireUnchangedWithoutTools(t *testing.T) {
	// The empty-message suppression must not touch a request that uses no tools:
	// this branch is additive-only for the existing consumers. A content-less
	// message with no results is still emitted verbatim.
	msgs := oaiMessages(Request{
		SystemCacheable: "sys",
		Messages: []Message{
			{Role: "user", Content: "halo"},
			{Role: "assistant", Content: "hai"},
			{Role: "user", Content: ""},
		},
	})
	got, err := json.Marshal(msgs)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `[{"role":"system","content":"sys"},{"role":"user","content":"halo"},` +
		`{"role":"assistant","content":"hai"},{"role":"user","content":""}]`
	if string(got) != want {
		t.Fatalf("no-tools wire changed:\n got %s\nwant %s", got, want)
	}
}

func TestOpenAIToolResultErrorIsVisibleToTheModel(t *testing.T) {
	// OpenAI's "tool" message has no is_error field. The flag must survive as text
	// rather than be dropped — the contract is that the model SEES the failure.
	if got := oaiToolResultContent(ToolResult{Content: "boom", IsError: true}); got != "Error: boom" {
		t.Fatalf("a failed result must be marked in the content, got %q", got)
	}
	if got := oaiToolResultContent(ToolResult{Content: "fine"}); got != "fine" {
		t.Fatalf("a successful result must pass through untouched, got %q", got)
	}
}

func TestGeminiCallIDsAreUniqueAcrossRounds(t *testing.T) {
	// Two ROUNDS calling the same tool. An ID derived from the part index and the
	// function name repeats here, and on a cross-provider failover the history then
	// carries two distinct calls sharing one ID.
	part := []geminiPart{{FunctionCall: &geminiFunctionCall{Name: "dispatch_scan"}}}
	first := geminiToolCalls(part)
	second := geminiToolCalls(part)
	if len(first) != 1 || len(second) != 1 {
		t.Fatalf("expected one call per round, got %d/%d", len(first), len(second))
	}
	if first[0].ID == second[0].ID {
		t.Fatalf("IDs must be unique across rounds, both rounds got %q", first[0].ID)
	}
	// Opaque: nothing that invites parsing back out of it.
	if strings.Contains(first[0].ID, "dispatch_scan") {
		t.Fatalf("the ID must not embed the function name: %q", first[0].ID)
	}
}

func TestGeminiPrevCallsClearedAfterUserTurnConsumesThem(t *testing.T) {
	call := ToolCall{ID: "gemini-x", Name: "dispatch_scan", Input: json.RawMessage(`{}`)}
	answered := []Message{
		{Role: "user", Content: "scan it"},
		{Role: "assistant", ToolCalls: []ToolCall{call}},
		{Role: "user", ToolResults: []ToolResult{{ToolCallID: call.ID, Content: "done"}}},
	}

	// (a) A caller must be able to interject plain text after an answered round.
	// With the calls left set, this third turn is validated against a round that
	// already has its results and fails with a spurious "missing tool result".
	contents, err := geminiContents(Request{Messages: append(append([]Message{}, answered...),
		Message{Role: "user", Content: "actually, stop"})})
	if err != nil {
		t.Fatalf("plain text after an answered round must be allowed, got: %v", err)
	}
	if len(contents) != 4 {
		t.Fatalf("expected four turns, got %d", len(contents))
	}
	if len(contents[3].Parts) != 1 || contents[3].Parts[0].Text != "actually, stop" {
		t.Fatalf("the interjected turn must carry only its text: %#v", contents[3].Parts)
	}

	// (b) The same round appended twice must NOT quietly answer the same call
	// again — that is the misattribution the guard exists to prevent.
	_, err = geminiContents(Request{Messages: append(append([]Message{}, answered...),
		Message{Role: "user", ToolResults: []ToolResult{{ToolCallID: call.ID, Content: "done"}}})})
	if err == nil {
		t.Fatal("a round answered twice must error, not emit two answering turns")
	}
	if !errors.Is(err, ErrToolResultMismatch) {
		t.Fatalf("expected errors.Is(err, ErrToolResultMismatch), got %v", err)
	}
}

func TestGeminiToolResultMismatchIsMatchable(t *testing.T) {
	// Callers must be able to errors.Is this rather than match on the prose.
	calls := geminiToolCalls([]geminiPart{{FunctionCall: &geminiFunctionCall{Name: "dispatch_scan"}}})
	_, err := geminiToolResponses([]ToolResult{{ToolCallID: "no-such-call"}}, calls)
	if !errors.Is(err, ErrToolResultMismatch) {
		t.Fatalf("unknown call: expected ErrToolResultMismatch, got %v", err)
	}
	_, err = geminiToolResponses(nil, calls)
	if !errors.Is(err, ErrToolResultMismatch) {
		t.Fatalf("missing result: expected ErrToolResultMismatch, got %v", err)
	}
}

func TestAnthropicInputSchemaForwardsTheWholeSchema(t *testing.T) {
	// OpenAI and Gemini pass ToolDef.Schema wholesale. Lifting only
	// properties/required here would give Claude a schema with its $defs stripped
	// and its $refs dangling — one tool, two behaviours, no error anywhere.
	schema := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"target": map[string]any{"$ref": "#/$defs/target"}},
		"required":             []string{"target"},
		"additionalProperties": false,
		"$defs":                map[string]any{"target": map[string]any{"type": "string"}},
	}
	got := anthropicInputSchema(schema)
	if got.ExtraFields["additionalProperties"] != false {
		t.Fatalf("additionalProperties must be forwarded, got %#v", got.ExtraFields)
	}
	if got.ExtraFields["$defs"] == nil {
		t.Fatalf("$defs must be forwarded, got %#v", got.ExtraFields)
	}
	// `type` is the SDK's own constant — forwarding it too would duplicate the key.
	if _, dup := got.ExtraFields["type"]; dup {
		t.Fatalf("type must not be forwarded; the SDK sets it: %#v", got.ExtraFields)
	}
	if _, dup := got.ExtraFields["properties"]; dup {
		t.Fatalf("properties has a typed field and must not be duplicated: %#v", got.ExtraFields)
	}
	// It has to actually reach the wire, not just sit in the struct.
	raw, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back map[string]any
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, k := range []string{"type", "properties", "required", "additionalProperties", "$defs"} {
		if _, ok := back[k]; !ok {
			t.Fatalf("key %q missing from the serialised schema: %s", k, raw)
		}
	}
	if n := len(back); n != 5 {
		t.Fatalf("expected exactly 5 keys, got %d: %s", n, raw)
	}
}
