package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// Parts in the shape Gemini returned live (2026-11-03, gemini-3.8-flash, Google
// Search + one function tool, includeServerSideToolInvocations on). futureField
// stands for a field aikit does not know; it must survive the round trip.
const (
	gsSearchCall = `{"thoughtSignature":"c2lnLWNhbGw=","toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"queries":["world cup winner"]},"id":"call_1"},"futureField":7}`
	gsFuncCall   = `{"functionCall":{"name":"lookup_ticket","args":{"id":"A1"},"id":"call_2"}}`
	gsSearchResp = `{"thoughtSignature":"c2lnLXJlc3A=","toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"search_suggestions":"<div/>"},"id":"call_1"}}`
)

var gsTurnParts = []string{gsSearchCall, gsFuncCall, gsSearchResp}

func gsReply(parts []string, finish string) string {
	return `{"candidates":[{"content":{"role":"model","parts":[` + strings.Join(parts, ",") +
		`]},"finishReason":"` + finish + `"}],"usageMetadata":{"promptTokenCount":71,"candidatesTokenCount":37}}`
}

func gsEvent(part, finish string) string {
	c := `{"content":{"role":"model","parts":[` + part + `]}`
	if finish != "" {
		c += `,"finishReason":"` + finish + `"`
	}
	return `{"candidates":[` + c + `}]}`
}

// gsFake is a Gemini endpoint that plays one scripted round per request: a JSON
// reply for :generateContent, SSE events for :streamGenerateContent. It keeps
// every raw request body.
type gsFake struct {
	mu     sync.Mutex
	rounds [][]string // per request: one JSON reply, or the stream's events
	bodies [][]byte
}

func (f *gsFake) start(t *testing.T) *googleProvider {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		f.mu.Lock()
		f.bodies = append(f.bodies, raw)
		i := len(f.bodies) - 1
		f.mu.Unlock()
		if i >= len(f.rounds) {
			t.Errorf("unexpected request %d", i+1)
			w.WriteHeader(500)
			return
		}
		if strings.Contains(r.URL.Path, ":streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, e := range f.rounds[i] {
				_, _ = io.WriteString(w, "data: "+e+"\n\n")
			}
			return
		}
		_, _ = io.WriteString(w, f.rounds[i][0])
	}))
	t.Cleanup(srv.Close)
	return &googleProvider{apiKey: "k", baseURL: srv.URL, http: newProviderHTTPClient(nil)}
}

func (f *gsFake) body(t *testing.T, i int) map[string]json.RawMessage {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(f.bodies[i], &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func sameJSON(a json.RawMessage, b string) bool {
	var x, y any
	return json.Unmarshal(a, &x) == nil && json.Unmarshal([]byte(b), &y) == nil && reflect.DeepEqual(x, y)
}

var gsTool = ToolDef{Name: "lookup_ticket", Description: "Look up a ticket",
	Schema: map[string]any{"type": "object", "properties": map[string]any{"id": map[string]any{"type": "string"}}}}

func gsTicketTool(context.Context, ToolCall) (string, error) {
	return `{"status":"open"}`, nil
}

func TestGeminiToolConfigOnlyWithSearchAndTools(t *testing.T) {
	for _, c := range []struct {
		name   string
		search bool
		tools  []ToolDef
		want   bool
	}{
		{"search and tools", true, []ToolDef{gsTool}, true},
		{"search only", true, nil, false},
		{"tools only", false, []ToolDef{gsTool}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			body, err := geminiBody("gemini-3.8-flash", 100, Request{Messages: userMsg("hi"), WebSearch: c.search, Tools: c.tools})
			if err != nil {
				t.Fatal(err)
			}
			raw, _ := json.Marshal(body)
			var m map[string]json.RawMessage
			_ = json.Unmarshal(raw, &m)
			tc, has := m["toolConfig"]
			if has != c.want {
				t.Fatalf("toolConfig present = %v, want %v (body %s)", has, c.want, raw)
			}
			if c.want && string(tc) != `{"includeServerSideToolInvocations":true}` {
				t.Errorf("toolConfig = %s", tc)
			}
		})
	}
}

// checkSearchTurn checks a reply to the recorded turn: one ToolCall (the function),
// no search text, and ProviderData holding the turn.
func checkSearchTurn(t *testing.T, resp Response) {
	t.Helper()
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "lookup_ticket" || string(resp.ToolCalls[0].Input) != `{"id":"A1"}` {
		t.Errorf("ToolCalls = %+v, want only lookup_ticket {\"id\":\"A1\"}", resp.ToolCalls)
	}
	if strings.Contains(resp.Text, "search_suggestions") || strings.Contains(resp.Text, "GOOGLE_SEARCH") {
		t.Errorf("Text holds a search step: %q", resp.Text)
	}
	var d struct {
		Provider string            `json:"provider"`
		Parts    []json.RawMessage `json:"parts"`
	}
	if err := json.Unmarshal(resp.ProviderData, &d); err != nil {
		t.Fatalf("ProviderData %q: %v", resp.ProviderData, err)
	}
	if d.Provider != "google" || len(d.Parts) < 3 {
		t.Fatalf("ProviderData = %s, want the google turn", resp.ProviderData)
	}
	for i, want := range gsTurnParts {
		if string(d.Parts[i]) != want {
			t.Errorf("ProviderData part %d = %s, want %s", i, d.Parts[i], want)
		}
	}
}

// checkEchoedTurn checks that round 2 sent the model turn verbatim, then the
// function's result.
func checkEchoedTurn(t *testing.T, f *gsFake, wantParts []string) {
	t.Helper()
	var contents []struct {
		Role  string            `json:"role"`
		Parts []json.RawMessage `json:"parts"`
	}
	if err := json.Unmarshal(f.body(t, 1)["contents"], &contents); err != nil {
		t.Fatal(err)
	}
	if len(contents) != 3 || contents[1].Role != "model" {
		t.Fatalf("round 2 contents = %+v, want user, model, user", contents)
	}
	if len(contents[1].Parts) != len(wantParts) {
		t.Fatalf("model turn has %d parts, want %d", len(contents[1].Parts), len(wantParts))
	}
	// The request body is HTML-escaped as a whole, so compare JSON values.
	for i, want := range wantParts {
		if !sameJSON(contents[1].Parts[i], want) {
			t.Errorf("model part %d = %s, want %s", i, contents[1].Parts[i], want)
		}
	}
	if !strings.Contains(string(contents[2].Parts[0]), `"functionResponse":{"name":"lookup_ticket"`) {
		t.Errorf("result turn = %s, want the lookup_ticket functionResponse", contents[2].Parts[0])
	}
	if !strings.Contains(string(f.body(t, 1)["toolConfig"]), "includeServerSideToolInvocations") {
		t.Error("round 2 lost toolConfig")
	}
}

func TestGeminiSearchToolLoopEchoesTurn(t *testing.T) {
	f := &gsFake{rounds: [][]string{
		{gsReply(gsTurnParts, "STOP")},
		{gsReply([]string{`{"text":"Spain won; ticket A1 is open."}`}, "STOP")},
	}}
	r := &router{
		providers: map[Provider]provider{ProviderGoogle: f.start(t)},
		profiles:  map[Profile]ModelRef{ProfileChat: {Provider: ProviderGoogle, Model: "gemini-3.1-pro"}},
		maxTokens: 100,
	}
	run, err := RunTools(context.Background(), r,
		Request{Messages: userMsg("search, then look up A1"), WebSearch: true, Tools: []ToolDef{gsTool}},
		map[string]ToolFunc{"lookup_ticket": gsTicketTool}, ToolLoopOptions{})
	if err != nil {
		t.Fatal(err)
	}
	checkSearchTurn(t, run.Rounds[0])
	checkEchoedTurn(t, f, gsTurnParts)
	if run.Response.Text != "Spain won; ticket A1 is open." {
		t.Errorf("final Text = %q", run.Response.Text)
	}
}

func TestGeminiSearchToolLoopStreamed(t *testing.T) {
	f := &gsFake{rounds: [][]string{
		{gsEvent(gsSearchCall, ""), gsEvent(gsFuncCall, ""), gsEvent(gsSearchResp, ""),
			gsEvent(`{"text":"Hel"}`, ""), gsEvent(`{"text":"lo"}`, ""), gsEvent(`{"text":"","thoughtSignature":"ZW5k"}`, "STOP")},
		{gsEvent(`{"text":"done"}`, "STOP")},
	}}
	r := &router{
		providers: map[Provider]provider{ProviderGoogle: f.start(t)},
		profiles:  map[Profile]ModelRef{ProfileChat: {Provider: ProviderGoogle, Model: "gemini-3.1-pro"}},
		maxTokens: 100,
	}
	var events []StreamEvent
	run, err := RunTools(context.Background(), r,
		Request{Messages: userMsg("search, then look up A1"), WebSearch: true, Tools: []ToolDef{gsTool},
			ToolEvents: ToolEventsStart, OnEvent: func(e StreamEvent) { events = append(events, e) }},
		map[string]ToolFunc{"lookup_ticket": gsTicketTool}, ToolLoopOptions{})
	if err != nil {
		t.Fatal(err)
	}
	first := run.Rounds[0]
	checkSearchTurn(t, first)
	if first.Text != "Hello" {
		t.Errorf("Text = %q, want Hello", first.Text)
	}
	var starts, texts []string
	for _, e := range events {
		switch e.Kind {
		case StreamToolStart:
			starts = append(starts, e.ToolName)
		case StreamText:
			texts = append(texts, e.Text)
		}
	}
	if len(starts) != 1 || starts[0] != "lookup_ticket" {
		t.Errorf("tool_start events = %v, want only lookup_ticket", starts)
	}
	if strings.Join(texts, "|") != "Hel|lo|done" {
		t.Errorf("text events = %v, want Hel lo done", texts)
	}
	// The streamed text pieces go back as one part.
	// The streamed text pieces go back as one part; the signed closing part stays apart.
	checkEchoedTurn(t, f, append(append([]string(nil), gsTurnParts...), `{"text":"Hello"}`, `{"text":"","thoughtSignature":"ZW5k"}`))
}

func TestGeminiRebuiltTurnGetsSkipSignature(t *testing.T) {
	call := ToolCall{ID: "c1", Name: "lookup_ticket", Input: json.RawMessage(`{"id":"A1"}`)}
	other, _ := json.Marshal(geminiProviderData{Provider: ProviderGoogle, Parts: []geminiPart{
		{FunctionCall: &geminiFunctionCall{Name: "another_tool"}, ThoughtSignature: "real"},
	}})
	for _, c := range []struct {
		name string
		data []byte
	}{
		{"no ProviderData", nil},
		{"calls do not match", other},
		{"another provider's data", []byte(`{"provider":"anthropic","parts":[{"text":"x"}]}`)},
		{"not JSON", []byte("garbage")},
	} {
		t.Run(c.name, func(t *testing.T) {
			contents, err := geminiContents(Request{Messages: []Message{
				{Role: "user", Content: "look up A1"},
				{Role: "assistant", Content: "checking", ToolCalls: []ToolCall{call, {ID: "c2", Name: "lookup_ticket"}}, ProviderData: c.data},
				{Role: "user", ToolResults: []ToolResult{{ToolCallID: "c1", Content: "open"}, {ToolCallID: "c2", Content: "closed"}}},
			}})
			if err != nil {
				t.Fatal(err)
			}
			raw, _ := json.Marshal(contents[1])
			want := `{"role":"model","parts":[{"text":"checking"},` +
				`{"functionCall":{"name":"lookup_ticket","args":{"id":"A1"}},"thoughtSignature":"skip_thought_signature_validator"},` +
				`{"functionCall":{"name":"lookup_ticket"}}]}`
			if string(raw) != want {
				t.Errorf("model turn = %s\nwant %s", raw, want)
			}
		})
	}
}

func TestGeminiNoProviderDataWithoutSignatures(t *testing.T) {
	f := &gsFake{rounds: [][]string{{gsReply([]string{`{"text":"hi"}`}, "STOP")}}}
	resp, err := f.start(t).complete(context.Background(), "gemini-3.1-pro", 100, Request{Messages: userMsg("hi")})
	if err != nil {
		t.Fatal(err)
	}
	if resp.ProviderData != nil {
		t.Errorf("ProviderData = %s, want nil", resp.ProviderData)
	}
}

// A plain function call (no search) carries its own signature; it must come back too.
func TestGeminiFunctionCallSignatureEchoed(t *testing.T) {
	call := `{"functionCall":{"name":"lookup_ticket","args":{"id":"A1"}},"thoughtSignature":"c2ln"}`
	f := &gsFake{rounds: [][]string{
		{gsReply([]string{call}, "STOP")},
		{gsReply([]string{`{"text":"open"}`}, "STOP")},
	}}
	r := &router{
		providers: map[Provider]provider{ProviderGoogle: f.start(t)},
		profiles:  map[Profile]ModelRef{ProfileChat: {Provider: ProviderGoogle, Model: "gemini-3.1-pro"}},
		maxTokens: 100,
	}
	if _, err := RunTools(context.Background(), r, Request{Messages: userMsg("look up A1"), Tools: []ToolDef{gsTool}},
		map[string]ToolFunc{"lookup_ticket": gsTicketTool}, ToolLoopOptions{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(f.body(t, 1)["contents"]), call) {
		t.Errorf("round 2 contents = %s, want the signed call verbatim", f.body(t, 1)["contents"])
	}
	if _, has := f.body(t, 0)["toolConfig"]; has {
		t.Error("toolConfig sent without WebSearch")
	}
}

// Other providers ignore a Gemini turn's ProviderData, so failover still works.
func TestProviderDataIgnoredByOpenAI(t *testing.T) {
	data, _ := json.Marshal(geminiProviderData{Provider: ProviderGoogle, Parts: []geminiPart{{Text: "x", ThoughtSignature: "s"}}})
	msgs := []Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello", ProviderData: data}}
	with, _ := json.Marshal(oaiMessages(Request{Messages: msgs}))
	msgs[1].ProviderData = nil
	without, _ := json.Marshal(oaiMessages(Request{Messages: msgs}))
	if string(with) != string(without) {
		t.Errorf("ProviderData changed the OpenAI body:\n%s\n%s", with, without)
	}
}
