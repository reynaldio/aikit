package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Grounding in the shape Gemini returned live (2026-11-05, gemini-3.8-flash).
const wsGrounding = `{"searchEntryPoint":{"renderedContent":"<div/>"},"webSearchQueries":["iceland population","","capital of peru"],` +
	`"groundingChunks":[{"web":{"uri":"https://r.test/a","title":"a.org"}},{"web":{"uri":"https://r.test/b","title":"b.org"}},{"web":{"uri":"https://r.test/unused","title":"c.org"}},{"maps":{"uri":"https://m.test"}}],` +
	`"groundingSupports":[{"segment":{"endIndex":9,"text":"model's own words"},"groundingChunkIndices":[1,0]},{"segment":{"text":"more"},"groundingChunkIndices":[0,3,99,-1]}]}`

var wsGeminiWant = []Citation{{URL: "https://r.test/b", Title: "b.org"}, {URL: "https://r.test/a", Title: "a.org"}}

func wsGeminiReply() string {
	return `{"candidates":[{"content":{"role":"model","parts":[{"text":"Lima."}]},"finishReason":"STOP","groundingMetadata":` + wsGrounding + `}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`
}

func TestGeminiWebSearchUsage(t *testing.T) {
	// Plain and streamed replies must give the same values.
	stream := []string{
		gsEvent(`{"text":"Li"}`, ""),
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"ma."}]},"finishReason":"STOP","groundingMetadata":` + wsGrounding + `}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`,
	}
	for name, tc := range map[string]struct {
		round []string
		req   Request
	}{
		"plain":  {[]string{wsGeminiReply()}, Request{Messages: userMsg("x"), WebSearch: true}},
		"stream": {stream, Request{Messages: userMsg("x"), WebSearch: true, OnEvent: func(StreamEvent) {}}},
	} {
		t.Run(name, func(t *testing.T) {
			p := (&gsFake{rounds: [][]string{tc.round}}).start(t)
			resp, err := p.complete(context.Background(), "gemini-3.1-pro", 100, tc.req)
			if err != nil {
				t.Fatal(err)
			}
			if resp.WebSearchRequests != 2 {
				t.Errorf("searches = %d, want 2 (the empty query is not counted)", resp.WebSearchRequests)
			}
			if !reflect.DeepEqual(resp.Citations, wsGeminiWant) {
				t.Errorf("citations = %+v, want %+v", resp.Citations, wsGeminiWant)
			}
		})
	}
}

// A turn that searches and then calls a function has no groundingMetadata: live
// on gemini-3.8-flash, the queries were only in the toolCall part.
func TestGeminiWebSearchCountedFromToolCallParts(t *testing.T) {
	search := `{"thoughtSignature":"cw==","toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"queries":["a","b"]},"id":"c1"}}`
	other := `{"toolCall":{"toolType":"URL_CONTEXT","args":{"queries":["not a search"]},"id":"c2"}}`
	p := (&gsFake{rounds: [][]string{
		{gsReply([]string{search, other, gsFuncCall}, "STOP")},
		{gsEvent(search, ""), gsEvent(other, ""), gsEvent(gsFuncCall, "STOP")},
	}}).start(t)
	req := Request{Messages: userMsg("x"), WebSearch: true, Tools: []ToolDef{gsTool}}
	for _, name := range []string{"plain", "stream"} {
		if name == "stream" {
			req.OnEvent = func(StreamEvent) {}
		}
		resp, err := p.complete(context.Background(), "gemini-3.1-pro", 100, req)
		if err != nil {
			t.Fatal(err)
		}
		if resp.WebSearchRequests != 2 || resp.Citations != nil {
			t.Errorf("%s: searches = %d citations = %+v, want 2 and nil", name, resp.WebSearchRequests, resp.Citations)
		}
	}
}

// A turn reporting the queries in both places must not count them twice.
func TestGeminiWebSearchNotCountedTwice(t *testing.T) {
	search := `{"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"queries":["iceland population","capital of peru"]},"id":"c1"}}`
	p := (&gsFake{rounds: [][]string{{`{"candidates":[{"content":{"parts":[` + search + `,{"text":"Lima."}]},"finishReason":"STOP","groundingMetadata":` + wsGrounding + `}]}`}}}).start(t)
	resp, err := p.complete(context.Background(), "gemini-3.1-pro", 100, Request{Messages: userMsg("x"), WebSearch: true})
	if err != nil {
		t.Fatal(err)
	}
	if resp.WebSearchRequests != 2 {
		t.Errorf("searches = %d, want 2", resp.WebSearchRequests)
	}
}

// Live on gemini-3.8-flash: with function tools on, the round after a search has no
// toolCall part and ran no search, yet groundingMetadata repeats the earlier queries.
// They must not be counted again, but its citations are the real ones.
func TestGeminiToolLoopDoesNotRecountRepeatedQueries(t *testing.T) {
	search := `{"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"queries":["iceland population","capital of peru"]},"id":"c1"}}`
	final := `{"candidates":[{"content":{"parts":[{"text":"Lima."}]},"finishReason":"STOP","groundingMetadata":` + wsGrounding + `}]}`
	for name, tc := range map[string]struct {
		first, second []string
		onEvent       func(StreamEvent)
	}{
		"plain":  {[]string{gsReply([]string{search, gsFuncCall}, "STOP")}, []string{final}, nil},
		"stream": {[]string{gsEvent(search, ""), gsEvent(gsFuncCall, "STOP")}, []string{final}, func(StreamEvent) {}},
	} {
		t.Run(name, func(t *testing.T) {
			f := &gsFake{rounds: [][]string{tc.first, tc.second}}
			r := &router{
				providers: map[Provider]provider{ProviderGoogle: f.start(t)},
				profiles:  map[Profile]ModelRef{ProfileChat: {Provider: ProviderGoogle, Model: "gemini-3.1-pro"}},
				maxTokens: 100,
				log:       slog.New(slog.NewTextHandler(io.Discard, nil)),
			}
			tools := map[string]ToolFunc{"lookup_ticket": func(context.Context, ToolCall) (string, error) { return "ok", nil }}
			run, err := RunTools(context.Background(), r, Request{Messages: userMsg("x"), WebSearch: true, Tools: []ToolDef{gsTool}, OnEvent: tc.onEvent}, tools, ToolLoopOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if run.WebSearchRequests() != 2 || run.Rounds[1].WebSearchRequests != 0 {
				t.Errorf("total = %d, round 2 = %d, want 2 and 0", run.WebSearchRequests(), run.Rounds[1].WebSearchRequests)
			}
			if !reflect.DeepEqual(run.Citations(), wsGeminiWant) {
				t.Errorf("citations = %+v", run.Citations())
			}
		})
	}
}

// A stream cut off after a search keeps the count from the toolCall part.
func TestGeminiWebSearchKeptWhenStreamCutOff(t *testing.T) {
	search := `{"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"queries":["a"]},"id":"c1"}}`
	p := (&gsFake{rounds: [][]string{{gsEvent(search, ""), gsEvent(`{"text":"par"}`, "")}}}).start(t)
	resp, err := p.complete(context.Background(), "gemini-3.1-pro", 100, Request{Messages: userMsg("x"), WebSearch: true, OnEvent: func(StreamEvent) {}})
	if err == nil {
		t.Fatal("want an error for the cut-off stream")
	}
	if resp.WebSearchRequests != 1 || !resp.UsageEstimated {
		t.Errorf("searches = %d estimated = %v, want 1 and true", resp.WebSearchRequests, resp.UsageEstimated)
	}
}

func TestGeminiNoWebSearchIsZeroAndNil(t *testing.T) {
	p := (&gsFake{rounds: [][]string{{gsReply([]string{`{"text":"hi"}`}, "STOP")}}}).start(t)
	resp, err := p.complete(context.Background(), "gemini-3.1-pro", 100, Request{Messages: userMsg("x")})
	if err != nil {
		t.Fatal(err)
	}
	if resp.WebSearchRequests != 0 || resp.Citations != nil {
		t.Errorf("searches = %d citations = %+v", resp.WebSearchRequests, resp.Citations)
	}
}

func TestGeminiBatchItemWebSearch(t *testing.T) {
	line := func(key, reply string) gbLine {
		var item gbLine
		if err := json.Unmarshal([]byte(`{"key":"`+key+`","response":`+reply+`}`), &item); err != nil {
			t.Fatal(err)
		}
		return item
	}
	g := &googleBatch{}
	// Without the setting the count is the metadata's.
	r := g.parseLine(batchChat, "gemini-3.1-pro", line("a", wsGeminiReply()))
	if r.err != nil || r.customID != "a" || r.resp.WebSearchRequests != 2 || len(r.resp.Citations) != 2 {
		t.Errorf("no setting: %+v", r)
	}
	// With it, repeated queries in the metadata are not counted, and the ID comes back clean.
	r = g.parseLine(batchChat, "gemini-3.1-pro", line("a"+geminiInvocationsKey, wsGeminiReply()))
	if r.err != nil || r.customID != "a" || r.resp.WebSearchRequests != 0 || len(r.resp.Citations) != 2 {
		t.Errorf("with setting: %+v", r)
	}
}

// Only a line sent with the setting on gets the marked key.
func TestGeminiBatchKeyMarksInvocations(t *testing.T) {
	if !geminiLineInvocations([]byte(`{"toolConfig":{"includeServerSideToolInvocations":true}}`)) {
		t.Error("toolConfig on: want true")
	}
	if geminiLineInvocations([]byte(`{"contents":[]}`)) {
		t.Error("no toolConfig: want false")
	}
}

// The batch path for Anthropic and the metering wrapper both carry the values.
func TestWebSearchThroughBatchUsage(t *testing.T) {
	var msg anthropic.Message
	reply := `{"id":"m","type":"message","role":"assistant","model":"claude-opus-5-5","stop_reason":"end_turn",
		"usage":{"input_tokens":5,"output_tokens":2,"server_tool_use":{"web_search_requests":3}},
		"content":[{"type":"text","text":"Lima","citations":[` + wsCitation("https://a.test", "A", "one") + `]}]}`
	if err := json.Unmarshal([]byte(reply), &msg); err != nil {
		t.Fatal(err)
	}
	resp, err := anthropicResponse("claude-opus-5-5", &msg)
	if err != nil {
		t.Fatal(err)
	}
	f := &fakeBatch{raws: []rawBatchResult{{customID: "a", resp: resp}}}
	var recorded []Usage
	b := WithBatchUsage(newFakeBatcher(testCfg(), ProviderAnthropic, f), nil, func(_ context.Context, u Usage) { recorded = append(recorded, u) })
	got, err := CollectResults(b.Results(context.Background(), makeBatchID(ProviderAnthropic, batchChat, "claude-opus-5-5", "p1")))
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Response.WebSearchRequests != 3 || len(got[0].Response.Citations) != 1 {
		t.Errorf("result = %+v", got[0].Response)
	}
	if len(recorded) != 1 || recorded[0].WebSearchRequests != 3 {
		t.Errorf("recorded = %+v", recorded)
	}
}

// A search with no token count must still reach the meter.
func TestHasUsageCountsSearches(t *testing.T) {
	if !hasUsage(Response{WebSearchRequests: 1}) {
		t.Error("a reply with a search has usage")
	}
}

// A last round with no function tools that echoes search steps must turn the setting
// on: otherwise repeated queries count as new, and new ones can't be told apart.
func TestGeminiEchoedSearchStepsTurnInvocationsOn(t *testing.T) {
	search := `{"thoughtSignature":"cw==","toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"queries":["iceland population","capital of peru"]},"id":"c1"}}`
	f := &gsFake{rounds: [][]string{{gsReply([]string{search, gsFuncCall}, "STOP")}, {wsGeminiReply()}, {wsGeminiReply()}}}
	p := f.start(t)
	first, err := p.complete(context.Background(), "gemini-3.1-pro", 100, Request{Messages: userMsg("x"), WebSearch: true, Tools: []ToolDef{gsTool}})
	if err != nil {
		t.Fatal(err)
	}
	history := append(userMsg("x"),
		Message{Role: "assistant", ToolCalls: first.ToolCalls, ProviderData: first.ProviderData},
		Message{Role: "user", ToolResults: []ToolResult{{ToolCallID: first.ToolCalls[0].ID, Content: "ok"}}})
	second, err := p.complete(context.Background(), "gemini-3.1-pro", 100, Request{Messages: history, WebSearch: true})
	if err != nil {
		t.Fatal(err)
	}
	if !sameJSON(f.body(t, 1)["toolConfig"], `{"includeServerSideToolInvocations":true}`) {
		t.Errorf("toolConfig = %s", f.body(t, 1)["toolConfig"])
	}
	if second.WebSearchRequests != 0 || len(second.Citations) != 2 {
		t.Errorf("searches = %d citations = %+v", second.WebSearchRequests, second.Citations)
	}
	// No echoed steps and no tools: the setting stays off.
	if _, err := p.complete(context.Background(), "gemini-3.1-pro", 100, Request{Messages: userMsg("x"), WebSearch: true}); err != nil {
		t.Fatal(err)
	}
	if got := f.body(t, 2)["toolConfig"]; got != nil {
		t.Errorf("toolConfig = %s, want none", got)
	}
}

// A stream cut off in a round after a search must not recount the repeated queries.
func TestGeminiCutOffStreamDoesNotRecount(t *testing.T) {
	partial := `{"candidates":[{"content":{"parts":[{"text":"Li"}]},"groundingMetadata":` + wsGrounding + `}]}`
	p := (&gsFake{rounds: [][]string{{partial}}}).start(t)
	resp, err := p.complete(context.Background(), "gemini-3.1-pro", 100, Request{Messages: userMsg("x"), WebSearch: true, Tools: []ToolDef{gsTool}, OnEvent: func(StreamEvent) {}})
	if err == nil {
		t.Fatal("want an error for the cut-off stream")
	}
	if resp.WebSearchRequests != 0 || len(resp.Citations) != 2 {
		t.Errorf("searches = %d citations = %+v", resp.WebSearchRequests, resp.Citations)
	}
}

// Citations in the shape Claude returned live (2026-11-05, claude-sonnet-5).
func wsCitation(url, title, text string) string {
	return `{"type":"web_search_result_location","url":"` + url + `","title":"` + title + `","encrypted_index":"enc","cited_text":"` + text + `"}`
}

var wsAnthropicWant = []Citation{
	{URL: "https://a.test", Title: "A", CitedText: "one"},
	{URL: "https://b.test", Title: "B", CitedText: "two"},
	{URL: "https://a.test", Title: "A", CitedText: "three"},
}

func TestAnthropicWebSearchUsage(t *testing.T) {
	c1, c2, c3 := wsCitation("https://a.test", "A", "one"), wsCitation("https://b.test", "B", "two"), wsCitation("https://a.test", "A", "three")
	reply := `{"id":"m","type":"message","role":"assistant","model":"claude-opus-5-5","stop_reason":"end_turn",
		"usage":{"input_tokens":5,"output_tokens":2,"server_tool_use":{"web_search_requests":3,"web_fetch_requests":0}},
		"content":[{"type":"server_tool_use","id":"s1","name":"web_search","input":{"query":"q"}},
		{"type":"text","text":"Lima","citations":[` + c1 + `,` + c2 + `]},{"type":"text","text":" is it","citations":[` + c1 + `,` + c3 + `]},{"type":"text","text":"."}]}`
	sse := msgStart(`{"input_tokens":5,"output_tokens":1}`) +
		blockStart(0, `{"type":"text","text":"","citations":[]}`) +
		blockDelta(0, `{"type":"text_delta","text":"Lima"}`) +
		blockDelta(0, `{"type":"citations_delta","citation":`+c1+`}`) +
		blockDelta(0, `{"type":"citations_delta","citation":`+c2+`}`) +
		blockStop(0) +
		blockStart(1, `{"type":"text","text":"","citations":[]}`) +
		blockDelta(1, `{"type":"text_delta","text":" is it."}`) +
		blockDelta(1, `{"type":"citations_delta","citation":`+c1+`}`) +
		blockDelta(1, `{"type":"citations_delta","citation":`+c3+`}`) +
		blockStop(1) +
		msgEnd("end_turn", `{"output_tokens":2,"server_tool_use":{"web_search_requests":3,"web_fetch_requests":0}}`)
	p := (&anthropicServer{reply: reply, sse: sse}).start(t)
	for name, req := range map[string]Request{
		"plain":  {Messages: userMsg("x"), WebSearch: true},
		"stream": {Messages: userMsg("x"), WebSearch: true, OnEvent: func(StreamEvent) {}},
	} {
		resp, err := p.complete(context.Background(), "claude-opus-5-5", 100, req)
		if err != nil {
			t.Fatal(err)
		}
		if resp.WebSearchRequests != 3 {
			t.Errorf("%s: searches = %d, want 3", name, resp.WebSearchRequests)
		}
		if !reflect.DeepEqual(resp.Citations, wsAnthropicWant) {
			t.Errorf("%s: citations = %+v, want %+v", name, resp.Citations, wsAnthropicWant)
		}
	}
}

// The count only arrives in message_delta, so a stream cut before it counts the
// searches that returned. A failed one is not billed. Citations already seen stay.
func TestAnthropicWebSearchStreamCutOff(t *testing.T) {
	sse := msgStart(`{"input_tokens":5,"output_tokens":1}`) +
		blockStart(0, `{"type":"web_search_tool_result","tool_use_id":"s1","content":[{"type":"web_search_result","url":"https://a.test","title":"A","encrypted_content":"e"}]}`) +
		blockStop(0) +
		blockStart(1, `{"type":"web_search_tool_result","tool_use_id":"s2","content":[]}`) +
		blockStop(1) +
		blockStart(2, `{"type":"web_search_tool_result","tool_use_id":"s3","content":{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}}`) +
		blockStop(2) +
		blockStart(3, `{"type":"text","text":"","citations":[]}`) +
		blockDelta(3, `{"type":"text_delta","text":"Lima"}`) +
		blockDelta(3, `{"type":"citations_delta","citation":`+wsCitation("https://a.test", "A", "one")+`}`)
	p := (&anthropicServer{sse: sse, abort: true}).start(t)
	resp, err := p.complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), WebSearch: true, OnEvent: func(StreamEvent) {}})
	if err == nil {
		t.Fatal("want an error for the cut-off stream")
	}
	if resp.WebSearchRequests != 2 || len(resp.Citations) != 1 || !resp.UsageEstimated {
		t.Errorf("searches = %d citations = %+v estimated = %v", resp.WebSearchRequests, resp.Citations, resp.UsageEstimated)
	}
}

// A schema reply has searches but no citations (live on claude-sonnet-5, 2026-11-05).
func TestAnthropicSchemaReplyHasCountWithoutCitations(t *testing.T) {
	reply := `{"id":"m","type":"message","role":"assistant","model":"claude-opus-5-5","stop_reason":"end_turn",
		"usage":{"input_tokens":5,"output_tokens":2,"server_tool_use":{"web_search_requests":2}},
		"content":[{"type":"text","text":"{\"answer\":\"Lima\"}"}]}`
	p := (&anthropicServer{reply: reply}).start(t)
	resp, err := p.complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), WebSearch: true,
		JSONSchema: map[string]any{"type": "object"}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.WebSearchRequests != 2 || resp.Citations != nil {
		t.Errorf("searches = %d citations = %+v", resp.WebSearchRequests, resp.Citations)
	}
}

func TestAnthropicWebSearchMaxUses(t *testing.T) {
	reply := `{"id":"m","type":"message","role":"assistant","model":"claude-opus-5-5","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[{"type":"text","text":"ok"}]}`
	for maxUses, want := range map[int]float64{0: 5, -1: 5, 12: 12} {
		as := &anthropicServer{reply: reply}
		p := as.start(t)
		_, err := p.complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), WebSearch: true, WebSearchMaxUses: maxUses})
		if err != nil {
			t.Fatal(err)
		}
		tool := as.bodies[0]["tools"].([]any)[0].(map[string]any)
		if tool["max_uses"] != want {
			t.Errorf("WebSearchMaxUses %d sent max_uses = %v, want %v", maxUses, tool["max_uses"], want)
		}
	}
}

func TestRunToolsWebSearchTotals(t *testing.T) {
	a := Citation{URL: "https://a.test"}
	b := Citation{URL: "https://b.test"}
	run := ToolRun{Rounds: []Response{
		{WebSearchRequests: 2, Citations: []Citation{a, b}},
		{},
		{WebSearchRequests: 1, Citations: []Citation{b}},
	}}
	if run.WebSearchRequests() != 3 {
		t.Errorf("total = %d, want 3", run.WebSearchRequests())
	}
	if got := run.Citations(); !reflect.DeepEqual(got, []Citation{a, b}) {
		t.Errorf("citations = %+v", got)
	}
	if (ToolRun{}).Citations() != nil {
		t.Error("an empty run must give nil")
	}
}

// RunTools leaves each round's own count alone, so a caller using WithUsage (one
// record per round) and one using the totals agree.
func TestRunToolsRoundsKeepTheirOwnCount(t *testing.T) {
	search := `{"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"queries":["a"]},"id":"c1"}}`
	f := &gsFake{rounds: [][]string{
		{gsReply([]string{search, gsFuncCall}, "STOP")},
		{gsReply([]string{search, `{"text":"done"}`}, "STOP")},
	}}
	r := &router{
		providers: map[Provider]provider{ProviderGoogle: f.start(t)},
		profiles:  map[Profile]ModelRef{ProfileChat: {Provider: ProviderGoogle, Model: "gemini-3.1-pro"}},
		maxTokens: 100,
		log:       slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	var recorded int
	c := WithUsage(r, nil, func(_ context.Context, u Usage) { recorded += u.WebSearchRequests })
	tools := map[string]ToolFunc{"lookup_ticket": func(context.Context, ToolCall) (string, error) { return "ok", nil }}
	run, err := RunTools(context.Background(), c, Request{Messages: userMsg("x"), WebSearch: true, Tools: []ToolDef{gsTool}}, tools, ToolLoopOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if run.Response.WebSearchRequests != 1 || run.WebSearchRequests() != 2 || recorded != 2 {
		t.Errorf("last = %d total = %d recorded = %d, want 1, 2, 2", run.Response.WebSearchRequests, run.WebSearchRequests(), recorded)
	}
}

// After a failover the values come from the provider that served the reply.
func TestWebSearchThroughFailover(t *testing.T) {
	r := &router{
		providers: map[Provider]provider{
			ProviderAnthropic: providerAt(ProviderAnthropic, statusServer(t, 500, errBody(ProviderAnthropic, 500))),
			ProviderGoogle:    (&gsFake{rounds: [][]string{{wsGeminiReply()}}}).start(t),
		},
		profiles:  map[Profile]ModelRef{ProfileChat: {Provider: ProviderAnthropic, Model: "claude-opus-5-5"}},
		fallbacks: map[Profile]ModelRef{ProfileChat: {Provider: ProviderGoogle, Model: "gemini-3.1-pro"}},
		maxTokens: 100,
		log:       slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	resp, err := r.Complete(context.Background(), Request{Messages: userMsg("x"), WebSearch: true})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Provider != ProviderGoogle || resp.WebSearchRequests != 2 || len(resp.Citations) != 2 {
		t.Errorf("resp = %+v", resp)
	}
}

func TestDedupeCitations(t *testing.T) {
	a := Citation{URL: "u", CitedText: "1"}
	if got := dedupeCitations([]Citation{a, {Title: "no url"}, {URL: "u", CitedText: "2"}, a}); !reflect.DeepEqual(got, []Citation{a, {URL: "u", CitedText: "2"}}) {
		t.Errorf("got %+v", got)
	}
	if dedupeCitations(nil) != nil {
		t.Error("nil must stay nil")
	}
}

// wsAnthropicFake plays one scripted reply per request, as JSON or as SSE by what the
// request asked for, and keeps every request body.
type wsAnthropicFake struct {
	mu     sync.Mutex
	json   []string
	sse    []string
	bodies []map[string]any
}

func (f *wsAnthropicFake) start(t *testing.T) *anthropicProvider {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		f.mu.Lock()
		f.bodies = append(f.bodies, body)
		i := len(f.bodies) - 1
		f.mu.Unlock()
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			if i < len(f.sse) {
				_, _ = io.WriteString(w, f.sse[i])
			}
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if i >= len(f.json) {
			w.WriteHeader(400)
			_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}`)
			return
		}
		_, _ = io.WriteString(w, f.json[i])
	}))
	t.Cleanup(srv.Close)
	return newAnthropic("k", nil, option.WithBaseURL(srv.URL)).(*anthropicProvider)
}

func (f *wsAnthropicFake) messages(i int) []any { return f.bodies[i]["messages"].([]any) }

const (
	wsSearchUse    = `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"lima <q>"}}`
	wsSearchResult = `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://a.test","title":"A","encrypted_content":"ENC+/=","page_age":"April 2025"}]}`
	wsClientCall   = `{"type":"tool_use","id":"toolu_1","name":"lookup_ticket","input":{"id":"A1"}}`
)

func wsMsg(stop string, searches int, blocks ...string) string {
	return `{"id":"m","type":"message","role":"assistant","model":"claude-opus-5-5","stop_reason":"` + stop +
		`","usage":{"input_tokens":10,"output_tokens":2,"cache_read_input_tokens":1,"cache_creation_input_tokens":3,"server_tool_use":{"web_search_requests":` +
		strconv.Itoa(searches) + `}},"content":[` + strings.Join(blocks, ",") + `]}`
}

func wsTextBlock(text string) string { return `{"type":"text","text":"` + text + `"}` }

// wsSSE streams the same blocks a JSON reply holds.
func wsSSE(stop string, searches int, blocks ...string) string {
	out := msgStart(`{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":1,"cache_creation_input_tokens":3}`)
	for i, b := range blocks {
		var m map[string]any
		_ = json.Unmarshal([]byte(b), &m)
		switch m["type"] {
		case "text":
			out += blockStart(i, `{"type":"text","text":"","citations":[]}`)
			out += blockDelta(i, `{"type":"text_delta","text":`+strconv.Quote(m["text"].(string))+`}`)
		case "server_tool_use", "tool_use":
			input, _ := json.Marshal(m["input"])
			m["input"] = map[string]any{}
			start, _ := json.Marshal(m)
			out += blockStart(i, string(start))
			out += blockDelta(i, `{"type":"input_json_delta","partial_json":`+strconv.Quote(string(input))+`}`)
		default:
			out += blockStart(i, b)
		}
		out += blockStop(i)
	}
	return out + msgEnd(stop, `{"output_tokens":2,"server_tool_use":{"web_search_requests":`+strconv.Itoa(searches)+`}}`)
}

func TestAnthropicEchoesSearchStepsInToolLoop(t *testing.T) {
	blocks := []string{wsSearchUse, wsSearchResult, wsTextBlock("Checking."), wsClientCall}
	for name, onEvent := range map[string]func(StreamEvent){"plain": nil, "stream": func(StreamEvent) {}} {
		t.Run(name, func(t *testing.T) {
			final := []string{wsTextBlock("Done.")}
			f := &wsAnthropicFake{
				json: []string{wsMsg("tool_use", 1, blocks...), wsMsg("end_turn", 0, final...)},
				sse:  []string{wsSSE("tool_use", 1, blocks...), wsSSE("end_turn", 0, final...)},
			}
			r := &router{
				providers: map[Provider]provider{ProviderAnthropic: f.start(t)},
				profiles:  map[Profile]ModelRef{ProfileChat: {Provider: ProviderAnthropic, Model: "claude-opus-5-5"}},
				maxTokens: 100,
				log:       slog.New(slog.NewTextHandler(io.Discard, nil)),
			}
			tools := map[string]ToolFunc{"lookup_ticket": func(context.Context, ToolCall) (string, error) { return "ok", nil }}
			req := Request{Messages: userMsg("x"), WebSearch: true, Tools: []ToolDef{gsTool}, OnEvent: onEvent}
			run, err := RunTools(context.Background(), r, req, tools, ToolLoopOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if run.WebSearchRequests() != 1 {
				t.Errorf("searches = %d, want 1", run.WebSearchRequests())
			}
			msgs := f.messages(1)
			got, _ := json.Marshal(msgs[1].(map[string]any)["content"])
			if !sameJSON(got, `[`+strings.Join(blocks, ",")+`]`) {
				t.Errorf("assistant turn sent back = %s", got)
			}
			if u := msgs[2].(map[string]any)["content"].([]any)[0].(map[string]any); u["type"] != "tool_result" || u["tool_use_id"] != "toolu_1" {
				t.Errorf("tool result = %+v", u)
			}
		})
	}
}

// A turn without search needs no echo, so nothing changes for it.
func TestAnthropicNoSearchNoProviderData(t *testing.T) {
	p := (&wsAnthropicFake{json: []string{wsMsg("end_turn", 0, wsTextBlock("hi"))}}).start(t)
	resp, err := p.complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x")})
	if err != nil || resp.ProviderData != nil {
		t.Errorf("data = %s, err = %v", resp.ProviderData, err)
	}
}

// The echo falls back to the rebuilt turn when it can't be trusted or used.
func TestAnthropicEchoFallsBack(t *testing.T) {
	data := func(provider string) []byte {
		return []byte(`{"provider":"` + provider + `","content":[` + wsSearchUse + `,` + wsClientCall + `]}`)
	}
	call := []ToolCall{{ID: "toolu_1", Name: "lookup_ticket", Input: json.RawMessage(`{}`)}}
	for name, tc := range map[string]struct {
		m         Message
		webSearch bool
		wantEcho  bool
	}{
		"matches":        {Message{ProviderData: data("anthropic"), ToolCalls: call}, true, true},
		"search off":     {Message{ProviderData: data("anthropic"), ToolCalls: call}, false, false},
		"other provider": {Message{ProviderData: data("google"), ToolCalls: call}, true, false},
		"other call":     {Message{ProviderData: data("anthropic"), ToolCalls: []ToolCall{{ID: "toolu_2", Name: "x"}}}, true, false},
		"call removed":   {Message{ProviderData: data("anthropic")}, true, false},
		"bad data":       {Message{ProviderData: []byte("{"), ToolCalls: call}, true, false},
	} {
		tc.m.Role, tc.m.Content = "assistant", "rebuilt"
		// The search here has no result, so it needs tool results right after it.
		answer := Message{Role: "user", ToolResults: []ToolResult{{ToolCallID: "toolu_1", Content: "ok"}}}
		_, msgs := splitSystemAndMessages(Request{Messages: []Message{tc.m, answer}, WebSearch: tc.webSearch})
		raw, _ := json.Marshal(msgs[0])
		if echoed := strings.Contains(string(raw), "srvtoolu_1"); echoed != tc.wantEcho {
			t.Errorf("%s: echoed = %v, body %s", name, echoed, raw)
		}
	}
}

func TestAnthropicContinuesPausedTurn(t *testing.T) {
	first := []string{wsSearchUse, wsSearchResult, wsTextBlock("Lima")}
	second := []string{wsTextBlock(" is it.")}
	for name, onEvent := range map[string]func(StreamEvent){"plain": nil, "stream": func(StreamEvent) {}} {
		t.Run(name, func(t *testing.T) {
			var streamed string
			req := Request{Messages: userMsg("x"), WebSearch: true}
			if onEvent != nil {
				req.OnEvent = func(e StreamEvent) { streamed += e.Text }
			}
			f := &wsAnthropicFake{
				json: []string{wsMsg("pause_turn", 2, first...), wsMsg("end_turn", 1, second...)},
				sse:  []string{wsSSE("pause_turn", 2, first...), wsSSE("end_turn", 1, second...)},
			}
			resp, err := f.start(t).complete(context.Background(), "claude-opus-5-5", 100, req)
			if err != nil {
				t.Fatal(err)
			}
			if resp.Text != "Lima is it." || resp.StopReason != StopEndTurn {
				t.Errorf("text = %q stop = %q", resp.Text, resp.StopReason)
			}
			if resp.WebSearchRequests != 3 || resp.InputTokens != 20 || resp.OutputTokens != 4 || resp.CachedTokens != 2 || resp.CacheWriteTokens != 6 {
				t.Errorf("usage = %+v", resp)
			}
			if onEvent != nil && streamed != "Lima is it." {
				t.Errorf("streamed = %q", streamed)
			}
			if len(f.bodies) != 2 || len(f.bodies[1]["tools"].([]any)) != 1 {
				t.Fatalf("requests = %+v", f.bodies)
			}
			msgs := f.messages(1)
			got, _ := json.Marshal(msgs[1].(map[string]any)["content"])
			if len(msgs) != 2 || !sameJSON(got, `[`+strings.Join(first, ",")+`]`) {
				t.Errorf("continuation messages = %+v", msgs)
			}
			var d anthropicProviderData
			if err := json.Unmarshal(resp.ProviderData, &d); err != nil || len(d.Content) != 4 {
				t.Errorf("provider data = %s (%v)", resp.ProviderData, err)
			}
		})
	}
}

// Each pause adds its own new content to the history, in order.
func TestAnthropicPausedTwiceKeepsEveryReply(t *testing.T) {
	f := &wsAnthropicFake{json: []string{
		wsMsg("pause_turn", 1, wsTextBlock("a")),
		wsMsg("pause_turn", 1, wsTextBlock("b")),
		wsMsg("end_turn", 0, wsTextBlock("c")),
	}}
	resp, err := f.start(t).complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), WebSearch: true})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "abc" || resp.WebSearchRequests != 2 {
		t.Errorf("resp = %+v", resp)
	}
	msgs := f.messages(2)
	a, _ := json.Marshal(msgs[1].(map[string]any)["content"])
	b, _ := json.Marshal(msgs[2].(map[string]any)["content"])
	if len(msgs) != 3 || !sameJSON(a, `[`+wsTextBlock("a")+`]`) || !sameJSON(b, `[`+wsTextBlock("b")+`]`) {
		t.Errorf("messages = %+v", msgs)
	}
}

func TestAnthropicPausedPastLimitIsTruncated(t *testing.T) {
	f := &wsAnthropicFake{}
	for i := 0; i <= maxPauseContinuations; i++ {
		f.json = append(f.json, wsMsg("pause_turn", 1, wsTextBlock("x")))
	}
	resp, err := f.start(t).complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), WebSearch: true})
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != StopTruncated || len(f.bodies) != maxPauseContinuations+1 || resp.WebSearchRequests != maxPauseContinuations+1 {
		t.Errorf("stop = %q requests = %d searches = %d", resp.StopReason, len(f.bodies), resp.WebSearchRequests)
	}
}

// A continuation that fails must not lose what the paused replies billed.
func TestAnthropicContinuationErrorKeepsPausedUsage(t *testing.T) {
	f := &wsAnthropicFake{json: []string{wsMsg("pause_turn", 2, wsTextBlock("Lima"))}}
	resp, err := f.start(t).complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), WebSearch: true})
	if err == nil {
		t.Fatal("want the continuation's error")
	}
	if resp.Text != "Lima" || resp.WebSearchRequests != 2 || resp.InputTokens != 10 || resp.OutputTokens != 2 {
		t.Errorf("resp = %+v", resp)
	}
}

// A paused batch item can't be continued here, so it reads as incomplete.
func TestAnthropicPausedBatchItemIsTruncated(t *testing.T) {
	var msg anthropic.Message
	if err := json.Unmarshal([]byte(wsMsg("pause_turn", 1, wsSearchUse, wsSearchResult)), &msg); err != nil {
		t.Fatal(err)
	}
	resp, err := anthropicResponse("claude-opus-5-5", &msg)
	if err != nil || resp.StopReason != StopTruncated || resp.WebSearchRequests != 1 || resp.ProviderData == nil {
		t.Errorf("resp = %+v err = %v", resp, err)
	}
}

// Search steps must pair up across the history, or the API rejects it. Anything that
// doesn't pair is rebuilt, which always worked.
func TestAnthropicEchoPairing(t *testing.T) {
	pending := func(blocks ...string) []byte {
		return []byte(`{"provider":"anthropic","content":[` + strings.Join(blocks, ",") + `]}`)
	}
	call := []ToolCall{{ID: "toolu_1", Name: "lookup_ticket", Input: json.RawMessage(`{}`)}}
	a := Message{Role: "assistant", Content: "rebuilt A", ToolCalls: call, ProviderData: pending(wsSearchUse, wsClientCall)}
	b := Message{Role: "assistant", Content: "rebuilt B", ProviderData: pending(wsSearchResult, wsTextBlock("done"))}
	results := Message{Role: "user", ToolResults: []ToolResult{{ToolCallID: "toolu_1", Content: "ok"}}}
	withText := results
	withText.Content = "and also"
	sys := Message{Role: "system", Content: "be brief"}
	user := Message{Role: "user", Content: "next"}
	lost := b
	lost.ProviderData = nil
	lostA := a
	lostA.ProviderData = nil
	done := Message{Role: "assistant", Content: "rebuilt", ProviderData: pending(wsSearchUse, wsSearchResult, wsTextBlock("x"))}
	for name, tc := range map[string]struct {
		msgs []Message
		want []bool // for each assistant turn: was it echoed?
	}{
		"resolved turn, then user text":    {[]Message{done, user}, []bool{true}},
		"pending, then tool results":       {[]Message{a, results}, []bool{true}},
		"pending, result in the next turn": {[]Message{a, results, b, user}, []bool{true, true}},
		"system turn between":              {[]Message{a, sys, results, sys, b}, []bool{true, true}},
		"pending at the end":               {[]Message{a}, []bool{false}},
		"pending, then user text":          {[]Message{a, user}, []bool{false}},
		"text on the tool results":         {[]Message{a, withText}, []bool{false}},
		"another user turn after results":  {[]Message{a, results, user}, []bool{false}},
		"next turn lost its data":          {[]Message{a, results, lost}, []bool{false, false}},
		"result's search was trimmed":      {[]Message{lostA, results, b}, []bool{false, false}},
		"result with no turn before":       {[]Message{b}, []bool{false}},
	} {
		_, msgs := splitSystemAndMessages(Request{Messages: tc.msgs, WebSearch: true})
		var got []bool
		for _, m := range msgs {
			if m.Role != anthropic.MessageParamRoleAssistant {
				continue
			}
			raw, _ := json.Marshal(m)
			got = append(got, !strings.Contains(string(raw), "rebuilt"))
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: echoed = %v, want %v", name, got, tc.want)
		}
	}
}

// Every block type goes back in the form the API takes, and empty text is dropped.
func TestAnthropicEchoJSONBlocks(t *testing.T) {
	c := wsCitation("https://a.test", "A", "one")
	blocks := []string{
		`{"type":"thinking","thinking":"hm","signature":"sig"}`,
		`{"type":"redacted_thinking","data":"zzz"}`,
		wsSearchUse,
		`{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}}`,
		`{"type":"text","text":"Lima","citations":[` + c + `]}`,
	}
	in := append(append([]string{}, blocks...), `{"type":"text","text":""}`)
	var msg anthropic.Message
	if err := json.Unmarshal([]byte(wsMsg("end_turn", 1, in...)), &msg); err != nil {
		t.Fatal(err)
	}
	raw, ok := anthropicEchoJSON(msg.Content)
	if !ok || len(raw) != len(blocks) {
		t.Fatalf("ok = %v, blocks = %d", ok, len(raw))
	}
	for i, want := range blocks {
		if !sameJSON(raw[i], want) {
			t.Errorf("block %d = %s, want %s", i, raw[i], want)
		}
	}
	// A block type this code doesn't know can't be sent back.
	var unknown anthropic.Message
	if err := json.Unmarshal([]byte(wsMsg("end_turn", 0, `{"type":"future_block"}`)), &unknown); err != nil {
		t.Fatal(err)
	}
	if _, ok := anthropicEchoJSON(unknown.Content); ok {
		t.Error("unknown block: want false")
	}
}

// A paused reply with nothing to send back can't continue, and must not send an
// empty assistant message.
func TestAnthropicPausedWithNoContentStops(t *testing.T) {
	f := &wsAnthropicFake{json: []string{wsMsg("pause_turn", 1, `{"type":"text","text":""}`)}}
	resp, err := f.start(t).complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), WebSearch: true})
	if err != nil || resp.StopReason != StopTruncated || len(f.bodies) != 1 {
		t.Errorf("stop = %q err = %v requests = %d", resp.StopReason, err, len(f.bodies))
	}
}

// A continuation stream that breaks keeps the paused usage and its own partial.
func TestAnthropicContinuationStreamCutOffKeepsAll(t *testing.T) {
	cut := msgStart(`{"input_tokens":7,"output_tokens":1}`) +
		blockStart(0, `{"type":"text","text":"","citations":[]}`) +
		blockDelta(0, `{"type":"text_delta","text":" is"}`)
	f := &wsAnthropicFake{sse: []string{wsSSE("pause_turn", 2, wsTextBlock("Lima")), cut}}
	resp, err := f.start(t).complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), WebSearch: true, OnEvent: func(StreamEvent) {}})
	if err == nil {
		t.Fatal("want the cut-off error")
	}
	if resp.Text != "Lima is" || resp.WebSearchRequests != 2 || resp.InputTokens != 17 || !resp.UsageEstimated {
		t.Errorf("resp = %+v", resp)
	}
}

func TestAnthropicPausedSchemaErrorSaysIncomplete(t *testing.T) {
	_, err := checkJSON(Response{StopReason: StopTruncated, Provider: ProviderAnthropic, Model: "m"})
	if !errors.Is(err, ErrSchemaViolation) || !strings.Contains(err.Error(), "incomplete") {
		t.Errorf("err = %v", err)
	}
}
