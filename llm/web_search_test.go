package llm

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"reflect"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
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
