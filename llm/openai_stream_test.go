package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// oaiStreamFake is a fake /chat/completions endpoint. It records request bodies and
// answers with status; for a 2xx it writes each chunk as one SSE event, flushing after
// each. With hold set it then goes quiet until the client leaves.
type oaiStreamFake struct {
	status int
	chunks []string // data payloads, e.g. `{"choices":[...]}` or `[DONE]`
	raw    string   // reply body for a non-2xx status
	hold   bool
	bodies []map[string]any
}

func (f *oaiStreamFake) start(t *testing.T, logs io.Writer) *openaiProvider {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		f.bodies = append(f.bodies, body)
		if f.status >= 300 {
			w.WriteHeader(f.status)
			_, _ = w.Write([]byte(f.raw))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range f.chunks {
			_, _ = w.Write([]byte("data: " + c + "\n\n"))
			w.(http.Flusher).Flush()
		}
		if f.hold {
			<-r.Context().Done()
		}
	}))
	t.Cleanup(srv.Close)
	p := &openaiProvider{apiKey: "k", baseURL: srv.URL, http: http.DefaultClient}
	if logs != nil {
		p.log = slog.New(slog.NewTextHandler(logs, nil))
	}
	return p
}

func oaiTextChunk(s string) string {
	b, _ := json.Marshal(s)
	return `{"choices":[{"index":0,"delta":{"content":` + string(b) + `}}]}`
}

func oaiFinishChunk(fr string) string {
	return `{"choices":[{"index":0,"delta":{},"finish_reason":"` + fr + `"}]}`
}

const oaiUsageChunk = `{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":30}}}`

func oaiToolChunk(index int, id, name, args string) string {
	a, _ := json.Marshal(args)
	fn := `{"arguments":` + string(a)
	if name != "" {
		fn += `,"name":"` + name + `"`
	}
	fn += `}`
	idPart := ""
	if id != "" {
		idPart = `"id":"` + id + `","type":"function",`
	}
	return `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":` + strconv.Itoa(index) + `,` + idPart + `"function":` + fn + `}]}}]}`
}

func streamReq(onEvent func(StreamEvent)) Request {
	return Request{Messages: userMsg("hi"), OnEvent: onEvent}
}

func TestOpenAIStreamRequestShape(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{oaiTextChunk("a"), oaiFinishChunk("stop"), oaiUsageChunk, "[DONE]"}}
	p := f.start(t, nil)
	var got []string
	if _, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectText(&got))); err != nil {
		t.Fatal(err)
	}
	if f.bodies[0]["stream"] != true {
		t.Errorf("stream = %v, want true", f.bodies[0]["stream"])
	}
	so, _ := f.bodies[0]["stream_options"].(map[string]any)
	if so["include_usage"] != true {
		t.Errorf("stream_options = %v, want include_usage true", f.bodies[0]["stream_options"])
	}

	plain := &oaiStreamFake{chunks: []string{`{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}]}`}}
	// The plain path reads one JSON document, so reuse the fake with a bare body.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		plain.bodies = append(plain.bodies, body)
		_, _ = w.Write([]byte(plain.chunks[0]))
	}))
	defer srv.Close()
	pp := &openaiProvider{apiKey: "k", baseURL: srv.URL, http: http.DefaultClient}
	if _, err := pp.complete(context.Background(), "gpt-4o", 100, Request{Messages: userMsg("hi")}); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"stream", "stream_options"} {
		if _, ok := plain.bodies[0][k]; ok {
			t.Errorf("without OnEvent the body must not carry %q: %v", k, plain.bodies[0])
		}
	}
}

func TestOpenAIStreamTextInOrder(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
		oaiTextChunk("Hel"), oaiTextChunk("lo, "), oaiTextChunk("world."),
		oaiFinishChunk("stop"), "[DONE]",
	}}
	p := f.start(t, nil)
	var got []string
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectText(&got)))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "Hel|lo, |world." {
		t.Errorf("events = %q", got)
	}
	if resp.Text != "Hello, world." || resp.StopReason != StopEndTurn {
		t.Errorf("resp = %+v", resp)
	}
}

func TestOpenAIStreamToolCallsAssembled(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{
		oaiToolChunk(0, "call_a", "lookup", ""),
		oaiToolChunk(1, "call_b", "weather", ""),
		oaiToolChunk(0, "", "", `{"q":`),
		oaiToolChunk(1, "", "", `{"city":`),
		oaiToolChunk(0, "", "", `"cats"`),
		oaiToolChunk(1, "", "", `"Jakarta"`),
		oaiToolChunk(0, "", "", `}`),
		oaiToolChunk(1, "", "", `}`),
		oaiFinishChunk("tool_calls"), "[DONE]",
	}}
	p := f.start(t, nil)
	var got []string
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectText(&got)))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("tool arguments must not be sent as text: %q", got)
	}
	if resp.StopReason != StopToolUse || len(resp.ToolCalls) != 2 {
		t.Fatalf("resp = %+v", resp)
	}
	a, b := resp.ToolCalls[0], resp.ToolCalls[1]
	if a.ID != "call_a" || a.Name != "lookup" || string(a.Input) != `{"q":"cats"}` {
		t.Errorf("call 0 = %+v (%s)", a, a.Input)
	}
	if b.ID != "call_b" || b.Name != "weather" || string(b.Input) != `{"city":"Jakarta"}` {
		t.Errorf("call 1 = %+v (%s)", b, b.Input)
	}
}

func TestOpenAIStreamNoArgumentToolCall(t *testing.T) {
	for name, chunks := range map[string][]string{
		"never sent": {`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"ping"}}]}}]}`},
		"empty":      {oaiToolChunk(0, "c1", "ping", "")},
	} {
		f := &oaiStreamFake{chunks: append(chunks, oaiFinishChunk("tool_calls"), "[DONE]")}
		p := f.start(t, nil)
		resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(func(StreamEvent) {}))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(resp.ToolCalls) != 1 || string(resp.ToolCalls[0].Input) != `{}` {
			t.Errorf("%s: calls = %+v", name, resp.ToolCalls)
		}
	}
}

func TestOpenAIStreamUsageFromFinalChunk(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{oaiTextChunk("a"), oaiFinishChunk("stop"), oaiUsageChunk, "[DONE]"}}
	var logs bytes.Buffer
	p := f.start(t, &logs)
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(func(StreamEvent) {}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.InputTokens != 70 || resp.CachedTokens != 30 || resp.OutputTokens != 20 {
		t.Errorf("tokens = in %d cached %d out %d, want 70/30/20", resp.InputTokens, resp.CachedTokens, resp.OutputTokens)
	}
	if logs.Len() != 0 {
		t.Errorf("unexpected log: %s", logs.String())
	}
}

func TestOpenAIStreamUsageOnChoiceChunk(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{
		oaiTextChunk("a"),
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4}}`,
		"[DONE]",
	}}
	p := f.start(t, nil)
	resp, err := p.complete(context.Background(), "deepseek-chat", 100, streamReq(func(StreamEvent) {}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.InputTokens != 10 || resp.OutputTokens != 4 {
		t.Errorf("tokens = %d/%d, want 10/4", resp.InputTokens, resp.OutputTokens)
	}
}

func TestOpenAIStreamNoUsageWarns(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{oaiTextChunk("a"), oaiFinishChunk("stop"), "[DONE]"}}
	var logs bytes.Buffer
	p := f.start(t, &logs)
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(func(StreamEvent) {}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.InputTokens != 0 || resp.OutputTokens != 0 || resp.CachedTokens != 0 {
		t.Errorf("tokens must be 0, got %+v", resp)
	}
	if !strings.Contains(logs.String(), "llm: stream reply had no usage; tokens reported as 0") {
		t.Errorf("log = %q", logs.String())
	}
}

func TestOpenAIStreamNoUsageNilLoggerIsSafe(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{oaiTextChunk("a"), oaiFinishChunk("stop"), "[DONE]"}}
	p := f.start(t, nil)
	if _, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(func(StreamEvent) {})); err != nil {
		t.Fatal(err)
	}
}

func TestOpenAIStreamEndsWithoutDone(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{oaiTextChunk("partial")}}
	p := f.start(t, nil)
	var got []string
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectText(&got)))
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("err = %v, want io.ErrUnexpectedEOF", err)
	}
	if !shouldFailover(err) {
		t.Errorf("a truncated stream must be a failover-able error: %v", err)
	}
	if resp.Text != "" {
		t.Errorf("resp = %+v, want zero", resp)
	}
}

func TestOpenAIStreamErrorEvent(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{oaiTextChunk("a"), `{"error":{"message":"the model is overloaded"}}`}}
	p := f.start(t, nil)
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(func(StreamEvent) {}))
	if err == nil || !strings.Contains(err.Error(), "the model is overloaded") {
		t.Fatalf("err = %v", err)
	}
	if resp.Text != "" {
		t.Errorf("resp = %+v, want zero", resp)
	}
}

func TestOpenAIStreamStatusErrorBeforeStream(t *testing.T) {
	f := &oaiStreamFake{status: 429, raw: `{"error":{"message":"slow down"}}`}
	p := f.start(t, nil)
	var got []string
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectText(&got)))
	if err == nil || !strings.Contains(err.Error(), "openai: slow down (status 429)") {
		t.Fatalf("err = %v", err)
	}
	if len(got) != 0 || resp.Text != "" {
		t.Errorf("events = %q resp = %+v", got, resp)
	}
}

func TestOpenAIStreamRefusalField(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{
		oaiTextChunk("I can "),
		`{"choices":[{"index":0,"delta":{"refusal":"Sorry, "}}]}`,
		`{"choices":[{"index":0,"delta":{"refusal":"no."}}]}`,
		oaiFinishChunk("stop"), oaiUsageChunk, "[DONE]",
	}}
	p := f.start(t, nil)
	var got []string
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectText(&got)))
	var re *RefusalError
	if !errors.Is(err, ErrRefused) || !errors.As(err, &re) {
		t.Fatalf("err = %v, want a refusal", err)
	}
	if re.Category != "refusal" || re.Explanation != "Sorry, no." {
		t.Errorf("refusal = %+v", re)
	}
	if resp.Text != "I can " || strings.Join(got, "") != "I can " {
		t.Errorf("text = %q events = %q (the refusal text must not be sent)", resp.Text, got)
	}
	if resp.OutputTokens != 20 {
		t.Errorf("usage not populated: %+v", resp)
	}
}

func TestOpenAIStreamContentFilter(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{oaiTextChunk("Once upon"), oaiFinishChunk("content_filter"), oaiUsageChunk, "[DONE]"}}
	p := f.start(t, nil)
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(func(StreamEvent) {}))
	var re *RefusalError
	if !errors.As(err, &re) || re.Category != "content_filter" {
		t.Fatalf("err = %v, want content_filter refusal", err)
	}
	if resp.Text != "Once upon" || resp.InputTokens != 70 {
		t.Errorf("resp = %+v", resp)
	}
}

func TestOpenAIStreamOnlyChoiceZero(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{
		`{"choices":[{"index":1,"delta":{"content":"other"}}]}`,
		oaiTextChunk("mine"), oaiFinishChunk("stop"), "[DONE]",
	}}
	p := f.start(t, nil)
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(func(StreamEvent) {}))
	if err != nil || resp.Text != "mine" {
		t.Errorf("resp = %+v err = %v", resp, err)
	}
}

func TestOpenAIStreamQuietPastDeadline(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{oaiTextChunk("first")}, hold: true}
	p := f.start(t, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	var got []string
	_, err := p.complete(ctx, "gpt-4o", 100, streamReq(collectText(&got)))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
}

func TestOpenAIStreamCancelledFromOnEvent(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{oaiTextChunk("first"), oaiTextChunk("second")}, hold: true}
	p := f.start(t, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var got []string
	record := collectText(&got)
	_, err := p.complete(ctx, "gpt-4o", 100, streamReq(func(ev StreamEvent) { record(ev); cancel() }))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(got) != 1 || got[0] != "first" {
		t.Errorf("events = %q, want exactly one", got)
	}
}

func TestOpenAIStreamSkipsEmptyPayload(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{oaiTextChunk("a"), "", oaiTextChunk("b"), oaiFinishChunk("stop"), oaiUsageChunk, "[DONE]"}}
	p := f.start(t, nil)
	var got []string
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectText(&got)))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "a|b" || resp.Text != "ab" {
		t.Errorf("events = %q, resp = %+v", got, resp)
	}
}

func TestOpenAIStreamToolStartBeforeNextPiece(t *testing.T) {
	started := make(chan struct{})
	var gated atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		send := func(c string) {
			_, _ = w.Write([]byte("data: " + c + "\n\n"))
			w.(http.Flusher).Flush()
		}
		send(oaiToolChunk(0, "call_1", "search_docs", ""))
		select {
		case <-started:
			gated.Store(true)
		case <-time.After(2 * time.Second):
		}
		for _, c := range []string{
			oaiToolChunk(0, "", "", `{"q":`), oaiToolChunk(0, "", "", `"a"`), oaiToolChunk(0, "", "", `}`),
			oaiFinishChunk("tool_calls"), "[DONE]",
		} {
			send(c)
		}
	}))
	t.Cleanup(srv.Close)
	p := &openaiProvider{apiKey: "k", baseURL: srv.URL, http: http.DefaultClient}
	var got []StreamEvent
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(func(ev StreamEvent) {
		got = append(got, ev)
		if ev.Kind == StreamToolStart {
			close(started)
		}
	}))
	if err != nil {
		t.Fatal(err)
	}
	if !gated.Load() {
		t.Error("tool_start was not sent before the next piece was read")
	}
	checkEvents(t, got, []StreamEvent{
		toolEv(StreamToolStart, "call_1", "search_docs"),
		toolEv(StreamToolReady, "call_1", "search_docs"),
	})
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "call_1" || resp.ToolCalls[0].Name != "search_docs" {
		t.Errorf("tool calls = %+v", resp.ToolCalls)
	}
}

func TestOpenAIStreamToolEventsInterleaved(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{
		oaiToolChunk(0, "call_a", "lookup", ""),
		oaiToolChunk(1, "call_b", "weather", ""),
		oaiToolChunk(0, "", "", `{}`),
		oaiToolChunk(1, "", "", `{}`),
		oaiFinishChunk("tool_calls"), "[DONE]",
	}}
	p := f.start(t, nil)
	var got []StreamEvent
	if _, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectEvents(&got))); err != nil {
		t.Fatal(err)
	}
	checkEvents(t, got, []StreamEvent{
		toolEv(StreamToolStart, "call_a", "lookup"),
		toolEv(StreamToolStart, "call_b", "weather"),
		toolEv(StreamToolReady, "call_a", "lookup"),
		toolEv(StreamToolReady, "call_b", "weather"),
	})
}

func TestOpenAIStreamToolStartWaitsForName(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{
		oaiToolChunk(0, "call_a", "", ""),
		oaiToolChunk(0, "", "lookup", `{}`),
		oaiFinishChunk("tool_calls"), "[DONE]",
	}}
	p := f.start(t, nil)
	var got []StreamEvent
	if _, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectEvents(&got))); err != nil {
		t.Fatal(err)
	}
	checkEvents(t, got, []StreamEvent{
		toolEv(StreamToolStart, "call_a", "lookup"),
		toolEv(StreamToolReady, "call_a", "lookup"),
	})
}

func TestOpenAIStreamNamelessToolCallSendsNoEvents(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{
		oaiToolChunk(0, "call_a", "", `{}`),
		oaiFinishChunk("tool_calls"), "[DONE]",
	}}
	p := f.start(t, nil)
	var got []StreamEvent
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectEvents(&got)))
	if err != nil {
		t.Fatal(err)
	}
	checkEvents(t, got, nil)
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "call_a" {
		t.Errorf("tool calls = %+v", resp.ToolCalls)
	}
}

func TestOpenAIStreamTextThenToolEvents(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{
		oaiTextChunk("Let me "), oaiTextChunk("check."),
		oaiToolChunk(0, "call_a", "lookup", `{}`),
		oaiFinishChunk("tool_calls"), "[DONE]",
	}}
	p := f.start(t, nil)
	var got []StreamEvent
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectEvents(&got)))
	if err != nil {
		t.Fatal(err)
	}
	checkEvents(t, got, []StreamEvent{
		{Kind: StreamText, Text: "Let me "},
		{Kind: StreamText, Text: "check."},
		toolEv(StreamToolStart, "call_a", "lookup"),
		toolEv(StreamToolReady, "call_a", "lookup"),
	})
	if resp.Text != "Let me check." {
		t.Errorf("text = %q", resp.Text)
	}
}

func TestOpenAIStreamCancelledFromToolStart(t *testing.T) {
	f := &oaiStreamFake{chunks: []string{
		oaiToolChunk(0, "call_a", "lookup", ""),
		oaiToolChunk(0, "", "", `{}`),
		oaiFinishChunk("tool_calls"), "[DONE]",
	}, hold: true}
	p := f.start(t, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var got []StreamEvent
	record := collectEvents(&got)
	_, err := p.complete(ctx, "gpt-4o", 100, streamReq(func(ev StreamEvent) { record(ev); cancel() }))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	checkEvents(t, got, []StreamEvent{toolEv(StreamToolStart, "call_a", "lookup")})
}
