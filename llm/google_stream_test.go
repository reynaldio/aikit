package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// gemStreamFake is a fake :streamGenerateContent endpoint. The first len(fails)
// requests get those statuses; later ones stream events as SSE, flushing after each.
// With hold set it then goes quiet until the client leaves.
type gemStreamFake struct {
	mu     sync.Mutex // guards paths, alts and bodies
	fails  []int
	events []string // data payloads
	hold   bool
	paths  []string
	alts   []string
	bodies []map[string]any
}

// seen returns copies of the recorded request paths and alt values, and the request count.
func (f *gemStreamFake) seen() (paths, alts []string, n int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.paths...), append([]string(nil), f.alts...), len(f.bodies)
}

func (f *gemStreamFake) start(t *testing.T) *googleProvider {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		f.bodies = append(f.bodies, body)
		f.paths = append(f.paths, r.URL.Path)
		f.alts = append(f.alts, r.URL.Query().Get("alt"))
		if n := len(f.bodies); n <= len(f.fails) {
			w.WriteHeader(f.fails[n-1])
			_, _ = w.Write([]byte(`{"error":{"message":"unknown field thinkingLevel"}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range f.events {
			_, _ = w.Write([]byte("data: " + e + "\n\n"))
			w.(http.Flusher).Flush()
		}
		if f.hold {
			<-r.Context().Done()
		}
	}))
	t.Cleanup(srv.Close)
	return &googleProvider{apiKey: "k", baseURL: srv.URL, http: srv.Client()}
}

func gemText(s string) string {
	b, _ := json.Marshal(s)
	return `{"candidates":[{"content":{"parts":[{"text":` + string(b) + `}]}}]}`
}

func gemFinish(text, fr string) string {
	b, _ := json.Marshal(text)
	return `{"candidates":[{"finishReason":"` + fr + `","content":{"parts":[{"text":` + string(b) + `}]}}]}`
}

const gemUsage = `"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":20,"cachedContentTokenCount":30,"thoughtsTokenCount":5}`

func TestGeminiStreamRequestShape(t *testing.T) {
	f := &gemStreamFake{events: []string{gemFinish("a", "STOP")}}
	g := f.start(t)
	var got []string
	if _, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(collectText(&got))); err != nil {
		t.Fatal(err)
	}
	paths, alts, _ := f.seen()
	if !strings.HasSuffix(paths[0], ":streamGenerateContent") || alts[0] != "sse" {
		t.Errorf("path = %q alt = %q", paths[0], alts[0])
	}
	// Without OnEvent the plain endpoint is used, as before.
	// The fake answers with SSE, which the plain decoder rejects; only the
	// request it made is under test here.
	_, _ = g.complete(context.Background(), "gemini-3.1-pro", 100, Request{Messages: userMsg("hi")})
	paths, alts, _ = f.seen()
	if !strings.HasSuffix(paths[1], ":generateContent") || alts[1] != "" {
		t.Errorf("plain path = %q alt = %q", paths[1], alts[1])
	}
}

func TestGeminiStreamTextInOrder(t *testing.T) {
	f := &gemStreamFake{events: []string{gemText("Hel"), gemText("lo "), gemFinish("world", "STOP")}}
	g := f.start(t)
	var got []string
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(collectText(&got)))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "Hel|lo |world" || resp.Text != "Hello world" || resp.StopReason != StopEndTurn {
		t.Errorf("events = %q resp = %+v", got, resp)
	}
}

func TestGeminiStreamSkipsThoughts(t *testing.T) {
	f := &gemStreamFake{events: []string{
		`{"candidates":[{"content":{"parts":[{"text":"pondering","thought":true}]}}]}`,
		`{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"answer"}]}}],` + gemUsage + `}`,
	}}
	g := f.start(t)
	var got []string
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(collectText(&got)))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0] != "answer" || resp.Text != "answer" {
		t.Errorf("events = %q resp.Text = %q", got, resp.Text)
	}
	if resp.OutputTokens != 25 {
		t.Errorf("OutputTokens = %d, want 25 (thoughts counted)", resp.OutputTokens)
	}
}

func TestGeminiStreamFunctionCalls(t *testing.T) {
	f := &gemStreamFake{events: []string{
		`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{"q":"x"}}}]}}]}`,
		`{"candidates":[{"finishReason":"STOP","content":{"parts":[{"functionCall":{"name":"ping"}}]}}]}`,
	}}
	g := f.start(t)
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(func(StreamEvent) {}))
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ToolCalls) != 2 || resp.StopReason != StopToolUse {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.ToolCalls[0].Name != "lookup" || string(resp.ToolCalls[0].Input) != `{"q":"x"}` {
		t.Errorf("call 0 = %+v", resp.ToolCalls[0])
	}
	if resp.ToolCalls[1].Name != "ping" || string(resp.ToolCalls[1].Input) != `{}` {
		t.Errorf("call 1 = %+v", resp.ToolCalls[1])
	}
}

func TestGeminiStreamUsageFromLastEventWithCounts(t *testing.T) {
	f := &gemStreamFake{events: []string{
		`{"candidates":[{"content":{"parts":[{"text":"a"}]}}],"usageMetadata":{"promptTokenCount":90,"candidatesTokenCount":1}}`,
		`{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"b"}]}}],` + gemUsage + `}`,
		`{"usageMetadata":{}}`,
	}}
	g := f.start(t)
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(func(StreamEvent) {}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.InputTokens != 70 || resp.OutputTokens != 25 || resp.CachedTokens != 30 {
		t.Errorf("usage = in %d out %d cached %d", resp.InputTokens, resp.OutputTokens, resp.CachedTokens)
	}
}

func TestGeminiStreamEndsWithoutFinishReason(t *testing.T) {
	f := &gemStreamFake{events: []string{gemText("partial")}}
	g := f.start(t)
	_, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(func(StreamEvent) {}))
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("err = %v, want io.ErrUnexpectedEOF", err)
	}
}

func TestGeminiStreamPromptBlocked(t *testing.T) {
	f := &gemStreamFake{events: []string{`{"promptFeedback":{"blockReason":"SAFETY"}}`}}
	g := f.start(t)
	var got []string
	_, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(collectText(&got)))
	var re *RefusalError
	if !errors.As(err, &re) || re.Category != "SAFETY" {
		t.Fatalf("err = %v, want SAFETY refusal", err)
	}
	if len(got) != 0 {
		t.Errorf("events = %q, want none", got)
	}
}

func TestGeminiStreamRefusalMidway(t *testing.T) {
	f := &gemStreamFake{events: []string{gemText("Once "), gemFinish("upon", "SAFETY")}}
	g := f.start(t)
	var got []string
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(collectText(&got)))
	var re *RefusalError
	if !errors.As(err, &re) || re.Category != "SAFETY" {
		t.Fatalf("err = %v, want SAFETY refusal", err)
	}
	if resp.Text != "Once upon" || strings.Join(got, "") != resp.Text {
		t.Errorf("resp.Text = %q events = %q", resp.Text, got)
	}
}

func TestGeminiStreamFlash400RetryStillWorks(t *testing.T) {
	f := &gemStreamFake{fails: []int{400}, events: []string{gemFinish("ok", "STOP")}}
	g := f.start(t)
	var got []string
	resp, err := g.complete(context.Background(), "gemini-3.8-flash", 100, streamReq(collectText(&got)))
	if err != nil {
		t.Fatal(err)
	}
	_, _, nReq := f.seen()
	if nReq != 2 || resp.Text != "ok" || len(got) != 1 {
		t.Errorf("requests = %d resp = %+v events = %q", nReq, resp, got)
	}
}

func TestGeminiStreamErrorEvent(t *testing.T) {
	f := &gemStreamFake{events: []string{gemText("a"), `{"error":{"message":"overloaded"}}`}}
	g := f.start(t)
	_, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(func(StreamEvent) {}))
	if err == nil || !strings.Contains(err.Error(), "overloaded") {
		t.Fatalf("err = %v, want overloaded", err)
	}
}

func TestGeminiStreamQuietPastDeadline(t *testing.T) {
	f := &gemStreamFake{events: []string{gemText("first")}, hold: true}
	g := f.start(t)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	_, err := g.complete(ctx, "gemini-3.1-pro", 100, streamReq(func(StreamEvent) {}))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
}

func TestGeminiStreamCancelledFromOnEvent(t *testing.T) {
	f := &gemStreamFake{events: []string{gemText("first"), gemText("second")}, hold: true}
	g := f.start(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var got []string
	record := collectText(&got)
	_, err := g.complete(ctx, "gemini-3.1-pro", 100, streamReq(func(ev StreamEvent) { record(ev); cancel() }))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(got) != 1 || got[0] != "first" {
		t.Errorf("events = %q, want exactly one", got)
	}
}

func TestGeminiStreamSkipsEmptyPayload(t *testing.T) {
	f := &gemStreamFake{events: []string{gemText("a"), "", gemFinish("b", "STOP")}}
	g := f.start(t)
	var got []string
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(collectText(&got)))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "a|b" || resp.Text != "ab" {
		t.Errorf("events = %q, resp = %+v", got, resp)
	}
}

// The plain (non-streamed) path shares geminiBuildResponse with the stream, so it
// drops thought parts from Text too.
func TestGeminiPlainResponseDropsThoughts(t *testing.T) {
	var out geminiResponse
	if err := json.Unmarshal([]byte(`{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"pondering","thought":true},{"text":"answer"}]}}]}`), &out); err != nil {
		t.Fatal(err)
	}
	resp, err := geminiBuildResponse("gemini-3.1-pro", out)
	if err != nil || resp.Text != "answer" {
		t.Fatalf("resp = %+v, err = %v", resp, err)
	}
}

func gemCall(name, args, fr string) string {
	frPart := ""
	if fr != "" {
		frPart = `"finishReason":"` + fr + `",`
	}
	return `{"candidates":[{` + frPart + `"content":{"parts":[{"functionCall":{"name":"` + name + `","args":` + args + `}}]}}]}`
}

func recordAll(got *[]StreamEvent) func(StreamEvent) {
	return func(ev StreamEvent) { *got = append(*got, ev) }
}

func TestGeminiStreamToolEventsOneCall(t *testing.T) {
	f := &gemStreamFake{events: []string{gemCall("search_docs", `{"q":"x"}`, "STOP")}}
	g := f.start(t)
	var got []StreamEvent
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(recordAll(&got)))
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID == "" {
		t.Fatalf("resp = %+v", resp)
	}
	if string(resp.ToolCalls[0].Input) != `{"q":"x"}` {
		t.Errorf("input = %s", resp.ToolCalls[0].Input)
	}
	id := resp.ToolCalls[0].ID
	checkEvents(t, got, []StreamEvent{
		toolEv(StreamToolStart, id, "search_docs"),
		toolEv(StreamToolReady, id, "search_docs"),
	})
}

func TestGeminiStreamToolEventsTwoCalls(t *testing.T) {
	f := &gemStreamFake{events: []string{gemCall("a", `{}`, ""), gemCall("b", `{}`, "STOP")}}
	g := f.start(t)
	var got []StreamEvent
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(recordAll(&got)))
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ToolCalls) != 2 || resp.ToolCalls[0].ID == resp.ToolCalls[1].ID {
		t.Fatalf("resp = %+v", resp)
	}
	if string(resp.ToolCalls[0].Input) != `{}` || string(resp.ToolCalls[1].Input) != `{}` {
		t.Errorf("inputs = %s, %s", resp.ToolCalls[0].Input, resp.ToolCalls[1].Input)
	}
	id1, id2 := resp.ToolCalls[0].ID, resp.ToolCalls[1].ID
	checkEvents(t, got, []StreamEvent{
		toolEv(StreamToolStart, id1, "a"),
		toolEv(StreamToolReady, id1, "a"),
		toolEv(StreamToolStart, id2, "b"),
		toolEv(StreamToolReady, id2, "b"),
	})
}

func TestGeminiStreamToolEventsAfterText(t *testing.T) {
	f := &gemStreamFake{events: []string{gemText("Let me "), gemText("look."), gemCall("lookup", `{}`, "STOP")}}
	g := f.start(t)
	var got []StreamEvent
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(recordAll(&got)))
	if err != nil {
		t.Fatal(err)
	}
	if string(resp.ToolCalls[0].Input) != `{}` {
		t.Errorf("input = %s", resp.ToolCalls[0].Input)
	}
	id := resp.ToolCalls[0].ID
	checkEvents(t, got, []StreamEvent{
		{Kind: StreamText, Text: "Let me "},
		{Kind: StreamText, Text: "look."},
		toolEv(StreamToolStart, id, "lookup"),
		toolEv(StreamToolReady, id, "lookup"),
	})
	if resp.Text != "Let me look." {
		t.Errorf("resp.Text = %q", resp.Text)
	}
}

func TestGeminiStreamNoToolEventsForSearchGrounding(t *testing.T) {
	f := &gemStreamFake{events: []string{
		`{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"answer"}]},"groundingMetadata":{"webSearchQueries":["q"]}}]}`,
	}}
	g := f.start(t)
	var got []StreamEvent
	if _, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(recordAll(&got))); err != nil {
		t.Fatal(err)
	}
	checkEvents(t, got, []StreamEvent{{Kind: StreamText, Text: "answer"}})
}

func TestGeminiStreamCancelledFromToolStart(t *testing.T) {
	f := &gemStreamFake{events: []string{gemCall("a", `{}`, ""), gemCall("b", `{}`, "STOP")}, hold: true}
	g := f.start(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var got []StreamEvent
	record := recordAll(&got)
	_, err := g.complete(ctx, "gemini-3.1-pro", 100, streamReq(func(ev StreamEvent) { record(ev); cancel() }))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(got) != 1 || got[0].Kind != StreamToolStart {
		t.Errorf("events = %+v, want exactly the start event", got)
	}
}

func TestGeminiStreamCancelledFromToolReady(t *testing.T) {
	f := &gemStreamFake{events: []string{gemCall("a", `{}`, ""), gemCall("b", `{}`, "STOP")}, hold: true}
	g := f.start(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var got []StreamEvent
	record := recordAll(&got)
	_, err := g.complete(ctx, "gemini-3.1-pro", 100, streamReq(func(ev StreamEvent) {
		record(ev)
		if ev.Kind == StreamToolReady {
			cancel()
		}
	}))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(got) != 2 || got[0].Kind != StreamToolStart || got[1].Kind != StreamToolReady {
		t.Errorf("events = %+v, want start then ready only", got)
	}
}

func TestGeminiStreamRefusalAfterCallKeepsEventIDs(t *testing.T) {
	f := &gemStreamFake{events: []string{gemCall("a", `{"q":1}`, "SAFETY")}}
	g := f.start(t)
	var got []StreamEvent
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(recordAll(&got)))
	var re *RefusalError
	if !errors.As(err, &re) || re.Category != "SAFETY" {
		t.Fatalf("err = %v, want SAFETY refusal", err)
	}
	if len(resp.ToolCalls) != 1 || len(got) != 2 {
		t.Fatalf("resp = %+v events = %+v", resp, got)
	}
	if resp.ToolCalls[0].ID == "" || resp.ToolCalls[0].ID != got[0].ToolCallID {
		t.Errorf("call ID = %q, event ID = %q", resp.ToolCalls[0].ID, got[0].ToolCallID)
	}
}

// --- partial usage on an early end ---

const gemUsagePartial = `"usageMetadata":{"promptTokenCount":50,"cachedContentTokenCount":10,"candidatesTokenCount":7,"thoughtsTokenCount":3}`

// cancelOn returns an OnEvent that cancels ctx on the n-th text event.
func cancelOn(n int, cancel context.CancelFunc) func(StreamEvent) {
	seen := 0
	return func(ev StreamEvent) {
		if ev.Kind != StreamText {
			return
		}
		seen++
		if seen >= n {
			cancel()
		}
	}
}

func TestGeminiPartialRealUsage(t *testing.T) {
	f := &gemStreamFake{events: []string{
		`{"candidates":[{"content":{"parts":[{"text":"hello"}]}}],` + gemUsagePartial + `}`,
		gemText("never read"),
	}, hold: true}
	g := f.start(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resp, err := g.complete(ctx, "gemini-3.1-pro", 100, streamReq(cancelOn(1, cancel)))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if resp.Text != "hello" || resp.InputTokens != 40 || resp.CachedTokens != 10 || resp.OutputTokens != 10 || resp.UsageEstimated {
		t.Errorf("resp = %+v", resp)
	}
	if resp.ToolCalls != nil || resp.StopReason != "" {
		t.Errorf("partial reply has tool calls or stop reason: %+v", resp)
	}
}

func TestGeminiPartialEstimatedUsage(t *testing.T) {
	f := &gemStreamFake{events: []string{gemText("Hello "), gemText("there, friend"), gemText("never read")}, hold: true}
	g := f.start(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := streamReq(cancelOn(2, cancel))
	resp, err := g.complete(ctx, "gemini-3.1-pro", 100, req)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if resp.Text != "Hello there, friend" || !resp.UsageEstimated {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.InputTokens != estimateRequestTokens(req) || resp.OutputTokens != estimateTokens("Hello there, friend") {
		t.Errorf("in %d out %d, want %d / %d", resp.InputTokens, resp.OutputTokens, estimateRequestTokens(req), estimateTokens("Hello there, friend"))
	}
}

func TestGeminiPartialEndWithoutFinishReason(t *testing.T) {
	f := &gemStreamFake{events: []string{
		`{"usageMetadata":{"promptTokenCount":80,"cachedContentTokenCount":20}}`,
		`{"candidates":[{"content":{"parts":[{"text":"musing","thought":true}]}}]}`,
		gemText("answer"),
	}}
	g := f.start(t)
	var got []string
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(collectText(&got)))
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("err = %v, want io.ErrUnexpectedEOF", err)
	}
	if resp.Text != "answer" || resp.InputTokens != 60 || resp.CachedTokens != 20 {
		t.Errorf("resp = %+v", resp)
	}
	if resp.OutputTokens != estimateTokens("answermusing") || !resp.UsageEstimated {
		t.Errorf("OutputTokens = %d estimated = %v, want guess over text + thought", resp.OutputTokens, resp.UsageEstimated)
	}
}

func TestGeminiPartialGuessCountsCallArgs(t *testing.T) {
	f := &gemStreamFake{events: []string{gemCall("lookup", `{"q":"abcdefgh"}`, "")}}
	g := f.start(t)
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(func(StreamEvent) {}))
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("err = %v, want io.ErrUnexpectedEOF", err)
	}
	if resp.OutputTokens != estimateTokens(`{"q":"abcdefgh"}`) || resp.ToolCalls != nil {
		t.Errorf("resp = %+v", resp)
	}
}

func TestGeminiPartialZeroBeforeAnyEvent(t *testing.T) {
	f := &gemStreamFake{hold: true}
	g := f.start(t)
	ctx, cancel := context.WithCancel(context.Background())
	// Cancel before the request; the first read fails with no payload seen.
	cancel()
	resp, err := g.complete(ctx, "gemini-3.1-pro", 100, streamReq(func(StreamEvent) {}))
	if err == nil {
		t.Fatal("want an error")
	}
	if !reflect.DeepEqual(resp, Response{}) {
		t.Errorf("resp = %+v, want zero", resp)
	}
}

func TestGeminiPartialZeroWhenFirstPayloadIsError(t *testing.T) {
	f := &gemStreamFake{events: []string{`{"error":{"message":"overloaded"},` + gemUsagePartial + `}`}}
	g := f.start(t)
	resp, err := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(func(StreamEvent) {}))
	if err == nil || !strings.Contains(err.Error(), "overloaded") {
		t.Fatalf("err = %v, want overloaded", err)
	}
	if !reflect.DeepEqual(resp, Response{}) {
		t.Errorf("resp = %+v, want zero", resp)
	}
}

func TestGeminiPartialAfterEarlierEventThenError(t *testing.T) {
	f := &gemStreamFake{events: []string{gemText("abcd"), `{"error":{"message":"overloaded"}}`}}
	g := f.start(t)
	resp, _ := g.complete(context.Background(), "gemini-3.1-pro", 100, streamReq(func(StreamEvent) {}))
	if resp.Text != "abcd" || resp.OutputTokens != 1 || !resp.UsageEstimated {
		t.Errorf("resp = %+v", resp)
	}
}

func TestGeminiPartialThroughFlashThinkingLoop(t *testing.T) {
	f := &gemStreamFake{events: []string{
		`{"candidates":[{"content":{"parts":[{"text":"hi"}]}}],` + gemUsagePartial + `}`,
	}, hold: true}
	g := f.start(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resp, err := g.complete(ctx, "gemini-3.8-flash", 100, streamReq(cancelOn(1, cancel)))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if resp.Text != "hi" || resp.InputTokens != 40 || resp.OutputTokens != 10 {
		t.Errorf("resp = %+v", resp)
	}
	if _, _, n := f.seen(); n != 1 {
		t.Errorf("requests = %d, want 1", n)
	}
}
