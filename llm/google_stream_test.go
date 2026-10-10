package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// gemStreamFake is a fake :streamGenerateContent endpoint. The first len(fails)
// requests get those statuses; later ones stream events as SSE, flushing after each.
// With hold set it then goes quiet until the client leaves.
type gemStreamFake struct {
	fails  []int
	events []string // data payloads
	hold   bool
	paths  []string
	alts   []string
	bodies []map[string]any
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
	if !strings.HasSuffix(f.paths[0], ":streamGenerateContent") || f.alts[0] != "sse" {
		t.Errorf("path = %q alt = %q", f.paths[0], f.alts[0])
	}
	// Without OnEvent the plain endpoint is used, as before.
	if _, err := g.complete(context.Background(), "gemini-3.1-pro", 100, Request{Messages: userMsg("hi")}); err == nil {
		t.Log("plain call unexpectedly decoded an SSE body")
	}
	if !strings.HasSuffix(f.paths[1], ":generateContent") || f.alts[1] != "" {
		t.Errorf("plain path = %q alt = %q", f.paths[1], f.alts[1])
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
	if len(f.bodies) != 2 || resp.Text != "ok" || len(got) != 1 {
		t.Errorf("requests = %d resp = %+v events = %q", len(f.bodies), resp, got)
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
