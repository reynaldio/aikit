package llm

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go/option"
)

const geminiOverloaded = `{"error":{"code":503,"message":"The model is overloaded. Please try again later.","status":"UNAVAILABLE"}}`

const oaiPlainReply = `{"choices":[{"index":0,"message":{"role":"assistant","content":"fallback ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`

// attemptServer counts requests and hands each one to h with its 0-based attempt number.
func attemptServer(t *testing.T, h func(w http.ResponseWriter, n int)) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		h(w, int(n.Add(1))-1)
	}))
	t.Cleanup(srv.Close)
	return srv, &n
}

func TestGeminiRetries503ThenSucceeds(t *testing.T) {
	srv, n := attemptServer(t, func(w http.ResponseWriter, i int) {
		if i == 0 {
			w.Header().Set("retry-after-ms", "1")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(geminiOverloaded))
			return
		}
		_, _ = w.Write([]byte(geminiJSONReply))
	})
	g := &googleProvider{apiKey: "k", baseURL: srv.URL, http: newProviderHTTPClient(nil)}
	resp, err := g.complete(context.Background(), "gemini-3.1-pro-preview", 100, Request{Messages: userMsg("read")})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(resp.Text, "NATIONAL_ID") {
		t.Errorf("Text = %q, want the second reply", resp.Text)
	}
	if got := n.Load(); got != 2 {
		t.Errorf("attempts = %d, want 2", got)
	}
}

func TestOpenAIStreamRetries503ThenStreams(t *testing.T) {
	srv, n := attemptServer(t, func(w http.ResponseWriter, i int) {
		if i == 0 {
			w.Header().Set("retry-after-ms", "1")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":{"message":"overloaded"}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range []string{oaiTextChunk("Hel"), oaiTextChunk("lo"), oaiFinishChunk("stop"), oaiUsageChunk, "[DONE]"} {
			fmt.Fprintf(w, "data: %s\n\n", c)
		}
	})
	p := newOpenAI("k", srv.URL, 0, nil)
	var got []string
	resp, err := p.complete(context.Background(), "gpt-4o", 100, streamReq(collectText(&got)))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != "Hel" || got[1] != "lo" {
		t.Errorf("events = %v, want [Hel lo]", got)
	}
	if strings.Join(got, "") != resp.Text {
		t.Errorf("events %q != resp.Text %q", strings.Join(got, ""), resp.Text)
	}
	if c := n.Load(); c != 2 {
		t.Errorf("attempts = %d, want 2", c)
	}
}

func TestAnthropicSDKRetriesAreOff(t *testing.T) {
	srv, n := attemptServer(t, func(w http.ResponseWriter, _ int) {
		w.Header().Set("retry-after-ms", "1")
		w.WriteHeader(529)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`))
	})
	p := newAnthropic("k", nil, option.WithBaseURL(srv.URL))
	_, err := p.complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("hi")})
	if err == nil {
		t.Fatal("want an error")
	}
	if got := n.Load(); got != 3 {
		t.Errorf("attempts = %d, want exactly 3 (SDK retries would make it 9)", got)
	}
	if !shouldFailover(err) {
		t.Errorf("shouldFailover(%v) = false, want true", err)
	}
}

// failoverRouter wires a Google primary (handled by primary) to an OpenAI fallback
// that answers one plain 200, both on the real shared client.
func failoverRouter(t *testing.T, primary func(w http.ResponseWriter, n int)) (r *router, primaryN, fallbackN *atomic.Int32) {
	t.Helper()
	gsrv, gn := attemptServer(t, primary)
	osrv, on := attemptServer(t, func(w http.ResponseWriter, _ int) {
		_, _ = w.Write([]byte(oaiPlainReply))
	})
	g := &googleProvider{apiKey: "k", baseURL: gsrv.URL, http: newProviderHTTPClient(nil)}
	o := newOpenAI("k", osrv.URL, 0, nil).(*openaiProvider)
	r = &router{
		providers: map[Provider]provider{ProviderGoogle: g, ProviderOpenAI: o},
		profiles:  map[Profile]ModelRef{ProfileChat: {Provider: ProviderGoogle, Model: "flash"}},
		fallbacks: map[Profile]ModelRef{ProfileChat: {Provider: ProviderOpenAI, Model: "gpt-4o"}},
		maxTokens: 100,
		log:       slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	return r, gn, on
}

func TestRouterFailsOverAfterRetries(t *testing.T) {
	r, gn, on := failoverRouter(t, func(w http.ResponseWriter, _ int) {
		w.Header().Set("retry-after-ms", "1")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(geminiOverloaded))
	})
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat, Messages: userMsg("hi")})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Provider != ProviderOpenAI {
		t.Errorf("Provider = %q, want openai", resp.Provider)
	}
	if gn.Load() != 3 || on.Load() != 1 {
		t.Errorf("attempts primary=%d fallback=%d, want 3 and 1", gn.Load(), on.Load())
	}
}

func TestRouterFailsOverAtOnceOnLongRetryAfter(t *testing.T) {
	r, gn, on := failoverRouter(t, func(w http.ResponseWriter, _ int) {
		w.Header().Set("Retry-After", "60")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}`))
	})
	start := time.Now()
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat, Messages: userMsg("hi")})
	if err != nil {
		t.Fatal(err)
	}
	if time.Since(start) >= 2*time.Second {
		t.Errorf("took %v, want a fail-over without waiting", time.Since(start))
	}
	if resp.Provider != ProviderOpenAI {
		t.Errorf("Provider = %q, want openai", resp.Provider)
	}
	if gn.Load() != 1 || on.Load() != 1 {
		t.Errorf("attempts primary=%d fallback=%d, want 1 and 1", gn.Load(), on.Load())
	}
}
