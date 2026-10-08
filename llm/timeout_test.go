package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// forever stands in for a server that never answers within the test.
const forever = time.Hour

// slowServer replies with body after delay. With partial set it sends the
// headers and the first half of body at once, then stalls for delay before the
// rest — a reply cut off mid-body. Handlers give up when the client does.
func slowServer(t *testing.T, delay time.Duration, body string, partial bool) string {
	t.Helper()
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if partial {
			_, _ = w.Write([]byte(body[:len(body)/2]))
			w.(http.Flusher).Flush()
			body = body[len(body)/2:]
		}
		select {
		case <-time.After(delay):
			_, _ = w.Write([]byte(body))
		case <-r.Context().Done():
		case <-done:
		}
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(done) }) // runs first: frees stalled handlers so Close returns
	return srv.URL
}

// timeoutTarget is one provider under test, built against a fake server with the
// production HTTP client and the given deadline-less default.
type timeoutTarget struct {
	name  string
	reply string
	build func(url string, timeout time.Duration) provider
	model string
}

var timeoutTargets = []timeoutTarget{
	{"google", geminiJSONReply, func(url string, timeout time.Duration) provider {
		g := newGoogle("k", timeout).(*googleProvider)
		g.baseURL = url
		return g
	}, "gemini-3.1-pro-preview"}, // not flash: one attempt, no thinking-config retries
	{"openai", openaiJSONReply, func(url string, timeout time.Duration) provider {
		o := newOpenAI("k", "", timeout).(*openaiProvider)
		o.baseURL = url // strictSchema stays as for the real OpenAI endpoint
		return o
	}, "gpt-5.6-luna"},
	{"openai-compatible", openaiJSONReply, func(url string, timeout time.Duration) provider {
		return newOpenAI("k", url, timeout) // a custom OpenAIBaseURL
	}, "deepseek-chat"},
}

func TestCallerDeadlineOutlastsDefault(t *testing.T) {
	// The reply takes 150ms; the deadline-less default is 50ms. The caller's 1s
	// deadline is what counts, so the call succeeds — v0.6.0's fixed client
	// Timeout, scaled the same way, would have cut it off.
	for _, tt := range timeoutTargets {
		p := tt.build(slowServer(t, 150*time.Millisecond, tt.reply, false), 50*time.Millisecond)
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		_, err := p.complete(ctx, tt.model, 100, Request{Messages: userMsg("x")})
		cancel()
		if err != nil {
			t.Errorf("%s: %v", tt.name, err)
		}
	}
}

func TestCallerDeadlineBounds(t *testing.T) {
	for _, tt := range timeoutTargets {
		p := tt.build(slowServer(t, 2*time.Second, tt.reply, false), forever)
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		start := time.Now()
		_, err := p.complete(ctx, tt.model, 100, Request{Messages: userMsg("x")})
		cancel()
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("%s: err = %v, want context.DeadlineExceeded", tt.name, err)
		}
		if el := time.Since(start); el > time.Second {
			t.Errorf("%s: took %v, want ~200ms", tt.name, el)
		}
	}
}

func TestNoDeadlineGetsDefault(t *testing.T) {
	for _, tt := range timeoutTargets {
		p := tt.build(slowServer(t, forever, tt.reply, false), 300*time.Millisecond)
		start := time.Now()
		_, err := p.complete(context.Background(), tt.model, 100, Request{Messages: userMsg("x")})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("%s: err = %v, want context.DeadlineExceeded", tt.name, err)
		}
		if el := time.Since(start); el > 2*time.Second {
			t.Errorf("%s: took %v, want ~300ms", tt.name, el)
		}
	}
}

func TestCancelAbortsRequest(t *testing.T) {
	for _, tt := range timeoutTargets {
		p := tt.build(slowServer(t, forever, tt.reply, false), forever)
		ctx, cancel := context.WithCancel(context.Background())
		time.AfterFunc(100*time.Millisecond, cancel)
		_, err := p.complete(ctx, tt.model, 100, Request{Messages: userMsg("x")})
		cancel()
		if !errors.Is(err, context.Canceled) {
			t.Errorf("%s: err = %v, want context.Canceled", tt.name, err)
		}
	}
}

func TestDeadlineMidBodyIsAContextError(t *testing.T) {
	// The headers and half the JSON arrive, then the reply stalls past the
	// deadline. That must read as the deadline, not as a garbled reply, or a
	// caller's timeout classification misses it.
	for _, tt := range timeoutTargets {
		p := tt.build(slowServer(t, forever, tt.reply, true), forever)
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		_, err := p.complete(ctx, tt.model, 100, Request{Messages: userMsg("x")})
		cancel()
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("%s: err = %v, want context.DeadlineExceeded", tt.name, err)
		}
	}
}

func TestDefaultRequestTimeoutScales(t *testing.T) {
	cases := []struct {
		override  time.Duration
		maxTokens int
		want      time.Duration
	}{
		{0, 1024, 120 * time.Second}, // small requests behave as before
		{0, 24000, 11*time.Minute + 15*time.Second},
		{0, 48000, 22*time.Minute + 30*time.Second},
		{0, 128000, time.Hour},
		{5 * time.Second, 128000, 5 * time.Second}, // Config.RequestTimeout wins
	}
	for _, c := range cases {
		if got := defaultRequestTimeout(c.override, c.maxTokens); got != c.want {
			t.Errorf("defaultRequestTimeout(%v, %d) = %v, want %v", c.override, c.maxTokens, got, c.want)
		}
	}
}

func TestProviderHTTPClientHasNoFixedCap(t *testing.T) {
	for name, hc := range map[string]*http.Client{
		"google": newGoogle("k", 0).(*googleProvider).http,
		"openai": newOpenAI("k", "", 0).(*openaiProvider).http,
	} {
		if hc.Timeout != 0 {
			t.Errorf("%s: http.Client.Timeout = %v, want none", name, hc.Timeout)
		}
		if tr, ok := hc.Transport.(*http.Transport); !ok || tr.ResponseHeaderTimeout != 0 {
			t.Errorf("%s: a ResponseHeaderTimeout would cap non-streaming replies", name)
		}
	}
}
