package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/option"
)

// errBody is a provider-shaped error body.
func errBody(p Provider, status int) string {
	switch p {
	case ProviderAnthropic:
		return `{"type":"error","error":{"type":"api_error","message":"failed"}}`
	case ProviderGoogle:
		return fmt.Sprintf(`{"error":{"code":%d,"message":"failed"}}`, status)
	}
	return `{"error":{"message":"failed"}}`
}

// replyServer answers every request with h. The body is drained first.
func replyServer(t *testing.T, h func(w http.ResponseWriter)) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		h(w)
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

// statusServer answers every request with status and body. retry-after-ms keeps
// the transport's retries of a 429/5xx quick.
func statusServer(t *testing.T, status int, body string) string {
	return replyServer(t, func(w http.ResponseWriter) {
		w.Header().Set("retry-after-ms", "1")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	})
}

// resetServer resets every connection without a response.
func resetServer(t *testing.T) string {
	return replyServer(t, func(w http.ResponseWriter) {
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		if tc, ok := conn.(*net.TCPConn); ok {
			_ = tc.SetLinger(0) // close sends RST
		}
		conn.Close()
	})
}

// streamServer answers every request with a 200 event stream holding body.
func streamServer(t *testing.T, body string) string {
	return replyServer(t, func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	})
}

func providerAt(p Provider, url string) provider {
	switch p {
	case ProviderAnthropic:
		return newAnthropic("k", nil, option.WithBaseURL(url))
	case ProviderGoogle:
		return &googleProvider{apiKey: "k", baseURL: url, http: newProviderHTTPClient(nil)}
	}
	return newOpenAI("k", url, 0, nil)
}

var errorProviders = []Provider{ProviderAnthropic, ProviderGoogle, ProviderOpenAI}

// testModel names a model that takes one attempt (no Gemini Flash knob walk).
var testModel = map[Provider]string{
	ProviderAnthropic: "claude-opus-5-5",
	ProviderGoogle:    "gemini-3.1-pro",
	ProviderOpenAI:    "gpt-4o",
}

func wantProviderError(t *testing.T, err error, p Provider, status int, retryable bool) *ProviderError {
	t.Helper()
	var pe *ProviderError
	if !errors.As(err, &pe) {
		t.Fatalf("errors.As(%T %v) found no *ProviderError", err, err)
	}
	if pe.Provider != p || pe.Status != status || pe.Retryable != retryable {
		t.Errorf("ProviderError = {%s %d %v}, want {%s %d %v}", pe.Provider, pe.Status, pe.Retryable, p, status, retryable)
	}
	if pe.Err == nil || pe.Error() != pe.Err.Error() {
		t.Errorf("Error() = %q, want the underlying message", pe.Error())
	}
	return pe
}

func TestProviderErrorStatus(t *testing.T) {
	cases := []struct {
		status    int
		retryable bool
		only      Provider // "" = every provider
	}{
		{400, false, ""},
		{401, false, ""},
		{404, false, ""},
		{408, true, ""},
		{409, true, ""},
		{429, true, ""},
		{500, true, ""},
		{503, true, ""},
		{529, true, ProviderAnthropic},
	}
	for _, p := range errorProviders {
		for _, c := range cases {
			if c.only != "" && c.only != p {
				continue
			}
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%d/stream=%v", p, c.status, stream), func(t *testing.T) {
					url := statusServer(t, c.status, errBody(p, c.status))
					req := Request{Messages: userMsg("hi")}
					if stream {
						req.OnEvent = func(StreamEvent) {}
					}
					_, err := providerAt(p, url).complete(context.Background(), testModel[p], 100, req)
					wantProviderError(t, err, p, c.status, c.retryable)
				})
			}
		}
	}
}

func TestProviderErrorConnectionReset(t *testing.T) {
	for _, p := range errorProviders {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", p, stream), func(t *testing.T) {
				req := Request{Messages: userMsg("hi")}
				if stream {
					req.OnEvent = func(StreamEvent) {}
				}
				// One attempt: the transport would retry a reset with real backoff.
				ctx := withNoRetry(context.Background())
				_, err := providerAt(p, resetServer(t)).complete(ctx, testModel[p], 100, req)
				wantProviderError(t, err, p, 0, true)
			})
		}
	}
}

// A body that is not JSON keeps its status: an HTML 503 from a proxy is still a 503.
func TestProviderErrorNonJSONBody(t *testing.T) {
	for _, p := range errorProviders {
		t.Run(string(p), func(t *testing.T) {
			url := statusServer(t, 502, "<html>Bad Gateway</html>")
			_, err := providerAt(p, url).complete(context.Background(), testModel[p], 100, Request{Messages: userMsg("hi")})
			wantProviderError(t, err, p, 502, true)
		})
	}
}

// A stream that ends early is a transport failure and keeps the usage so far.
func TestProviderErrorStreamCutOff(t *testing.T) {
	bodies := map[Provider]string{
		ProviderAnthropic: msgStart(`{"input_tokens":40,"output_tokens":1}`),
		ProviderGoogle:    "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}],\"usageMetadata\":{\"promptTokenCount\":40}}\n\n",
		ProviderOpenAI:    "data: " + oaiTextChunk("Hel") + "\n\n",
	}
	for _, p := range errorProviders {
		t.Run(string(p), func(t *testing.T) {
			resp, err := providerAt(p, streamServer(t, bodies[p])).complete(context.Background(), testModel[p], 100,
				Request{Messages: userMsg("hi"), OnEvent: func(StreamEvent) {}})
			wantProviderError(t, err, p, 0, true)
			if !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Errorf("errors.Is(err, io.ErrUnexpectedEOF) = false for %v", err)
			}
			if resp.InputTokens == 0 {
				t.Errorf("InputTokens = 0, want the partial usage kept")
			}
		})
	}
}

// An error event inside an accepted stream takes its status from the event.
func TestProviderErrorMidStreamEvent(t *testing.T) {
	cases := []struct {
		p         Provider
		body      string
		status    int
		retryable bool
	}{
		{ProviderAnthropic, msgStart(`{"input_tokens":40,"output_tokens":1}`) +
			sseEvent("error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`), 529, true},
		{ProviderAnthropic, msgStart(`{"input_tokens":40,"output_tokens":1}`) +
			sseEvent("error", `{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}`), 400, false},
		{ProviderGoogle, "data: {\"error\":{\"code\":503,\"message\":\"overloaded\"}}\n\n", 503, true},
		{ProviderGoogle, "data: {\"error\":{\"message\":\"failed\"}}\n\n", 0, true},
		{ProviderOpenAI, "data: {\"error\":{\"message\":\"failed\"}}\n\n", 0, true},
	}
	for _, c := range cases {
		t.Run(fmt.Sprintf("%s/%d", c.p, c.status), func(t *testing.T) {
			resp, err := providerAt(c.p, streamServer(t, c.body)).complete(context.Background(), testModel[c.p], 100,
				Request{Messages: userMsg("hi"), OnEvent: func(StreamEvent) {}})
			wantProviderError(t, err, c.p, c.status, c.retryable)
			if c.p == ProviderAnthropic && resp.InputTokens == 0 {
				t.Errorf("InputTokens = 0, want the billed usage kept")
			}
		})
	}
}

func TestProviderErrorNotForEndedContext(t *testing.T) {
	for _, p := range errorProviders {
		t.Run(string(p), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			_, err := providerAt(p, statusServer(t, 503, errBody(p, 503))).complete(ctx, testModel[p], 100, Request{Messages: userMsg("hi")})
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("err = %v, want context.Canceled", err)
			}
			var pe *ProviderError
			if errors.As(err, &pe) {
				t.Errorf("a canceled request is a *ProviderError: %v", pe)
			}
		})
	}
}

// The OpenAI-compatible endpoints report their own name.
func TestProviderErrorNamesCompatibleProvider(t *testing.T) {
	url := statusServer(t, 400, errBody(ProviderOpenAI, 400))
	r := New(Config{DeepSeekAPIKey: "k", DeepSeekBaseURL: url}).(*router)
	_, err := r.Complete(context.Background(), Request{
		Model: &ModelRef{Provider: ProviderDeepSeek, Model: "deepseek-chat"}, Messages: userMsg("hi"),
	})
	wantProviderError(t, err, ProviderDeepSeek, 400, false)
}

// errors.As finds the error through the router's streaming path and RunTools.
func TestProviderErrorThroughRouterAndToolLoop(t *testing.T) {
	url := statusServer(t, 400, errBody(ProviderGoogle, 400))
	r := &router{
		providers: map[Provider]provider{ProviderGoogle: providerAt(ProviderGoogle, url)},
		profiles:  map[Profile]ModelRef{ProfileChat: {Provider: ProviderGoogle, Model: "gemini-3.1-pro"}},
		maxTokens: 100,
		log:       slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	req := Request{Messages: userMsg("hi"), OnEvent: func(StreamEvent) {}}

	_, err := r.Complete(context.Background(), req)
	wantProviderError(t, err, ProviderGoogle, 400, false)

	_, err = RunTools(context.Background(), r, req, map[string]ToolFunc{}, ToolLoopOptions{})
	wantProviderError(t, err, ProviderGoogle, 400, false)

	req.OnEvent = nil
	_, err = RunTools(context.Background(), r, req, map[string]ToolFunc{}, ToolLoopOptions{})
	wantProviderError(t, err, ProviderGoogle, 400, false)
}

func TestProviderErrorEmbed(t *testing.T) {
	_, err := newTestGoogleEmbed(statusServer(t, 400, errBody(ProviderGoogle, 400))).embed(context.Background(), "gemini-embedding-001", []string{"a"}, "", 0)
	wantProviderError(t, err, ProviderGoogle, 400, false)
	_, err = newTestOpenAIEmbed(statusServer(t, 503, errBody(ProviderOpenAI, 503))).embed(context.Background(), "m", []string{"a"}, "", 0)
	wantProviderError(t, err, ProviderOpenAI, 503, true)
}

func TestProviderErrorBatchCalls(t *testing.T) {
	ctx := context.Background()

	ab := newAnthropicBatch("k", nil, option.WithBaseURL(statusServer(t, 404, `{"type":"error","error":{"type":"not_found_error","message":"no"}}`)))
	_, err := ab.status(ctx, "msgbatch_1")
	wantProviderError(t, err, ProviderAnthropic, 404, false)

	ob := newOpenAIBatch(ProviderMoonshot, "k", statusServer(t, 503, errBody(ProviderOpenAI, 503)), nil)
	_, err = ob.status(ctx, "batch_1")
	wantProviderError(t, err, ProviderMoonshot, 503, true)

	gb := newGoogleBatch("k", nil).(*googleBatch)
	_, err = gb.do(ctx, http.MethodGet, statusServer(t, 400, errBody(ProviderGoogle, 400)), nil, nil)
	wantProviderError(t, err, ProviderGoogle, 400, false)
}

func TestProviderErrorBatchItems(t *testing.T) {
	ctx := context.Background()

	// Anthropic: the item's error type gives the status.
	line := `{"custom_id":"a","result":{"type":"errored","error":{"type":"error","error":{"type":"overloaded_error","message":"busy"}}}}` + "\n"
	ab := newAnthropicBatch("k", nil, option.WithBaseURL(replyServer(t, func(w http.ResponseWriter) {
		_, _ = io.WriteString(w, line)
	})))
	n := 0
	for raw, err := range ab.results(ctx, batchChat, "claude-opus-5-5", "msgbatch_1") {
		if err != nil {
			t.Fatal(err)
		}
		n++
		wantProviderError(t, raw.err, ProviderAnthropic, 529, true)
	}
	if n != 1 {
		t.Fatalf("got %d results, want 1", n)
	}

	// OpenAI-compatible: the item's own HTTP status.
	ob := newOpenAIBatch(ProviderOpenAI, "k", "", nil).(*openaiBatch)
	var item oaiBatchLine
	if err := jsonUnmarshal(`{"custom_id":"a","response":{"status_code":400,"body":{"error":{"message":"bad"}}}}`, &item); err != nil {
		t.Fatal(err)
	}
	wantProviderError(t, ob.parseLine(batchChat, "gpt-4o", item).err, ProviderOpenAI, 400, false)

	// Gemini: a google.rpc code.
	gb := newGoogleBatch("k", nil).(*googleBatch)
	for code, want := range map[int]struct {
		status    int
		retryable bool
	}{3: {400, false}, 8: {429, true}, 14: {503, true}} {
		var g gbLine
		if err := jsonUnmarshal(fmt.Sprintf(`{"key":"a","error":{"code":%d,"message":"x"}}`, code), &g); err != nil {
			t.Fatal(err)
		}
		wantProviderError(t, gb.parseLine(batchChat, "gemini-3.1-pro", g).err, ProviderGoogle, want.status, want.retryable)
	}
}

func jsonUnmarshal(s string, v any) error { return json.Unmarshal([]byte(s), v) }
