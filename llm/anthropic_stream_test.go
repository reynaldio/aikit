package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicServer is a fake /v1/messages endpoint. It records each request body
// and answers with reply (raw JSON for a plain request) or, when the request asks
// for a stream, with the SSE body sse. If abort is set, the handler writes sse and
// then drops the connection without finishing the response.
type anthropicServer struct {
	bodies []map[string]any
	reply  string
	sse    string
	abort  bool
}

func (as *anthropicServer) start(t *testing.T) *anthropicProvider {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("request is not JSON: %v", err)
		}
		as.bodies = append(as.bodies, body)
		if body["stream"] != true {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(as.reply))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if as.abort {
			// Promise more bytes than we send, then hang up: the client sees an
			// abrupt mid-stream close rather than a clean end of stream.
			w.Header().Set("Content-Length", fmt.Sprint(len(as.sse)+4096))
			_, _ = w.Write([]byte(as.sse))
			w.(http.Flusher).Flush()
			conn, _, err := w.(http.Hijacker).Hijack()
			if err == nil {
				_ = conn.Close()
			}
			return
		}
		_, _ = w.Write([]byte(as.sse))
	}))
	t.Cleanup(srv.Close)
	return &anthropicProvider{client: anthropic.NewClient(
		option.WithAPIKey("k"),
		option.WithBaseURL(srv.URL),
		option.WithMaxRetries(0),
	)}
}

// sseEvent renders one SSE frame the way the Messages API does.
func sseEvent(name, data string) string {
	return "event: " + name + "\ndata: " + data + "\n\n"
}

// claudeSSE builds a full, well-formed stream: one text block made of the given
// deltas, then a message_delta carrying stopReason (and stopDetails, when set).
func claudeSSE(stopReason, stopDetails string, deltas ...string) string {
	var b strings.Builder
	b.WriteString(sseEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"usage":{"input_tokens":120,"output_tokens":1,"cache_read_input_tokens":30,"cache_creation_input_tokens":7}}}`))
	b.WriteString(sseEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`))
	for _, d := range deltas {
		dj, _ := json.Marshal(d)
		b.WriteString(sseEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":`+string(dj)+`}}`))
	}
	b.WriteString(sseEvent("content_block_stop", `{"type":"content_block_stop","index":0}`))
	sd := ""
	if stopDetails != "" {
		sd = `,"stop_details":` + stopDetails
	}
	b.WriteString(sseEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"`+stopReason+`","stop_sequence":null`+sd+`},"usage":{"output_tokens":345}}`))
	b.WriteString(sseEvent("message_stop", `{"type":"message_stop"}`))
	return b.String()
}

const claudePlainReply = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[{"type":"text","text":"hello there"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":3}}`

func TestAnthropicSmallRequestIsNotStreamed(t *testing.T) {
	as := &anthropicServer{reply: claudePlainReply}
	p := as.start(t)
	resp, err := p.complete(context.Background(), "claude-opus-5-5", 8000, Request{Messages: userMsg("hi")})
	if err != nil {
		t.Fatal(err)
	}
	if len(as.bodies) != 1 {
		t.Fatalf("requests = %d, want 1", len(as.bodies))
	}
	if _, ok := as.bodies[0]["stream"]; ok {
		t.Errorf("a request within the non-streaming limit must not stream: %v", as.bodies[0])
	}
	if resp.Text != "hello there" || resp.InputTokens != 12 || resp.OutputTokens != 3 || resp.StopReason != StopEndTurn {
		t.Errorf("resp = %+v", resp)
	}
}

func TestAnthropicLargeRequestIsStreamedAndAssembled(t *testing.T) {
	as := &anthropicServer{sse: claudeSSE("end_turn", "", "Hello, ", "large ", "world.")}
	p := as.start(t)
	req := Request{
		Messages:        userMsg("write a long report"),
		SystemCacheable: "you are a reporter",
		JSONSchema:      docSchema(),
		Tools:           []ToolDef{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
		WebSearch:       true,
	}
	resp, err := p.complete(context.Background(), "claude-opus-5-5", 48000, req)
	if err != nil {
		t.Fatal(err)
	}
	body := as.bodies[0]
	if body["stream"] != true {
		t.Fatalf("a request above the non-streaming limit must stream: %v", body)
	}
	// Everything in params rides the streaming path unchanged.
	if body["max_tokens"] != float64(48000) || body["output_config"] == nil || body["system"] == nil {
		t.Errorf("streaming body lost params: %v", body)
	}
	if tools, _ := body["tools"].([]any); len(tools) != 2 {
		t.Errorf("tools = %v, want web search + lookup", body["tools"])
	}
	if resp.Text != "Hello, large world." {
		t.Errorf("text = %q", resp.Text)
	}
	if resp.InputTokens != 120 || resp.OutputTokens != 345 || resp.CachedTokens != 30 || resp.CacheWriteTokens != 7 {
		t.Errorf("usage = %+v", resp)
	}
	if resp.StopReason != StopEndTurn {
		t.Errorf("stop = %v", resp.StopReason)
	}
}

func TestAnthropicStreamedTruncationIsSchemaViolation(t *testing.T) {
	as := &anthropicServer{sse: claudeSSE("max_tokens", "", `{"docType":"NAT`)}
	p := as.start(t)

	resp, err := p.complete(context.Background(), "claude-opus-5-5", 48000, Request{Messages: userMsg("x")})
	if err != nil || resp.StopReason != StopTruncated {
		t.Fatalf("resp=%+v err=%v, want StopTruncated", resp, err)
	}

	r := &router{
		providers: map[Provider]provider{ProviderAnthropic: p},
		profiles:  map[Profile]ModelRef{ProfileFast: {Provider: ProviderAnthropic, Model: "claude-opus-5-5"}},
		maxTokens: 100,
	}
	resp, err = r.Complete(context.Background(), Request{Task: TaskExtract, Messages: userMsg("x"), JSONSchema: docSchema(), MaxTokens: 48000, NoFallback: true})
	if !errors.Is(err, ErrSchemaViolation) {
		t.Fatalf("err = %v, want ErrSchemaViolation", err)
	}
	if resp.OutputTokens != 345 || resp.InputTokens != 120 {
		t.Errorf("violation dropped metering: %+v", resp)
	}
	if as.bodies[len(as.bodies)-1]["stream"] != true {
		t.Errorf("router call did not take the streaming path")
	}
}

func TestAnthropicStreamedRefusalIsRefusalError(t *testing.T) {
	as := &anthropicServer{sse: claudeSSE("refusal", `{"type":"refusal","category":"cyber","explanation":"declined"}`, "I can")}
	p := as.start(t)
	resp, err := p.complete(context.Background(), "claude-opus-5-5", 48000, Request{Messages: userMsg("x")})
	var re *RefusalError
	if !errors.As(err, &re) {
		t.Fatalf("err = %v, want *RefusalError", err)
	}
	if re.Provider != ProviderAnthropic || re.Model != "claude-opus-5-5" || re.Category != "cyber" || re.Explanation != "declined" {
		t.Errorf("refusal = %+v", re)
	}
	if resp.OutputTokens != 345 || resp.InputTokens != 120 || resp.Text != "I can" {
		t.Errorf("refusal dropped usage/partial text: %+v", resp)
	}
}

func TestAnthropicStreamErrorEventIsReturned(t *testing.T) {
	sse := sseEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":1}}}`) +
		sseEvent("error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
	p := (&anthropicServer{sse: sse}).start(t)
	_, err := p.complete(context.Background(), "claude-opus-5-5", 48000, Request{Messages: userMsg("x")})
	if err == nil {
		t.Fatal("a mid-stream error event must surface as an error")
	}
	// The SDK's own API error must come through (not a generic "stream ended"), so
	// failover classifies an overload exactly as it would on the non-streaming path.
	if !strings.Contains(strings.ToLower(err.Error()), "overloaded") || !shouldFailover(err) {
		t.Errorf("err = %v, want the SDK's overloaded error, classified as failover-worthy", err)
	}
}

func TestAnthropicStreamAbruptCloseIsError(t *testing.T) {
	full := claudeSSE("end_turn", "", "partial")
	// Cut the stream after the first delta: no message_delta, no message_stop.
	cut := full[:strings.Index(full, "event: content_block_stop")]
	for _, abort := range []bool{true, false} {
		p := (&anthropicServer{sse: cut, abort: abort}).start(t)
		_, err := p.complete(context.Background(), "claude-opus-5-5", 48000, Request{Messages: userMsg("x")})
		if err == nil {
			t.Errorf("abort=%v: a stream that ends before message_stop must be an error", abort)
			continue
		}
		if !shouldFailover(err) {
			t.Errorf("abort=%v: a dropped stream is transient and should fail over, got %v", abort, err)
		}
	}
}

func TestAnthropicStreamRespectsCancellation(t *testing.T) {
	p := (&anthropicServer{sse: claudeSSE("end_turn", "", "x")}).start(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.complete(ctx, "claude-opus-5-5", 48000, Request{Messages: userMsg("x")}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}
