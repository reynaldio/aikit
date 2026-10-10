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
	"time"

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

// msgStart is a message_start frame with the given usage JSON.
func msgStart(usage string) string {
	return sseEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"usage":`+usage+`}}`)
}

// blockStart / blockDelta / blockStop render content-block frames at index i.
func blockStart(i int, block string) string {
	return sseEvent("content_block_start", fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":%s}`, i, block))
}

func blockDelta(i int, delta string) string {
	return sseEvent("content_block_delta", fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":%s}`, i, delta))
}

func blockStop(i int) string {
	return sseEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, i))
}

func msgEnd(stopReason, usage string) string {
	return sseEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"`+stopReason+`","stop_sequence":null},"usage":`+usage+`}`) +
		sseEvent("message_stop", `{"type":"message_stop"}`)
}

func TestAnthropicStreamedTruncatedToolInputIsStopTruncated(t *testing.T) {
	// max_tokens can land mid tool_use input. Non-streaming reports that as
	// StopTruncated with usage; the stream must too, not fail re-marshalling the
	// half-written input JSON.
	head := msgStart(`{"input_tokens":50,"output_tokens":1}`) +
		blockStart(0, `{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}`) +
		blockDelta(0, `{"type":"input_json_delta","partial_json":"{\"a\":\"abc"}`)
	end := msgEnd("max_tokens", `{"output_tokens":99}`)
	// With and without the block's own content_block_stop: the SDK re-marshals
	// on both content_block_stop and message_stop.
	for name, sse := range map[string]string{"block stopped": head + blockStop(0) + end, "block left open": head + end} {
		p := (&anthropicServer{sse: sse}).start(t)
		resp, err := p.complete(context.Background(), "claude-opus-5-5", 48000, Request{Messages: userMsg("x")})
		if err != nil {
			t.Errorf("%s: truncated tool input must not be an error: %v", name, err)
			continue
		}
		if resp.StopReason != StopTruncated || resp.InputTokens != 50 || resp.OutputTokens != 99 {
			t.Errorf("%s: resp = %+v, want StopTruncated with usage", name, resp)
		}
	}
}

func TestAnthropicStreamedToolUseThinkingAndWebSearch(t *testing.T) {
	sse := msgStart(`{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":4,"cache_creation_input_tokens":3}`) +
		blockStart(0, `{"type":"thinking","thinking":"","signature":""}`) +
		blockDelta(0, `{"type":"thinking_delta","thinking":"let me think"}`) +
		blockDelta(0, `{"type":"signature_delta","signature":"sig"}`) +
		blockStop(0) +
		blockStart(1, `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}`) +
		blockDelta(1, `{"type":"input_json_delta","partial_json":"{\"query\":"}`) +
		blockDelta(1, `{"type":"input_json_delta","partial_json":"\"go sdk\"}"}`) +
		blockStop(1) +
		blockStart(2, `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://example.com","title":"Ex","encrypted_content":"enc","page_age":null}]}`) +
		blockStop(2) +
		blockStart(3, `{"type":"text","text":""}`) +
		blockDelta(3, `{"type":"text_delta","text":"Found it. "}`) +
		blockStop(3) +
		blockStart(4, `{"type":"tool_use","id":"toolu_9","name":"lookup","input":{}}`) +
		blockDelta(4, `{"type":"input_json_delta","partial_json":"{\"id\":"}`) +
		blockDelta(4, `{"type":"input_json_delta","partial_json":"42}"}`) +
		blockStop(4) +
		// message_delta usage that carries input/cache values overrides message_start's.
		msgEnd("tool_use", `{"input_tokens":11,"output_tokens":77,"cache_read_input_tokens":5,"cache_creation_input_tokens":6}`)
	p := (&anthropicServer{sse: sse}).start(t)
	resp, err := p.complete(context.Background(), "claude-opus-5-5", 48000, Request{Messages: userMsg("x"), WebSearch: true})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Found it. " {
		t.Errorf("text = %q (thinking/server blocks must not leak into text)", resp.Text)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "toolu_9" || resp.ToolCalls[0].Name != "lookup" {
		t.Fatalf("tool calls = %+v, want only the client tool_use", resp.ToolCalls)
	}
	var in map[string]any
	if err := json.Unmarshal(resp.ToolCalls[0].Input, &in); err != nil || in["id"] != float64(42) {
		t.Errorf("tool input = %s (%v)", resp.ToolCalls[0].Input, err)
	}
	if resp.StopReason != StopToolUse {
		t.Errorf("stop = %v", resp.StopReason)
	}
	if resp.InputTokens != 11 || resp.OutputTokens != 77 || resp.CachedTokens != 5 || resp.CacheWriteTokens != 6 {
		t.Errorf("usage = %+v, want message_delta's values", resp)
	}
}

func TestAnthropicStreamMidStreamCancellation(t *testing.T) {
	flushed := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(msgStart(`{"input_tokens":5,"output_tokens":1}`)))
		w.(http.Flusher).Flush()
		close(flushed)
		<-r.Context().Done() // hang until the client goes away
	}))
	t.Cleanup(srv.Close)
	p := &anthropicProvider{client: anthropic.NewClient(option.WithAPIKey("k"), option.WithBaseURL(srv.URL), option.WithMaxRetries(0))}

	ctx, cancel := context.WithCancel(context.Background())
	go func() { <-flushed; cancel() }()
	done := make(chan error, 1)
	go func() {
		_, err := p.complete(ctx, "claude-opus-5-5", 48000, Request{Messages: userMsg("x")})
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("complete did not return promptly after cancellation")
	}
}

func TestAnthropicStreamCompleteReplyIsKeptAfterMessageStop(t *testing.T) {
	// Once message_stop has arrived the reply is whole and billed: nothing after it
	// (a connection the server holds open, a cancel that lands meanwhile) may
	// discard it.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(claudeSSE("end_turn", "", "all here")))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)
	p := &anthropicProvider{client: anthropic.NewClient(option.WithAPIKey("k"), option.WithBaseURL(srv.URL), option.WithMaxRetries(0))}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	resp, err := p.complete(ctx, "claude-opus-5-5", 48000, Request{Messages: userMsg("x")})
	if err != nil || resp.Text != "all here" || resp.OutputTokens != 345 {
		t.Fatalf("resp=%+v err=%v, want the complete reply", resp, err)
	}
}

// collectText returns a request-ready OnEvent that records text pieces.
func collectText(got *[]string) func(StreamEvent) {
	return func(ev StreamEvent) {
		if ev.Kind == StreamText {
			*got = append(*got, ev.Text)
		}
	}
}

func TestAnthropicOnEventForcesStreamingForSmallRequest(t *testing.T) {
	as := &anthropicServer{sse: claudeSSE("end_turn", "", "hi")}
	p := as.start(t)
	var got []string
	if _, err := p.complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), OnEvent: collectText(&got)}); err != nil {
		t.Fatal(err)
	}
	if as.bodies[0]["stream"] != true {
		t.Errorf("OnEvent must force streaming: %v", as.bodies[0])
	}
}

func TestAnthropicStreamSendsTextPiecesInOrder(t *testing.T) {
	p := (&anthropicServer{sse: claudeSSE("end_turn", "", "Hel", "lo ", "world")}).start(t)
	var got []string
	resp, err := p.complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), OnEvent: collectText(&got)})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "Hel|lo |world" || resp.Text != "Hello world" {
		t.Errorf("events = %q, text = %q", got, resp.Text)
	}
}

func TestAnthropicStreamSendsOnlyText(t *testing.T) {
	sse := msgStart(`{"input_tokens":10,"output_tokens":1}`) +
		blockStart(0, `{"type":"thinking","thinking":"","signature":""}`) +
		blockDelta(0, `{"type":"thinking_delta","thinking":"let me think"}`) +
		blockDelta(0, `{"type":"signature_delta","signature":"sig"}`) +
		blockStop(0) +
		blockStart(1, `{"type":"text","text":""}`) +
		blockDelta(1, `{"type":"text_delta","text":"Found "}`) +
		blockDelta(1, `{"type":"text_delta","text":"it."}`) +
		blockStop(1) +
		blockStart(2, `{"type":"tool_use","id":"toolu_9","name":"lookup","input":{}}`) +
		blockDelta(2, `{"type":"input_json_delta","partial_json":"{\"id\":"}`) +
		blockDelta(2, `{"type":"input_json_delta","partial_json":"42}"}`) +
		blockStop(2) +
		msgEnd("tool_use", `{"output_tokens":7}`)
	p := (&anthropicServer{sse: sse}).start(t)
	var got []string
	resp, err := p.complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), OnEvent: collectText(&got)})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "Found |it." || strings.Join(got, "") != resp.Text {
		t.Errorf("events = %q, text = %q", got, resp.Text)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "lookup" || string(resp.ToolCalls[0].Input) != `{"id":42}` {
		t.Errorf("tool calls = %+v", resp.ToolCalls)
	}
}

func TestAnthropicStreamTwoTextBlocksJoinWithoutSeparator(t *testing.T) {
	sse := msgStart(`{"input_tokens":10,"output_tokens":1}`) +
		blockStart(0, `{"type":"text","text":""}`) +
		blockDelta(0, `{"type":"text_delta","text":"Before. "}`) +
		blockStop(0) +
		blockStart(1, `{"type":"tool_use","id":"toolu_9","name":"lookup","input":{}}`) +
		blockDelta(1, `{"type":"input_json_delta","partial_json":"{}"}`) +
		blockStop(1) +
		blockStart(2, `{"type":"text","text":""}`) +
		blockDelta(2, `{"type":"text_delta","text":"After."}`) +
		blockStop(2) +
		msgEnd("tool_use", `{"output_tokens":7}`)
	p := (&anthropicServer{sse: sse}).start(t)
	var got []string
	resp, err := p.complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), OnEvent: collectText(&got)})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "") != resp.Text || resp.Text != "Before. After." {
		t.Errorf("events = %q, text = %q", got, resp.Text)
	}
}

func TestAnthropicStreamRefusalKeepsStreamedText(t *testing.T) {
	p := (&anthropicServer{sse: claudeSSE("refusal", `{"type":"refusal","category":"cyber","explanation":"declined"}`, "I ", "can")}).start(t)
	var got []string
	resp, err := p.complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), OnEvent: collectText(&got)})
	var re *RefusalError
	if !errors.As(err, &re) {
		t.Fatalf("err = %v, want *RefusalError", err)
	}
	if strings.Join(got, "|") != "I |can" || resp.Text != "I can" || resp.OutputTokens != 345 {
		t.Errorf("events = %q, resp = %+v", got, resp)
	}
}

func TestAnthropicStreamEndedEarlyKeepsSentEvents(t *testing.T) {
	full := claudeSSE("end_turn", "", "partial")
	cut := full[:strings.Index(full, "event: content_block_stop")]
	p := (&anthropicServer{sse: cut, abort: true}).start(t)
	var got []string
	_, err := p.complete(context.Background(), "claude-opus-5-5", 100, Request{Messages: userMsg("x"), OnEvent: collectText(&got)})
	if err == nil {
		t.Fatal("a stream that ends before message_stop must be an error")
	}
	if strings.Join(got, "|") != "partial" {
		t.Errorf("events = %q, want the one already sent", got)
	}
}

// quietServer writes message_start and one text delta, flushes, then goes quiet
// until the client leaves.
func quietServer(t *testing.T) *anthropicProvider {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(msgStart(`{"input_tokens":5,"output_tokens":1}`) +
			blockStart(0, `{"type":"text","text":""}`) +
			blockDelta(0, `{"type":"text_delta","text":"first"}`)))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)
	return &anthropicProvider{client: anthropic.NewClient(option.WithAPIKey("k"), option.WithBaseURL(srv.URL), option.WithMaxRetries(0))}
}

func TestAnthropicStreamCancelledFromOnEvent(t *testing.T) {
	p := quietServer(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var got []string
	record := collectText(&got)
	onEvent := func(ev StreamEvent) { record(ev); cancel() }
	_, err := p.complete(ctx, "claude-opus-5-5", 100, Request{Messages: userMsg("x"), OnEvent: onEvent})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(got) != 1 || got[0] != "first" {
		t.Errorf("events = %q, want exactly one", got)
	}
}

func TestAnthropicStreamQuietPastDeadline(t *testing.T) {
	p := quietServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	var got []string
	_, err := p.complete(ctx, "claude-opus-5-5", 100, Request{Messages: userMsg("x"), OnEvent: collectText(&got)})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
}
