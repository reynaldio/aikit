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

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

type abReq struct {
	method, path, accept string
	body                 []byte
}

// abServer records every request and answers with h (n is the 0-based call number).
func abServer(t *testing.T, h func(w http.ResponseWriter, r *http.Request, n int)) (*httptest.Server, func() []abReq) {
	t.Helper()
	var mu sync.Mutex
	var seen []abReq
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		n := len(seen)
		seen = append(seen, abReq{r.Method, r.URL.Path, r.Header.Get("Accept"), b})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		h(w, r, n)
	}))
	t.Cleanup(srv.Close)
	return srv, func() []abReq {
		mu.Lock()
		defer mu.Unlock()
		return append([]abReq(nil), seen...)
	}
}

func abBackend(srv *httptest.Server) batchBackend {
	return newAnthropicBatch("k", nil, option.WithBaseURL(srv.URL))
}

const abBatchJSON = `{"id":"msgbatch_1","type":"message_batch","processing_status":"in_progress","request_counts":{"processing":2,"succeeded":0,"errored":0,"canceled":0,"expired":0}}`

func abMsg(content, stop string, in, out int) string {
	return `{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":` + content +
		`,"stop_reason":"` + stop + `","usage":{"input_tokens":` + itoa(in) + `,"output_tokens":` + itoa(out) + `}}`
}

func itoa(n int) string { b, _ := json.Marshal(n); return string(b) }

func abFullRequest() Request {
	return Request{
		SystemCacheable: "be brief",
		Messages: []Message{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "hi", ToolCalls: []ToolCall{{ID: "t1", Name: "lookup", Input: json.RawMessage(`{"q":"x"}`)}}},
			{Role: "user", Content: "again", ToolResults: []ToolResult{{ToolCallID: "t1", Content: "ok"}}},
		},
		CacheHistory: true,
		WebSearch:    true,
		UserLocation: &UserLocation{City: "Jakarta", Country: "ID"},
		Tools:        []ToolDef{{Name: "lookup", Description: "find", Schema: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}, "required": []string{"q"}}}},
		JSONSchema:   map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}},
		Effort:       EffortHigh,
	}
}

// A2: every field anthropicParams sets must survive the copy into the batch params.
func TestAnthropicBatchParamsParity(t *testing.T) {
	be := newAnthropicBatch("k", nil).(*anthropicBatch)
	req := abFullRequest()
	live, err := json.Marshal(anthropicParams("claude-x", 777, req))
	if err != nil {
		t.Fatal(err)
	}
	batch, err := be.buildChat("claude-x", 777, req)
	if err != nil {
		t.Fatal(err)
	}
	var a, b map[string]any
	if err := json.Unmarshal(live, &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(batch, &b); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"system", "tools", "output_config", "messages", "max_tokens", "model"} {
		if _, ok := a[k]; !ok {
			t.Fatalf("live params lack %q: the test request is not using every feature", k)
		}
	}
	if !reflect.DeepEqual(a, b) {
		t.Errorf("batch params differ from live params\nlive:  %s\nbatch: %s", live, batch)
	}
}

func TestAnthropicBatchSupportsAndLimits(t *testing.T) {
	be := newAnthropicBatch("k", nil)
	if !be.supports(batchChat) || !be.supports(batchChatJSON) || be.supports(batchEmbed) {
		t.Error("supports: want chat and chatjson only")
	}
	if l := be.limits(batchChat); l.maxItems != 100000 || l.maxBytes != 256<<20 {
		t.Errorf("limits = %+v", l)
	}
}

// A3
func TestAnthropicBatchSubmitBody(t *testing.T) {
	srv, seen := abServer(t, func(w http.ResponseWriter, r *http.Request, n int) { _, _ = w.Write([]byte(abBatchJSON)) })
	be := abBackend(srv)
	var lines []batchLine
	for _, id := range []string{"b", "a"} {
		body, err := be.buildChat("claude-x", 300, Request{SystemCacheable: "sys", Messages: userMsg("hi " + id)})
		if err != nil {
			t.Fatal(err)
		}
		lines = append(lines, batchLine{customID: id, body: body})
	}
	id, err := be.submit(context.Background(), batchChat, "claude-x", lines)
	if err != nil {
		t.Fatal(err)
	}
	if id != "msgbatch_1" {
		t.Errorf("id = %q", id)
	}
	reqs := seen()
	if len(reqs) != 1 || reqs[0].method != "POST" || reqs[0].path != "/v1/messages/batches" {
		t.Fatalf("requests = %+v", reqs)
	}
	want := `{"requests":[{"custom_id":"b","params":` + string(lines[0].body) + `},{"custom_id":"a","params":` + string(lines[1].body) + `}]}`
	if string(reqs[0].body) != want {
		t.Errorf("body\n got %s\nwant %s", reqs[0].body, want)
	}
	var got struct {
		Requests []struct {
			CustomID string         `json:"custom_id"`
			Params   map[string]any `json:"params"`
		} `json:"requests"`
	}
	if err := json.Unmarshal(reqs[0].body, &got); err != nil {
		t.Fatal(err)
	}
	p := got.Requests[0].Params
	if p["model"] != "claude-x" || p["max_tokens"] != float64(300) {
		t.Errorf("params = %v", p)
	}
	if _, ok := p["stream"]; ok {
		t.Error("stream must not be set")
	}
	sys, _ := p["system"].([]any)
	if len(sys) != 1 || sys[0].(map[string]any)["cache_control"] == nil {
		t.Errorf("system = %v", p["system"])
	}
}

// A4
func TestAnthropicBatchCreate503NotRetried(t *testing.T) {
	srv, seen := abServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		w.Header().Set("retry-after-ms", "1")
		w.WriteHeader(503)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"api_error","message":"down"}}`))
	})
	be := abBackend(srv)
	_, err := be.submit(context.Background(), batchChat, "claude-x", []batchLine{{customID: "a", body: []byte(`{"model":"claude-x","max_tokens":1,"messages":[]}`)}})
	if err == nil || !strings.Contains(err.Error(), "may still have been created") {
		t.Fatalf("err = %v", err)
	}
	var ae *anthropic.Error
	if !errors.As(err, &ae) || ae.StatusCode != 503 {
		t.Errorf("SDK error not reachable: %v", err)
	}
	if n := len(seen()); n != 1 {
		t.Errorf("create requests = %d, want exactly 1", n)
	}
}

func TestAnthropicBatchCreateEmptyID(t *testing.T) {
	srv, _ := abServer(t, func(w http.ResponseWriter, r *http.Request, n int) { _, _ = w.Write([]byte(`{}`)) })
	_, err := abBackend(srv).submit(context.Background(), batchChat, "m", []batchLine{{customID: "a", body: []byte(`{}`)}})
	if err == nil || !strings.Contains(err.Error(), "may still have been created") {
		t.Fatalf("err = %v", err)
	}
}

// A5
func TestAnthropicBatchStatus(t *testing.T) {
	cases := []struct {
		status string
		want   BatchState
	}{{"in_progress", BatchRunning}, {"canceling", BatchRunning}, {"ended", BatchDone}}
	for _, c := range cases {
		t.Run(c.status, func(t *testing.T) {
			srv, seen := abServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
				_, _ = w.Write([]byte(`{"id":"msgbatch_1","type":"message_batch","processing_status":"` + c.status +
					`","request_counts":{"processing":1,"succeeded":5,"errored":2,"canceled":3,"expired":4}}`))
			})
			st, err := abBackend(srv).status(context.Background(), "msgbatch_1")
			if err != nil {
				t.Fatal(err)
			}
			want := BatchStatus{State: c.want, Total: 15, Succeeded: 5, Failed: 2, Canceled: 3, Expired: 4}
			if st != want {
				t.Errorf("status = %+v, want %+v", st, want)
			}
			if r := seen(); len(r) != 1 || r[0].method != "GET" || r[0].path != "/v1/messages/batches/msgbatch_1" {
				t.Errorf("requests = %+v", r)
			}
		})
	}
}

func abResultsBody() string {
	l := []string{
		`{"custom_id":"ok","result":{"type":"succeeded","message":` + abMsg(`[{"type":"text","text":"hello"}]`, "end_turn", 10, 5) + `}}`,
		`{"custom_id":"tool","result":{"type":"succeeded","message":` + abMsg(`[{"type":"tool_use","id":"t1","name":"lookup","input":{"q":"x"}}]`, "tool_use", 7, 3) + `}}`,
		`{"custom_id":"refused","result":{"type":"succeeded","message":` + abMsg(`[]`, "refusal", 4, 1) + `}}`,
		`{"custom_id":"bad","result":{"type":"errored","error":{"type":"error","error":{"type":"invalid_request_error","message":"too long"}}}}`,
		`{"custom_id":"can","result":{"type":"canceled"}}`,
		`{"custom_id":"exp","result":{"type":"expired"}}`,
	}
	return strings.Join(l, "\n") + "\n"
}

// A6
func TestAnthropicBatchResults(t *testing.T) {
	srv, seen := abServer(t, func(w http.ResponseWriter, r *http.Request, n int) { _, _ = w.Write([]byte(abResultsBody())) })
	var got []rawBatchResult
	for r, err := range abBackend(srv).results(context.Background(), batchChat, "claude-x", "msgbatch_1") {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, r)
	}
	if len(got) != 6 {
		t.Fatalf("got %d results", len(got))
	}
	ids := []string{"ok", "tool", "refused", "bad", "can", "exp"}
	for i, id := range ids {
		if got[i].customID != id {
			t.Errorf("order[%d] = %q, want %q", i, got[i].customID, id)
		}
	}
	if got[0].err != nil || got[0].resp.Text != "hello" || got[0].resp.InputTokens != 10 || got[0].resp.OutputTokens != 5 {
		t.Errorf("ok = %+v", got[0])
	}
	if got[1].err != nil || len(got[1].resp.ToolCalls) != 1 || got[1].resp.ToolCalls[0].Name != "lookup" || got[1].resp.StopReason != StopToolUse {
		t.Errorf("tool = %+v", got[1])
	}
	if !errors.Is(got[2].err, ErrRefused) || got[2].resp.InputTokens != 4 || got[2].resp.OutputTokens != 1 {
		t.Errorf("refused = %+v", got[2])
	}
	if got[3].err == nil || !strings.Contains(got[3].err.Error(), "invalid_request_error") || !strings.Contains(got[3].err.Error(), "too long") {
		t.Errorf("errored = %v", got[3].err)
	}
	if !errors.Is(got[4].err, ErrBatchItemCanceled) || !errors.Is(got[5].err, ErrBatchItemExpired) {
		t.Errorf("canceled/expired = %v / %v", got[4].err, got[5].err)
	}
	r := seen()
	if len(r) != 1 || r[0].method != "GET" || r[0].path != "/v1/messages/batches/msgbatch_1/results" || r[0].accept != "application/x-jsonl" {
		t.Errorf("requests = %+v", r)
	}
}

// A7
func TestAnthropicBatchResultsLongLine(t *testing.T) {
	big := strings.Repeat("x", 200<<10)
	body := `{"custom_id":"big","result":{"type":"succeeded","message":` + abMsg(`[{"type":"text","text":"`+big+`"}]`, "end_turn", 1, 2) + `}}` + "\n"
	srv, _ := abServer(t, func(w http.ResponseWriter, r *http.Request, n int) { _, _ = w.Write([]byte(body)) })
	n := 0
	for r, err := range abBackend(srv).results(context.Background(), batchChat, "m", "b") {
		if err != nil || r.err != nil || r.resp.Text != big {
			t.Fatalf("err=%v rerr=%v len=%d", err, r.err, len(r.resp.Text))
		}
		n++
	}
	if n != 1 {
		t.Errorf("n = %d", n)
	}
}

// A8
func TestAnthropicBatchResultsRetried(t *testing.T) {
	srv, seen := abServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		if n == 0 {
			w.Header().Set("retry-after-ms", "1")
			w.WriteHeader(503)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"api_error","message":"down"}}`))
			return
		}
		_, _ = w.Write([]byte(abResultsBody()))
	})
	n := 0
	for _, err := range abBackend(srv).results(context.Background(), batchChat, "m", "b") {
		if err != nil {
			t.Fatal(err)
		}
		n++
	}
	if n != 6 || len(seen()) != 2 {
		t.Errorf("results = %d, requests = %d", n, len(seen()))
	}
}

func TestAnthropicBatchResultsStopEarly(t *testing.T) {
	srv, _ := abServer(t, func(w http.ResponseWriter, r *http.Request, n int) { _, _ = w.Write([]byte(abResultsBody())) })
	n := 0
	for _, err := range abBackend(srv).results(context.Background(), batchChat, "m", "b") {
		if err != nil {
			t.Fatal(err)
		}
		n++
		break
	}
	if n != 1 {
		t.Errorf("n = %d", n)
	}
}

// A9
func TestAnthropicBatchCancel(t *testing.T) {
	srv, seen := abServer(t, func(w http.ResponseWriter, r *http.Request, n int) { _, _ = w.Write([]byte(abBatchJSON)) })
	if err := abBackend(srv).cancel(context.Background(), "msgbatch_1"); err != nil {
		t.Fatal(err)
	}
	if r := seen(); len(r) != 1 || r[0].method != "POST" || r[0].path != "/v1/messages/batches/msgbatch_1/cancel" {
		t.Errorf("requests = %+v", r)
	}
}

// A10
func TestAnthropicBatchThroughCore(t *testing.T) {
	srv, seen := abServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		switch {
		case r.Method == "POST":
			_, _ = w.Write([]byte(abBatchJSON))
		case strings.HasSuffix(r.URL.Path, "/results"):
			_, _ = w.Write([]byte(abResultsBody()))
		default:
			_, _ = w.Write([]byte(`{"id":"msgbatch_1","type":"message_batch","processing_status":"ended","request_counts":{"processing":0,"succeeded":2,"errored":0,"canceled":0,"expired":0}}`))
		}
	})
	cfg := Config{Profiles: map[Profile]ModelRef{ProfileChat: {Provider: ProviderAnthropic, Model: "claude-x"}}}
	b := newBatcher(cfg, map[Provider]batchBackend{ProviderAnthropic: abBackend(srv)})
	ctx := context.Background()
	id, err := b.SubmitChat(ctx, ChatBatch{Items: []ChatItem{
		{CustomID: "ok", Request: Request{Messages: userMsg("a")}},
		{CustomID: "tool", Request: Request{Messages: userMsg("b")}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(id), "anthropic:chat:") {
		t.Errorf("id = %q", id)
	}
	st, err := b.Status(ctx, id)
	if err != nil || st.State != BatchDone || st.Succeeded != 2 {
		t.Fatalf("status = %+v, %v", st, err)
	}
	res, err := CollectResults(b.Results(ctx, id))
	if err != nil || len(res) != 6 {
		t.Fatalf("results = %d, %v", len(res), err)
	}
	if res[0].CustomID != "ok" || res[0].Response.Text != "hello" || res[0].Response.Provider != ProviderAnthropic || res[0].Response.Model != "claude-x" {
		t.Errorf("res[0] = %+v", res[0])
	}
	_ = seen
}
