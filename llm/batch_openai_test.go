package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

type obReq struct {
	method, path, ctype, auth string
	body                      []byte
}

// obServer records every request; h answers (n is the 0-based call number).
func obServer(t *testing.T, h func(w http.ResponseWriter, r *http.Request, n int)) (*httptest.Server, func() []obReq) {
	t.Helper()
	var mu sync.Mutex
	var seen []obReq
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		n := len(seen)
		seen = append(seen, obReq{r.Method, r.URL.Path, r.Header.Get("Content-Type"), r.Header.Get("Authorization"), b})
		mu.Unlock()
		h(w, r, n)
	}))
	t.Cleanup(srv.Close)
	return srv, func() []obReq {
		mu.Lock()
		defer mu.Unlock()
		return append([]obReq(nil), seen...)
	}
}

func obBackend(name Provider, srv *httptest.Server) *openaiBatch {
	return newOpenAIBatch(name, "key-"+string(name), srv.URL+"/v1", nil).(*openaiBatch)
}

func obLines(t *testing.T, r obReq) (fields map[string]string, lines []map[string]any) {
	t.Helper()
	mt, params, err := mime.ParseMediaType(r.ctype)
	if err != nil || mt != "multipart/form-data" {
		t.Fatalf("content type %q", r.ctype)
	}
	mr := multipart.NewReader(strings.NewReader(string(r.body)), params["boundary"])
	fields = map[string]string{}
	for {
		p, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(p)
		if p.FormName() == "file" {
			fields["filename"] = p.FileName()
			fields["filetype"] = p.Header.Get("Content-Type")
			for l := range jsonlLines(strings.NewReader(string(b))) {
				var m map[string]any
				if err := json.Unmarshal([]byte(l), &m); err != nil {
					t.Fatalf("bad line %q: %v", l, err)
				}
				lines = append(lines, m)
			}
			continue
		}
		fields[p.FormName()] = string(b)
	}
	return
}

func obSubmitHandler(w http.ResponseWriter, r *http.Request, n int) {
	w.Header().Set("Content-Type", "application/json")
	if strings.HasSuffix(r.URL.Path, "/files") {
		_, _ = w.Write([]byte(`{"id":"file-in"}`))
		return
	}
	_, _ = w.Write([]byte(`{"id":"batch_1","status":"validating"}`))
}

func obSchemaReq() Request {
	return Request{
		Messages:   []Message{{Role: "user", Content: "hi"}},
		JSONSchema: map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}, "required": []string{"a"}},
	}
}

// O2 + O4
func TestOpenAIBatchUploadAndCreate(t *testing.T) {
	for _, name := range []Provider{ProviderOpenAI, ProviderMoonshot} {
		srv, seen := obServer(t, obSubmitHandler)
		be := obBackend(name, srv)
		body, err := be.buildChat("gpt-4o", 300, obSchemaReq())
		if err != nil {
			t.Fatal(err)
		}
		id, err := be.submit(context.Background(), batchChatJSON, "gpt-4o", []batchLine{{"a", body}, {"b", body}})
		if err != nil || id != "batch_1" {
			t.Fatalf("%s: id=%q err=%v", name, id, err)
		}
		reqs := seen()
		if len(reqs) != 2 || reqs[0].path != "/v1/files" || reqs[1].path != "/v1/batches" {
			t.Fatalf("%s: requests = %+v", name, reqs)
		}
		for _, r := range reqs {
			if r.auth != "Bearer key-"+string(name) {
				t.Errorf("%s: auth = %q", name, r.auth)
			}
		}
		f, lines := obLines(t, reqs[0])
		if f["purpose"] != "batch" || f["filename"] != "batch.jsonl" || f["filetype"] != "application/jsonl" || len(lines) != 2 {
			t.Fatalf("%s: upload = %v, %d lines", name, f, len(lines))
		}
		l := lines[0]
		if l["custom_id"] != "a" || l["method"] != "POST" || l["url"] != "/v1/chat/completions" {
			t.Errorf("line = %v", l)
		}
		b := l["body"].(map[string]any)
		if b["model"] != "gpt-4o" || b["max_tokens"] != float64(300) {
			t.Errorf("body = %v", b)
		}
		if _, ok := b["stream"]; ok {
			t.Error("stream set")
		}
		if _, ok := b["stream_options"]; ok {
			t.Error("stream_options set")
		}
		rf, has := b["response_format"].(map[string]any)
		if name == ProviderOpenAI {
			js, _ := rf["json_schema"].(map[string]any)
			if !has || js["strict"] != true {
				t.Errorf("openai response_format = %v", b["response_format"])
			}
		} else if has {
			t.Errorf("moonshot got response_format %v", rf)
		}
		var c map[string]string
		if err := json.Unmarshal(reqs[1].body, &c); err != nil {
			t.Fatal(err)
		}
		if c["input_file_id"] != "file-in" || c["endpoint"] != "/v1/chat/completions" || c["completion_window"] != "24h" {
			t.Errorf("create body = %v", c)
		}
	}
}

// O3
func TestOpenAIBatchEmbedLine(t *testing.T) {
	srv, seen := obServer(t, obSubmitHandler)
	be := obBackend(ProviderOpenAI, srv)
	for _, dims := range []int{0, 256} {
		body, err := be.buildEmbed("text-embedding-3-small", "hello", EmbedQuery, dims)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := be.submit(context.Background(), batchEmbed, "m", []batchLine{{"e1", body}}); err != nil {
			t.Fatal(err)
		}
	}
	reqs := seen()
	for i, want := range []float64{0, 256} {
		_, lines := obLines(t, reqs[i*2])
		if lines[0]["url"] != "/v1/embeddings" {
			t.Errorf("url = %v", lines[0]["url"])
		}
		b := lines[0]["body"].(map[string]any)
		if b["input"] != "hello" || b["encoding_format"] != "float" || b["model"] != "text-embedding-3-small" {
			t.Errorf("body = %v", b)
		}
		d, has := b["dimensions"]
		if (want == 0) == has || (has && d != want) {
			t.Errorf("dimensions = %v (%v), want %v", d, has, want)
		}
		var c map[string]string
		_ = json.Unmarshal(reqs[i*2+1].body, &c)
		if c["endpoint"] != "/v1/embeddings" {
			t.Errorf("endpoint = %q", c["endpoint"])
		}
	}
}

// O5: create is never retried; upload is.
func TestOpenAIBatchCreate503NotRetried(t *testing.T) {
	srv, seen := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		if strings.HasSuffix(r.URL.Path, "/files") {
			obSubmitHandler(w, r, n)
			return
		}
		w.Header().Set("retry-after-ms", "1")
		w.WriteHeader(503)
		_, _ = w.Write([]byte(`{"error":{"message":"down"}}`))
	})
	be := obBackend(ProviderOpenAI, srv)
	_, err := be.submit(context.Background(), batchChat, "m", []batchLine{{"a", []byte(`{}`)}})
	if err == nil || !strings.Contains(err.Error(), "may still have been created") || !strings.Contains(err.Error(), "down") {
		t.Fatalf("err = %v", err)
	}
	creates := 0
	for _, r := range seen() {
		if r.path == "/v1/batches" {
			creates++
		}
	}
	if creates != 1 {
		t.Errorf("create requests = %d, want exactly 1", creates)
	}
}

func TestOpenAIBatchCreateBadReplies(t *testing.T) {
	for name, reply := range map[string]string{"no id": `{"status":"validating"}`, "not json": `oops`} {
		srv, _ := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
			if strings.HasSuffix(r.URL.Path, "/files") {
				obSubmitHandler(w, r, n)
				return
			}
			_, _ = w.Write([]byte(reply))
		})
		_, err := obBackend(ProviderOpenAI, srv).submit(context.Background(), batchChat, "m", []batchLine{{"a", []byte(`{}`)}})
		if err == nil || !strings.Contains(err.Error(), "may still have been created") {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}

func TestOpenAIBatchUpload503ThenOK(t *testing.T) {
	var fileCalls atomic.Int32
	srv, seen := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		if strings.HasSuffix(r.URL.Path, "/files") && fileCalls.Add(1) == 1 {
			w.Header().Set("retry-after-ms", "1")
			w.WriteHeader(503)
			return
		}
		obSubmitHandler(w, r, n)
	})
	id, err := obBackend(ProviderOpenAI, srv).submit(context.Background(), batchChat, "m", []batchLine{{"a", []byte(`{}`)}})
	if err != nil || id != "batch_1" {
		t.Fatalf("id=%q err=%v", id, err)
	}
	creates, uploads := 0, 0
	var firstBody, secondBody []byte
	for _, r := range seen() {
		switch r.path {
		case "/v1/batches":
			creates++
		case "/v1/files":
			uploads++
			if uploads == 1 {
				firstBody = r.body
			} else {
				secondBody = r.body
			}
		}
	}
	if creates != 1 || uploads != 2 || len(firstBody) == 0 || string(firstBody) != string(secondBody) {
		t.Errorf("creates=%d uploads=%d, bodies equal=%v", creates, uploads, string(firstBody) == string(secondBody))
	}
}

func TestOpenAIBatchStatus503ThenOK(t *testing.T) {
	srv, seen := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		if n == 0 {
			w.Header().Set("retry-after-ms", "1")
			w.WriteHeader(503)
			return
		}
		_, _ = w.Write([]byte(`{"status":"in_progress"}`))
	})
	st, err := obBackend(ProviderOpenAI, srv).status(context.Background(), "b1")
	if err != nil || st.State != BatchRunning || len(seen()) != 2 {
		t.Fatalf("st=%+v err=%v calls=%d", st, err, len(seen()))
	}
}

// O6
func TestOpenAIBatchStatusMapping(t *testing.T) {
	cases := map[string]BatchState{
		"validating": BatchRunning, "in_progress": BatchRunning, "finalizing": BatchRunning, "cancelling": BatchRunning,
		"completed": BatchDone, "expired": BatchDone, "cancelled": BatchDone, "failed": BatchFailed,
	}
	for status, want := range cases {
		srv, seen := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
			_, _ = w.Write([]byte(`{"id":"b1","status":"` + status + `","request_counts":{"total":10,"completed":7,"failed":2},` +
				`"errors":{"data":[{"message":"bad file"},{"message":"too big"}]}}`))
		})
		st, err := obBackend(ProviderOpenAI, srv).status(context.Background(), "b1")
		if err != nil {
			t.Fatal(err)
		}
		if st.State != want || st.Total != 10 || st.Succeeded != 7 || st.Failed != 2 {
			t.Errorf("%s: %+v", status, st)
		}
		if want == BatchFailed && (st.Err == nil || !strings.Contains(st.Err.Error(), "bad file; too big")) {
			t.Errorf("failed Err = %v", st.Err)
		}
		if want != BatchFailed && st.Err != nil {
			t.Errorf("%s: Err = %v", status, st.Err)
		}
		if r := seen()[0]; r.method != "GET" || r.path != "/v1/batches/b1" {
			t.Errorf("request = %+v", r)
		}
	}
	srv, _ := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) { _, _ = w.Write([]byte(`{"status":"failed"}`)) })
	st, _ := obBackend(ProviderOpenAI, srv).status(context.Background(), "b1")
	if st.Err == nil || !strings.Contains(st.Err.Error(), "batch failed") {
		t.Errorf("default failed text = %v", st.Err)
	}
	srv, _ = obServer(t, func(w http.ResponseWriter, r *http.Request, n int) { _, _ = w.Write([]byte(`{"status":"brand_new"}`)) })
	st, err := obBackend(ProviderOpenAI, srv).status(context.Background(), "b1")
	if err != nil || st.State != BatchRunning {
		t.Errorf("unknown status: %+v %v", st, err)
	}
}

const obOutputFile = `{"custom_id":"ok","response":{"status_code":200,"body":{"choices":[{"message":{"content":"hello","tool_calls":[{"id":"t1","type":"function","function":{"name":"lookup","arguments":"{\"q\":1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4}}}},"error":null}
{"custom_id":"refused","response":{"status_code":200,"body":{"choices":[{"message":{"content":"","refusal":"no"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}},"error":null}
{"custom_id":"bad","response":{"status_code":400,"body":{"error":{"message":"bad param"}}},"error":null}
`

const obErrorFile = `{"custom_id":"late","response":null,"error":{"code":"batch_expired","message":"expired"}}
{"custom_id":"stopped","response":null,"error":{"code":"batch_cancelled","message":"cancelled"}}
{"custom_id":"weird","response":null,"error":{"code":"server_error","message":"oops"}}
`

func obResultsHandler(withErrFile bool, outFile, errFile string) func(http.ResponseWriter, *http.Request, int) {
	return func(w http.ResponseWriter, r *http.Request, n int) {
		switch r.URL.Path {
		case "/v1/batches/b1":
			if withErrFile {
				_, _ = w.Write([]byte(`{"status":"completed","output_file_id":"fo","error_file_id":"fe"}`))
			} else {
				_, _ = w.Write([]byte(`{"status":"completed","output_file_id":"fo","error_file_id":""}`))
			}
		case "/v1/files/fo/content":
			_, _ = w.Write([]byte(outFile))
		case "/v1/files/fe/content":
			_, _ = w.Write([]byte(errFile))
		default:
			w.WriteHeader(404)
		}
	}
}

func obCollect(t *testing.T, seq func(func(rawBatchResult, error) bool)) map[string]rawBatchResult {
	t.Helper()
	out := map[string]rawBatchResult{}
	for r, err := range seq {
		if err != nil {
			t.Fatal(err)
		}
		out[r.customID] = r
	}
	return out
}

// O7
func TestOpenAIBatchResults(t *testing.T) {
	srv, seen := obServer(t, obResultsHandler(true, obOutputFile, obErrorFile))
	be := obBackend(ProviderOpenAI, srv)
	var order []string
	got := map[string]rawBatchResult{}
	for r, err := range be.results(context.Background(), batchChat, "gpt-4o", "b1") {
		if err != nil {
			t.Fatal(err)
		}
		order = append(order, r.customID)
		got[r.customID] = r
	}
	if strings.Join(order, ",") != "ok,refused,bad,late,stopped,weird" {
		t.Errorf("order = %v", order)
	}
	ok := got["ok"]
	if ok.err != nil || ok.resp.Text != "hello" || ok.resp.InputTokens != 16 || ok.resp.CachedTokens != 4 || ok.resp.OutputTokens != 5 ||
		len(ok.resp.ToolCalls) != 1 || ok.resp.ToolCalls[0].Name != "lookup" {
		t.Errorf("ok = %+v", ok)
	}
	var re *RefusalError
	if !errors.As(got["refused"].err, &re) {
		t.Errorf("refused err = %v", got["refused"].err)
	}
	if e := got["bad"].err; e == nil || !strings.Contains(e.Error(), "bad param") || !strings.Contains(e.Error(), "400") {
		t.Errorf("bad err = %v", e)
	}
	if !errors.Is(got["late"].err, ErrBatchItemExpired) || !errors.Is(got["stopped"].err, ErrBatchItemCanceled) {
		t.Errorf("late=%v stopped=%v", got["late"].err, got["stopped"].err)
	}
	if e := got["weird"].err; e == nil || !strings.Contains(e.Error(), "oops") || !strings.Contains(e.Error(), "server_error") {
		t.Errorf("weird err = %v", e)
	}
	var paths []string
	for _, r := range seen() {
		paths = append(paths, r.path)
	}
	if strings.Join(paths, ",") != "/v1/batches/b1,/v1/files/fo/content,/v1/files/fe/content" {
		t.Errorf("paths = %v", paths)
	}

	// No error file: only the output file is read.
	srv, seen = obServer(t, obResultsHandler(false, obOutputFile, ""))
	obCollect(t, obBackend(ProviderOpenAI, srv).results(context.Background(), batchChat, "gpt-4o", "b1"))
	if len(seen()) != 2 {
		t.Errorf("requests without error file = %d", len(seen()))
	}
}

func TestOpenAIBatchResultsEmbed(t *testing.T) {
	out := `{"custom_id":"e1","response":{"status_code":200,"body":{"data":[{"index":0,"embedding":[3,4]}],"usage":{"prompt_tokens":9}}},"error":null}
{"custom_id":"e2","response":{"status_code":200,"body":{"data":[{"index":0,"embedding":[1,0]}]}},"error":null}
{"custom_id":"e3","response":{"status_code":200,"body":{"data":[]}},"error":null}
`
	srv, _ := obServer(t, obResultsHandler(false, out, ""))
	got := obCollect(t, obBackend(ProviderOpenAI, srv).results(context.Background(), batchEmbed, "m", "b1"))
	if len(got["e1"].vector) != 2 || got["e1"].vector[0] != 3 || got["e1"].resp.InputTokens != 9 || got["e1"].err != nil {
		t.Errorf("e1 = %+v", got["e1"])
	}
	if got["e2"].err != nil || got["e2"].resp.InputTokens != 0 || got["e2"].resp.UsageEstimated {
		t.Errorf("e2 = %+v", got["e2"])
	}
	if got["e3"].err == nil {
		t.Error("e3 should be an error")
	}
}

// O8
func TestOpenAIBatchLongLine(t *testing.T) {
	text := strings.Repeat("x", 500<<10)
	out := `{"custom_id":"big","response":{"status_code":200,"body":{"choices":[{"message":{"content":"` + text + `"},"finish_reason":"stop"}],"usage":{}}},"error":null}` + "\n"
	srv, _ := obServer(t, obResultsHandler(false, out, ""))
	got := obCollect(t, obBackend(ProviderOpenAI, srv).results(context.Background(), batchChat, "m", "b1"))
	if len(got["big"].resp.Text) != len(text) || got["big"].err != nil {
		t.Errorf("long line not parsed: %d", len(got["big"].resp.Text))
	}
}

type closeTrackTransport struct {
	base   http.RoundTripper
	opened atomic.Int32
	closed atomic.Int32
}

func (c *closeTrackTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	res, err := c.base.RoundTrip(r)
	if err == nil {
		c.opened.Add(1)
		res.Body = &closeTrackBody{res.Body, &c.closed}
	}
	return res, err
}

type closeTrackBody struct {
	io.ReadCloser
	n *atomic.Int32
}

func (b *closeTrackBody) Close() error { b.n.Add(1); return b.ReadCloser.Close() }

// Every response body is closed, also when the consumer stops early.
func TestOpenAIBatchBodiesClosed(t *testing.T) {
	srv, _ := obServer(t, obResultsHandler(true, obOutputFile, obErrorFile))
	be := obBackend(ProviderOpenAI, srv)
	tr := &closeTrackTransport{base: http.DefaultTransport}
	be.http = &http.Client{Transport: tr}
	for range be.results(context.Background(), batchChat, "m", "b1") {
		break
	}
	if tr.opened.Load() == 0 || tr.opened.Load() != tr.closed.Load() {
		t.Errorf("opened %d, closed %d", tr.opened.Load(), tr.closed.Load())
	}
}

// O9
func TestMoonshotBatch(t *testing.T) {
	srv, seen := obServer(t, obSubmitHandler)
	cfg := Config{
		MoonshotAPIKey: "mk", MoonshotBaseURL: srv.URL + "/v1",
		Profiles:   map[Profile]ModelRef{ProfileChat: {Provider: ProviderMoonshot, Model: "kimi-k2"}},
		EmbedModel: ModelRef{Provider: ProviderMoonshot, Model: "x"},
	}
	b := NewBatcher(cfg)
	if !b.Enabled() {
		t.Fatal("not enabled")
	}
	id, err := b.SubmitChat(context.Background(), ChatBatch{Items: []ChatItem{{CustomID: "a", Request: Request{Messages: []Message{{Role: "user", Content: "hi"}}}}}})
	if err != nil || !strings.HasPrefix(string(id), "moonshot:chat:") {
		t.Fatalf("id=%q err=%v", id, err)
	}
	for _, r := range seen() {
		if r.auth != "Bearer mk" {
			t.Errorf("auth = %q", r.auth)
		}
	}
	_, err = b.SubmitEmbed(context.Background(), EmbedBatch{Items: []EmbedItem{{CustomID: "a", Text: "t"}}})
	if !errors.Is(err, ErrBatchUnsupported) {
		t.Errorf("embed err = %v", err)
	}
	if got := newOpenAIBatch(ProviderMoonshot, "k", "", nil).limits(batchChat); got != moonshotBatchLimits {
		t.Errorf("limits = %+v", got)
	}
	if got := newOpenAIBatch(ProviderOpenAI, "k", "", nil).limits(batchChat); got.maxItems != 50000 || got.maxBytes != 200<<20 {
		t.Errorf("openai limits = %+v", got)
	}
}

// O10
func TestOpenAIBatchCancel(t *testing.T) {
	srv, seen := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) { _, _ = w.Write([]byte(`{"status":"cancelling"}`)) })
	if err := obBackend(ProviderOpenAI, srv).cancel(context.Background(), "b1"); err != nil {
		t.Fatal(err)
	}
	if r := seen()[0]; r.method != "POST" || r.path != "/v1/batches/b1/cancel" {
		t.Errorf("request = %+v", r)
	}
	srv, _ = obServer(t, func(w http.ResponseWriter, r *http.Request, n int) { w.WriteHeader(404) })
	if err := obBackend(ProviderOpenAI, srv).cancel(context.Background(), "b1"); err == nil {
		t.Error("404 should be an error")
	}
}

// O11
func TestOpenAIBatchThroughCore(t *testing.T) {
	embOut := `{"custom_id":"e1","response":{"status_code":200,"body":{"data":[{"embedding":[3,4]}],"usage":{"prompt_tokens":2}}},"error":null}` + "\n"
	for _, tc := range []struct {
		prefix string
		out    string
	}{{"openai:chat:", obOutputFile}, {"openai:embed:", embOut}} {
		srv, _ := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
			switch {
			case strings.HasSuffix(r.URL.Path, "/files"):
				_, _ = w.Write([]byte(`{"id":"fin"}`))
			case r.URL.Path == "/v1/batches":
				_, _ = w.Write([]byte(`{"id":"b1"}`))
			default:
				obResultsHandler(false, tc.out, "")(w, r, n)
			}
		})
		be := obBackend(ProviderOpenAI, srv)
		b := newBatcher(Config{
			Profiles:   map[Profile]ModelRef{ProfileChat: {Provider: ProviderOpenAI, Model: "gpt-4o"}},
			EmbedModel: ModelRef{Provider: ProviderOpenAI, Model: "text-embedding-3-small"},
		}, map[Provider]batchBackend{ProviderOpenAI: be})
		ctx := context.Background()
		var id BatchID
		var err error
		if strings.Contains(tc.prefix, "embed") {
			id, err = b.SubmitEmbed(ctx, EmbedBatch{Items: []EmbedItem{{CustomID: "e1", Text: "hello"}}})
		} else {
			id, err = b.SubmitChat(ctx, ChatBatch{Items: []ChatItem{{CustomID: "ok", Request: Request{Messages: []Message{{Role: "user", Content: "hi"}}}}}})
		}
		if err != nil || !strings.HasPrefix(string(id), tc.prefix) {
			t.Fatalf("id=%q err=%v", id, err)
		}
		res, err := CollectResults(b.Results(ctx, id))
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(tc.prefix, "embed") {
			v := res[0].Vector
			if len(v) != 2 || v[0] != 0.6 || v[1] != 0.8 {
				t.Errorf("vector = %v", v)
			}
		} else if res[0].Response.Text != "hello" || res[0].Response.Provider != ProviderOpenAI || res[0].Response.Model != "gpt-4o" {
			t.Errorf("chat result = %+v", res[0])
		}
	}
}

// NewBatcher: OpenAI only on its own URL.
func TestNewBatcherOpenAIBackends(t *testing.T) {
	if !NewBatcher(Config{OpenAIAPIKey: "k"}).Enabled() {
		t.Error("openai should be enabled")
	}
	if NewBatcher(Config{OpenAIAPIKey: "k", OpenAIBaseURL: "http://x"}).Enabled() {
		t.Error("openai with a custom base URL must not batch")
	}
	if !NewBatcher(Config{MoonshotAPIKey: "k"}).Enabled() {
		t.Error("moonshot should be enabled")
	}
}
