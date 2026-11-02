package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

func gbBackend(srv *httptest.Server) *googleBatch {
	be := newGoogleBatch("gkey", nil).(*googleBatch)
	be.baseURL = srv.URL + "/v1beta"
	be.uploadURL = srv.URL + "/upload/v1beta"
	be.downloadURL = srv.URL + "/download/v1beta"
	return be
}

// gbSubmitHandler answers the two upload steps and the create call.
func gbSubmitHandler(createBody string) func(http.ResponseWriter, *http.Request, int) {
	return func(w http.ResponseWriter, r *http.Request, n int) {
		switch {
		case r.URL.Path == "/upload/v1beta/files":
			w.Header().Set("X-Goog-Upload-URL", "http://"+r.Host+"/upload-target")
		case r.URL.Path == "/upload-target":
			_, _ = w.Write([]byte(`{"file":{"name":"files/f1"}}`))
		default:
			_, _ = w.Write([]byte(createBody))
		}
	}
}

// GB2
func TestGeminiBatchThinking(t *testing.T) {
	for _, tc := range []struct {
		model  string
		level  string
		budget *int
		none   bool
	}{
		{"gemini-3.1-flash-lite", "minimal", nil, false},
		{"gemini-2.5-flash", "", new(int), false},
		{"gemini-3.1-pro-preview", "", nil, true},
		{"gemini-10-flash", "minimal", nil, false},
		{"gemini-flash-latest", "", nil, true},
	} {
		got := geminiBatchThinking(tc.model)
		if tc.none {
			if got != nil {
				t.Errorf("%s: want nil, got %+v", tc.model, got)
			}
			continue
		}
		if got == nil || got.ThinkingLevel != tc.level || (tc.budget == nil) != (got.ThinkingBudget == nil) || (tc.budget != nil && *got.ThinkingBudget != 0) {
			t.Errorf("%s: got %+v", tc.model, got)
		}
	}
}

// GB3
func TestGoogleBatchUpload(t *testing.T) {
	srv, seen := obServer(t, gbSubmitHandler(`{"name":"batches/abc"}`))
	be := gbBackend(srv)
	body, err := be.buildChat("gemini-3.1-flash-lite", 77, Request{SystemCacheable: "sys", Messages: []Message{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	id, err := be.submit(context.Background(), batchChat, "gemini-3.1-flash-lite", []batchLine{{"a", body}, {"b", body}})
	if err != nil || id != "batches/abc" {
		t.Fatalf("id=%q err=%v", id, err)
	}
	reqs := seen()
	if len(reqs) != 3 {
		t.Fatalf("requests = %d", len(reqs))
	}
	start, up := reqs[0], reqs[1]
	if start.path != "/upload/v1beta/files" || start.method != "POST" {
		t.Errorf("start = %+v", start)
	}
	if !strings.Contains(string(start.body), `"display_name":"aikit-batch"`) {
		t.Errorf("start body %s", start.body)
	}
	if up.path != "/upload-target" {
		t.Errorf("upload path %q", up.path)
	}
	lines := strings.Split(strings.TrimSpace(string(up.body)), "\n")
	if len(lines) != 2 {
		t.Fatalf("lines = %d", len(lines))
	}
	var l struct {
		Key     string `json:"key"`
		Request struct {
			Contents          []any `json:"contents"`
			SystemInstruction any   `json:"system_instruction"`
			GenerationConfig  struct {
				MaxOutputTokens int `json:"maxOutputTokens"`
			} `json:"generationConfig"`
		} `json:"request"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &l); err != nil {
		t.Fatal(err)
	}
	if l.Key != "a" || len(l.Request.Contents) != 1 || l.Request.SystemInstruction == nil || l.Request.GenerationConfig.MaxOutputTokens != 77 {
		t.Errorf("line = %s", lines[0])
	}
}

// Header checks need the raw headers, so they get their own server.
func TestGoogleBatchUploadHeaders(t *testing.T) {
	var startH, upH http.Header
	var upBodyLen int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/upload/v1beta/files":
			startH = r.Header.Clone()
			w.Header().Set("X-Goog-Upload-URL", "http://"+r.Host+"/up")
		case "/up":
			upH = r.Header.Clone()
			b, _ := io.ReadAll(r.Body)
			upBodyLen = len(b)
			_, _ = w.Write([]byte(`{"file":{"name":"files/f1"}}`))
		default:
			_, _ = w.Write([]byte(`{"name":"batches/abc"}`))
		}
	}))
	defer srv.Close()
	be := gbBackend(srv)
	if _, err := be.submit(context.Background(), batchChat, "m", []batchLine{{"a", []byte(`{}`)}}); err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"X-Goog-Api-Key":                      "gkey",
		"X-Goog-Upload-Protocol":              "resumable",
		"X-Goog-Upload-Command":               "start",
		"X-Goog-Upload-Header-Content-Type":   "application/jsonl",
		"Content-Type":                        "application/json",
		"X-Goog-Upload-Header-Content-Length": "",
	}
	for k, v := range want {
		got := startH.Get(k)
		if got == "" || (v != "" && got != v) {
			t.Errorf("start header %s = %q", k, got)
		}
	}
	if upH.Get("X-Goog-Upload-Command") != "upload, finalize" || upH.Get("X-Goog-Upload-Offset") != "0" {
		t.Errorf("upload headers %v", upH)
	}
	if startH.Get("X-Goog-Upload-Header-Content-Length") == "" || upBodyLen == 0 {
		t.Errorf("length %q body %d", startH.Get("X-Goog-Upload-Header-Content-Length"), upBodyLen)
	}
}

// GB4
func TestGoogleBatchEmbedLines(t *testing.T) {
	be := newGoogleBatch("k", nil).(*googleBatch)
	// The line is the live item: top-level taskType and outputDimensionality, no
	// embedContentConfig.
	parse := func(model, text string, kind EmbedKind, dims int) map[string]any {
		b, err := be.buildEmbed(model, text, kind, dims)
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatal(err)
		}
		if _, ok := m["embedContentConfig"]; ok {
			t.Errorf("embedContentConfig sent: %s", b)
		}
		return m
	}
	m := parse("gemini-embedding-001", "hello", EmbedQuery, 0)
	if m["taskType"] != "RETRIEVAL_QUERY" || m["outputDimensionality"] != nil || m["model"] != "models/gemini-embedding-001" {
		t.Errorf("001 query: %v", m)
	}
	if m = parse("gemini-embedding-001", "hello", EmbedDocument, 768); m["taskType"] != "RETRIEVAL_DOCUMENT" || m["outputDimensionality"] != float64(768) {
		t.Errorf("001 document 768: %v", m)
	}
	if m = parse("gemini-embedding-001", "hello", "", 0); m["taskType"] != nil || m["outputDimensionality"] != nil {
		t.Errorf("empty kind: %v", m)
	}
	if m = parse("gemini-embedding-001", "hello", "", 256); m["outputDimensionality"] != float64(256) || m["taskType"] != nil {
		t.Errorf("dims: %v", m)
	}
	m = parse("gemini-embedding-2", "hello", EmbedQuery, 0)
	text := m["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"]
	if text != "task: search result | query: hello" || m["taskType"] != nil || m["outputDimensionality"] != nil {
		t.Errorf("embedding-2: %v", m)
	}
	if m = parse("gemini-embedding-2", "hello", EmbedQuery, 128); m["outputDimensionality"] != float64(128) || m["taskType"] != nil {
		t.Errorf("embedding-2 dims: %v", m)
	}
}

// The embed file keys carry a token guess: <id>.<n>. Chat keys stay plain.
func TestGoogleBatchSubmitKeys(t *testing.T) {
	keys := func(kind batchKind, model string, bodies map[string][]byte) map[string]string {
		srv, seen := obServer(t, gbSubmitHandler(`{"name":"batches/abc"}`))
		var lines []batchLine
		for _, id := range []string{"e1", "e2"} {
			lines = append(lines, batchLine{id, bodies[id]})
		}
		if _, err := gbBackend(srv).submit(context.Background(), kind, model, lines); err != nil {
			t.Fatal(err)
		}
		out := map[string]string{}
		for _, l := range strings.Split(strings.TrimSpace(string(seen()[1].body)), "\n") {
			var x struct {
				Key string `json:"key"`
			}
			if err := json.Unmarshal([]byte(l), &x); err != nil {
				t.Fatal(err)
			}
			out[x.Key] = l
		}
		return out
	}
	be := newGoogleBatch("k", nil).(*googleBatch)
	long := strings.Repeat("abcd", 10)
	for _, model := range []string{"gemini-embedding-001", "gemini-embedding-2"} {
		b1, _ := be.buildEmbed(model, "hello", EmbedQuery, 0)
		b2, _ := be.buildEmbed(model, long, EmbedDocument, 0)
		got := keys(batchEmbed, model, map[string][]byte{"e1": b1, "e2": b2})
		// The guess uses the text in the line, so embedding-2 counts its prefix.
		t1, t2 := "hello", long
		if model == "gemini-embedding-2" {
			t1, t2 = "task: search result | query: hello", "title: none | text: "+long
		}
		k1, k2 := "e1."+strconv.Itoa(estimateTokens(t1)), "e2."+strconv.Itoa(estimateTokens(t2))
		if _, ok := got[k1]; !ok || len(got) != 2 {
			t.Errorf("%s: keys = %v, want %s", model, got, k1)
		}
		if _, ok := got[k2]; !ok {
			t.Errorf("%s: keys = %v, want %s", model, got, k2)
		}
	}
	// A body that cannot be decoded keeps the plain key.
	got := keys(batchEmbed, "gemini-embedding-001", map[string][]byte{"e1": []byte(`"x"`), "e2": []byte(`{"content":{"parts":[{"text":"abcd"},{"text":"efgh"}]}}`)})
	if _, ok := got["e1"]; !ok {
		t.Errorf("bad body keys = %v", got)
	}
	if _, ok := got["e2.2"]; !ok {
		t.Errorf("joined parts keys = %v", got)
	}
	// Chat keys stay plain.
	cb, _ := be.buildChat("gemini-3.1-flash-lite", 10, Request{Messages: []Message{{Role: "user", Content: "hi"}}})
	got = keys(batchChat, "gemini-3.1-flash-lite", map[string][]byte{"e1": cb, "e2": cb})
	if _, ok := got["e1"]; !ok || len(got) != 2 {
		t.Errorf("chat keys = %v", got)
	}
}

// GB5
func TestGoogleBatchCreate(t *testing.T) {
	for _, tc := range []struct {
		kind  batchKind
		reply string
		path  string
	}{
		{batchChat, `{"name":"operations/x","metadata":{"name":"batches/m1"}}`, "/v1beta/models/gemini-3.1-flash-lite:batchGenerateContent"},
		{batchChat, `{"name":"batches/n1"}`, "/v1beta/models/gemini-3.1-flash-lite:batchGenerateContent"},
		{batchEmbed, `{"name":"batches/e1"}`, "/v1beta/models/gemini-3.1-flash-lite:asyncBatchEmbedContent"},
	} {
		srv, seen := obServer(t, gbSubmitHandler(tc.reply))
		be := gbBackend(srv)
		id, err := be.submit(context.Background(), tc.kind, "gemini-3.1-flash-lite", []batchLine{{"a", []byte(`{}`)}})
		if err != nil || !strings.HasPrefix(id, "batches/") {
			t.Fatalf("id=%q err=%v", id, err)
		}
		reqs := seen()
		c := reqs[len(reqs)-1]
		if c.path != tc.path {
			t.Errorf("create path %q want %q", c.path, tc.path)
		}
		var b struct {
			Batch struct {
				DisplayName string `json:"displayName"`
				Model       string `json:"model"`
				InputConfig struct {
					FileName string `json:"fileName"`
				} `json:"inputConfig"`
			} `json:"batch"`
		}
		if err := json.Unmarshal(c.body, &b); err != nil {
			t.Fatal(err)
		}
		if b.Batch.Model != "models/gemini-3.1-flash-lite" || b.Batch.InputConfig.FileName != "files/f1" || b.Batch.DisplayName != "aikit-batch" {
			t.Errorf("create body %s", c.body)
		}
	}
}

// GB6
func TestGoogleBatchCreate503NotRetried(t *testing.T) {
	srv, seen := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		if strings.HasPrefix(r.URL.Path, "/upload") {
			gbSubmitHandler("")(w, r, n)
			return
		}
		w.Header().Set("retry-after-ms", "1")
		w.WriteHeader(503)
		_, _ = w.Write([]byte(`{"error":{"message":"down"}}`))
	})
	be := gbBackend(srv)
	_, err := be.submit(context.Background(), batchChat, "m", []batchLine{{"a", []byte(`{}`)}})
	if err == nil || !strings.Contains(err.Error(), "may still have been created") {
		t.Fatalf("err = %v", err)
	}
	creates := 0
	for _, r := range seen() {
		if strings.HasSuffix(r.path, ":batchGenerateContent") {
			creates++
		}
	}
	if creates != 1 {
		t.Errorf("create requests = %d, want 1", creates)
	}
}

func TestGoogleBatchCreateBadReplies(t *testing.T) {
	for _, reply := range []string{`not json`, `{}`, `{"name":"operations/x"}`} {
		srv, _ := obServer(t, gbSubmitHandler(reply))
		be := gbBackend(srv)
		_, err := be.submit(context.Background(), batchChat, "m", []batchLine{{"a", []byte(`{}`)}})
		if err == nil || !strings.Contains(err.Error(), "may still have been created") {
			t.Errorf("reply %q: err = %v", reply, err)
		}
	}
}

func TestGoogleBatchUpload503ThenOK(t *testing.T) {
	startCalls := 0
	srv, _ := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		if r.URL.Path == "/upload/v1beta/files" {
			startCalls++
			if startCalls == 1 {
				w.Header().Set("retry-after-ms", "1")
				w.WriteHeader(503)
				return
			}
		}
		gbSubmitHandler(`{"name":"batches/abc"}`)(w, r, n)
	})
	be := gbBackend(srv)
	id, err := be.submit(context.Background(), batchChat, "m", []batchLine{{"a", []byte(`{}`)}})
	if err != nil || id != "batches/abc" || startCalls != 2 {
		t.Fatalf("id=%q err=%v calls=%d", id, err, startCalls)
	}
}

func TestGoogleBatchFinalize503ThenOK(t *testing.T) {
	calls := 0
	srv, _ := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		if r.URL.Path == "/upload-target" {
			calls++
			if calls == 1 {
				w.Header().Set("retry-after-ms", "1")
				w.WriteHeader(503)
				return
			}
		}
		gbSubmitHandler(`{"name":"batches/abc"}`)(w, r, n)
	})
	be := gbBackend(srv)
	if _, err := be.submit(context.Background(), batchChat, "m", []batchLine{{"a", []byte(`{}`)}}); err != nil || calls != 2 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}

func TestGoogleBatchStatus503ThenOK(t *testing.T) {
	calls := 0
	srv, _ := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		calls++
		if calls == 1 {
			w.Header().Set("retry-after-ms", "1")
			w.WriteHeader(503)
			return
		}
		_, _ = w.Write([]byte(`{"state":"JOB_STATE_RUNNING"}`))
	})
	st, err := gbBackend(srv).status(context.Background(), "batches/abc")
	if err != nil || st.State != BatchRunning || calls != 2 {
		t.Fatalf("st=%+v err=%v calls=%d", st, err, calls)
	}
}

// GB7
func TestGoogleBatchStatusMapping(t *testing.T) {
	for _, tc := range []struct {
		name, reply string
		state       BatchState
		errText     string
	}{
		{"running", `{"state":"JOB_STATE_RUNNING"}`, BatchRunning, ""},
		{"pending", `{"state":"BATCH_STATE_PENDING"}`, BatchRunning, ""},
		{"unspec", `{"state":"JOB_STATE_UNSPECIFIED"}`, BatchRunning, ""},
		{"ok", `{"state":"BATCH_STATE_SUCCEEDED","batchStats":{"requestCount":"3","successfulRequestCount":"2","failedRequestCount":"1"}}`, BatchDone, ""},
		{"ok-meta", `{"name":"batches/x","metadata":{"state":"JOB_STATE_SUCCEEDED","batchStats":{"requestCount":3,"successfulRequestCount":3}}}`, BatchDone, ""},
		{"failed", `{"state":"JOB_STATE_FAILED","error":{"message":"bad input"}}`, BatchFailed, "bad input"},
		{"cancelled", `{"state":"BATCH_STATE_CANCELLED"}`, BatchDone, ""},
		{"expired", `{"state":"JOB_STATE_EXPIRED"}`, BatchDone, ""},
	} {
		srv, seen := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) { _, _ = w.Write([]byte(tc.reply)) })
		st, err := gbBackend(srv).status(context.Background(), "batches/abc")
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if st.State != tc.state {
			t.Errorf("%s: state %q", tc.name, st.State)
		}
		if tc.errText != "" && (st.Err == nil || !strings.Contains(st.Err.Error(), tc.errText)) {
			t.Errorf("%s: err %v", tc.name, st.Err)
		}
		if p := seen()[0].path; p != "/v1beta/batches/abc" {
			t.Errorf("path %q", p)
		}
		if tc.name == "ok" && (st.Total != 3 || st.Succeeded != 2 || st.Failed != 1) {
			t.Errorf("counts %+v", st)
		}
		if tc.name == "ok-meta" && (st.Total != 3 || st.Succeeded != 3) {
			t.Errorf("counts %+v", st)
		}
	}
}

const gbOutput = `{"key":"ok","response":{"candidates":[{"content":{"parts":[{"text":"hello"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3}}}
{"key":"fn","response":{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get","args":{"x":1}}}]},"finishReason":"STOP"}]}}
{"key":"blocked","response":{"candidates":[{"content":{"parts":[]},"finishReason":"SAFETY"}],"usageMetadata":{"promptTokenCount":4}}}
{"key":"bad","error":{"code":400,"message":"nope"}}
`

const gbBatchWithFile = `{"state":"JOB_STATE_SUCCEEDED","output":{"responsesFile":"files/out1"}}`

func gbResultsHandler(batch, out string) func(http.ResponseWriter, *http.Request, int) {
	return func(w http.ResponseWriter, r *http.Request, n int) {
		if strings.HasPrefix(r.URL.Path, "/download/") {
			_, _ = w.Write([]byte(out))
			return
		}
		_, _ = w.Write([]byte(batch))
	}
}

// GB8
func TestGoogleBatchResults(t *testing.T) {
	for _, batch := range []string{
		gbBatchWithFile,
		`{"state":"JOB_STATE_SUCCEEDED","dest":{"fileName":"files/out1"}}`,
		`{"state":"JOB_STATE_SUCCEEDED","output":{"fileName":"files/out1"}}`,
		`{"name":"batches/abc","metadata":{"output":{"responsesFile":"files/out1"}}}`,
	} {
		srv, seen := obServer(t, gbResultsHandler(batch, gbOutput))
		got := obCollect(t, gbBackend(srv).results(context.Background(), batchChat, "gemini-3.1-flash-lite", "batches/abc"))
		if got["ok"].resp.Text != "hello" || got["ok"].resp.InputTokens != 10 || got["ok"].resp.OutputTokens != 3 || got["ok"].err != nil {
			t.Errorf("%s: ok = %+v", batch, got["ok"])
		}
		if len(got["fn"].resp.ToolCalls) != 1 || got["fn"].resp.ToolCalls[0].Name != "get" {
			t.Errorf("fn = %+v", got["fn"])
		}
		var ref *RefusalError
		if !errors.As(got["blocked"].err, &ref) {
			t.Errorf("blocked err = %v", got["blocked"].err)
		}
		if e := got["bad"].err; e == nil || !strings.Contains(e.Error(), "nope") || !strings.Contains(e.Error(), "400") {
			t.Errorf("bad err = %v", e)
		}
		var dl obReq
		for _, r := range seen() {
			if strings.HasPrefix(r.path, "/download/") {
				dl = r
			}
		}
		if dl.path != "/download/v1beta/files/out1:download" {
			t.Errorf("download path %q", dl.path)
		}
	}
}

func TestGoogleBatchDownloadQuery(t *testing.T) {
	var q string
	srv, _ := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		if strings.HasPrefix(r.URL.Path, "/download/") {
			q = r.URL.RawQuery
		}
		gbResultsHandler(gbBatchWithFile, "")(w, r, n)
	})
	obCollect(t, gbBackend(srv).results(context.Background(), batchChat, "m", "batches/abc"))
	if q != "alt=media" {
		t.Errorf("query %q", q)
	}
}

func TestGoogleBatchResultsEmbedAndNoFile(t *testing.T) {
	out := `{"key":"e1","response":{"embedding":{"values":[3,4]},"usageMetadata":{"promptTokenCount":5}}}` + "\n"
	srv, _ := obServer(t, gbResultsHandler(gbBatchWithFile, out))
	got := obCollect(t, gbBackend(srv).results(context.Background(), batchEmbed, "m", "batches/abc"))
	if v := got["e1"].vector; len(v) != 2 || v[0] != 3 || got["e1"].resp.InputTokens != 5 {
		t.Errorf("e1 = %+v", got["e1"])
	}
	// Token guess in the key.
	out = `{"key":"e1.4","response":{"embedding":{"values":[1]}}}` + "\n" +
		`{"key":"e2.9","response":{"embedding":{"values":[1]},"usageMetadata":{"promptTokenCount":5}}}` + "\n" +
		`{"key":"e3","response":{"embedding":{"values":[1]}}}` + "\n"
	srv3, _ := obServer(t, gbResultsHandler(gbBatchWithFile, out))
	got = obCollect(t, gbBackend(srv3).results(context.Background(), batchEmbed, "m", "batches/abc"))
	if r := got["e1"].resp; r.InputTokens != 4 || !r.UsageEstimated {
		t.Errorf("guess: %+v", got)
	}
	if r := got["e2"].resp; r.InputTokens != 5 || r.UsageEstimated {
		t.Errorf("usage wins: %+v", got["e2"])
	}
	if r := got["e3"].resp; r.InputTokens != 0 || r.UsageEstimated {
		t.Errorf("no suffix: %+v", got["e3"])
	}
	srv2, _ := obServer(t, gbResultsHandler(`{"state":"JOB_STATE_SUCCEEDED"}`, ""))
	n := 0
	for _, err := range gbBackend(srv2).results(context.Background(), batchChat, "m", "batches/abc") {
		if err != nil {
			t.Errorf("err = %v", err)
		}
		n++
	}
	if n != 0 {
		t.Errorf("want no results, got %d", n)
	}
}

// GB9
func TestGoogleBatchLongLine(t *testing.T) {
	big := strings.Repeat("x", 300_000)
	out := `{"key":"big","response":{"candidates":[{"content":{"parts":[{"text":"` + big + `"}]},"finishReason":"STOP"}]}}` + "\n"
	srv, _ := obServer(t, gbResultsHandler(gbBatchWithFile, out))
	got := obCollect(t, gbBackend(srv).results(context.Background(), batchChat, "m", "batches/abc"))
	if len(got["big"].resp.Text) != 300_000 {
		t.Errorf("long line: %d", len(got["big"].resp.Text))
	}
}

func TestGoogleBatchBodiesClosed(t *testing.T) {
	srv, _ := obServer(t, gbResultsHandler(gbBatchWithFile, gbOutput))
	be := gbBackend(srv)
	tr := &closeTrackTransport{base: http.DefaultTransport}
	be.http = &http.Client{Transport: tr}
	for range be.results(context.Background(), batchChat, "m", "batches/abc") {
		break
	}
	if tr.opened.Load() == 0 || tr.opened.Load() != tr.closed.Load() {
		t.Errorf("opened %d, closed %d", tr.opened.Load(), tr.closed.Load())
	}
}

// GB10
func TestGoogleBatchCancel(t *testing.T) {
	srv, seen := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) { _, _ = w.Write([]byte(`{}`)) })
	if err := gbBackend(srv).cancel(context.Background(), "batches/abc"); err != nil {
		t.Fatal(err)
	}
	r := seen()[0]
	if r.method != "POST" || r.path != "/v1beta/batches/abc:cancel" {
		t.Errorf("cancel = %s %s", r.method, r.path)
	}
}

func TestGoogleBatchSupportsLimits(t *testing.T) {
	be := newGoogleBatch("k", nil)
	for _, k := range []batchKind{batchChat, batchChatJSON, batchEmbed} {
		if !be.supports(k) {
			t.Errorf("supports %s", k)
		}
	}
	if l := be.limits(batchChat); l.maxItems != 0 || l.maxBytes != 2<<30 {
		t.Errorf("limits %+v", l)
	}
}

// GB11
func TestGoogleBatchThroughCore(t *testing.T) {
	srv, _ := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/upload") || r.URL.Path == "/upload-target":
			gbSubmitHandler("")(w, r, n)
		case strings.HasSuffix(r.URL.Path, ":batchGenerateContent"):
			_, _ = w.Write([]byte(`{"name":"batches/abc"}`))
		default:
			gbResultsHandler(gbBatchWithFile, gbOutput)(w, r, n)
		}
	})
	b := newBatcher(Config{
		Profiles: map[Profile]ModelRef{ProfileChat: {Provider: ProviderGoogle, Model: "gemini-3.1-flash-lite"}},
	}, map[Provider]batchBackend{ProviderGoogle: gbBackend(srv)})
	ctx := context.Background()
	id, err := b.SubmitChat(ctx, ChatBatch{Items: []ChatItem{{CustomID: "ok", Request: Request{Messages: []Message{{Role: "user", Content: "hi"}}}}}})
	if err != nil || string(id) != "google:chat:gemini-3.1-flash-lite:batches/abc" {
		t.Fatalf("id=%q err=%v", id, err)
	}
	if id.Provider() != ProviderGoogle || id.Model() != "gemini-3.1-flash-lite" {
		t.Errorf("parse back: %q %q", id.Provider(), id.Model())
	}
	res, err := CollectResults(b.Results(ctx, id))
	if err != nil || len(res) != 4 {
		t.Fatalf("res=%d err=%v", len(res), err)
	}
	if res[0].CustomID != "ok" || res[0].Response.Text != "hello" || res[0].Response.Provider != ProviderGoogle {
		t.Errorf("res[0] = %+v", res[0])
	}
}

func TestGoogleBatchEmbedThroughCore(t *testing.T) {
	out := `{"key":"e-1.4","response":{"embedding":{"values":[3,4]}}}` + "\n"
	srv, _ := obServer(t, func(w http.ResponseWriter, r *http.Request, n int) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/upload") || r.URL.Path == "/upload-target":
			gbSubmitHandler("")(w, r, n)
		case strings.HasSuffix(r.URL.Path, ":asyncBatchEmbedContent"):
			_, _ = w.Write([]byte(`{"name":"batches/abc"}`))
		default:
			gbResultsHandler(gbBatchWithFile, out)(w, r, n)
		}
	})
	b := newBatcher(Config{
		EmbedModel: ModelRef{Provider: ProviderGoogle, Model: "gemini-embedding-001"},
	}, map[Provider]batchBackend{ProviderGoogle: gbBackend(srv)})
	ctx := context.Background()
	id, err := b.SubmitEmbed(ctx, EmbedBatch{Items: []EmbedItem{{CustomID: "e-1", Text: "hello"}}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := CollectResults(b.Results(ctx, id))
	if err != nil || len(res) != 1 {
		t.Fatalf("res=%v err=%v", res, err)
	}
	if res[0].CustomID != "e-1" || res[0].Response.InputTokens != 4 || !res[0].Response.UsageEstimated {
		t.Errorf("res[0] = %+v", res[0])
	}
}
