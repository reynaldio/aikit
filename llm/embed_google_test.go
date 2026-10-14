package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func newTestGoogleEmbed(url string) *googleEmbed {
	return &googleEmbed{apiKey: "k", baseURL: url, http: newProviderHTTPClient(nil)}
}

// googleServer records request bodies and replies with one vector per request.
func googleServer(t *testing.T, usage string) (*httptest.Server, func() []map[string]any, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var bodies []map[string]any
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var m map[string]any
		_ = json.NewDecoder(r.Body).Decode(&m)
		mu.Lock()
		bodies = append(bodies, m)
		paths = append(paths, r.URL.Path+"|"+r.Header.Get("x-goog-api-key"))
		mu.Unlock()
		reqs, _ := m["requests"].([]any)
		parts := make([]string, len(reqs))
		for i := range reqs {
			parts[i] = `{"values":[1]}`
		}
		fmt.Fprintf(w, `{"embeddings":[%s]%s}`, strings.Join(parts, ","), usage)
	}))
	t.Cleanup(srv.Close)
	return srv,
		func() []map[string]any { mu.Lock(); defer mu.Unlock(); return bodies },
		func() []string { mu.Lock(); defer mu.Unlock(); return paths }
}

func googleReqs(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	raw, _ := body["requests"].([]any)
	out := make([]map[string]any, len(raw))
	for i, r := range raw {
		out[i] = r.(map[string]any)
	}
	return out
}

func googleText(r map[string]any) string {
	return r["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"].(string)
}

func TestGoogleEmbedPathAndKey(t *testing.T) {
	srv, _, paths := googleServer(t, "")
	b := newTestGoogleEmbed(srv.URL)
	for _, m := range []string{"gemini-embedding-001", "models/gemini-embedding-001"} {
		if _, err := b.embed(context.Background(), m, []string{"a"}, "", 0); err != nil {
			t.Fatal(err)
		}
	}
	for _, p := range paths() {
		if p != "/models/gemini-embedding-001:batchEmbedContents|k" {
			t.Fatalf("path %q", p)
		}
	}
}

func TestGoogleEmbedTaskType(t *testing.T) {
	srv, bodies, _ := googleServer(t, "")
	b := newTestGoogleEmbed(srv.URL)
	for _, k := range []EmbedKind{EmbedQuery, EmbedDocument, ""} {
		if _, err := b.embed(context.Background(), "gemini-embedding-001", []string{"hi", "yo"}, k, 0); err != nil {
			t.Fatal(err)
		}
	}
	want := []string{"RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", ""}
	for i, body := range bodies() {
		for _, r := range googleReqs(t, body) {
			if r["model"] != "models/gemini-embedding-001" {
				t.Fatalf("model %v", r["model"])
			}
			if got := googleText(r); got != "hi" && got != "yo" {
				t.Fatalf("text changed: %q", got)
			}
			cfg, ok := r["embedContentConfig"].(map[string]any)
			if want[i] == "" {
				if ok {
					t.Fatalf("config present: %v", cfg)
				}
				continue
			}
			if cfg["taskType"] != want[i] {
				t.Fatalf("kind %d: %v", i, cfg)
			}
		}
	}
}

func TestGoogleEmbedPrefixModels(t *testing.T) {
	srv, bodies, _ := googleServer(t, "")
	b := newTestGoogleEmbed(srv.URL)
	for _, k := range []EmbedKind{EmbedQuery, EmbedDocument, ""} {
		if _, err := b.embed(context.Background(), "gemini-embedding-2", []string{"hi"}, k, 0); err != nil {
			t.Fatal(err)
		}
	}
	want := []string{"task: search result | query: hi", "title: none | text: hi", "hi"}
	for i, body := range bodies() {
		r := googleReqs(t, body)[0]
		if got := googleText(r); got != want[i] {
			t.Fatalf("kind %d: %q", i, got)
		}
		if _, ok := r["embedContentConfig"]; ok {
			t.Fatalf("config present for kind %d", i)
		}
	}
}

func TestGoogleEmbedDimensions(t *testing.T) {
	srv, bodies, _ := googleServer(t, "")
	b := newTestGoogleEmbed(srv.URL)
	for _, d := range []int{768, 0} {
		if _, err := b.embed(context.Background(), "gemini-embedding-001", []string{"a"}, EmbedQuery, d); err != nil {
			t.Fatal(err)
		}
	}
	cfg0 := googleReqs(t, bodies()[0])[0]["embedContentConfig"].(map[string]any)
	if cfg0["outputDimensionality"] != float64(768) {
		t.Fatalf("cfg %v", cfg0)
	}
	cfg1 := googleReqs(t, bodies()[1])[0]["embedContentConfig"].(map[string]any)
	if _, ok := cfg1["outputDimensionality"]; ok {
		t.Fatalf("cfg %v", cfg1)
	}
	// dims alone (no kind) still sends the config.
	if _, err := b.embed(context.Background(), "gemini-embedding-2", []string{"a"}, "", 256); err != nil {
		t.Fatal(err)
	}
	cfg2 := googleReqs(t, bodies()[2])[0]["embedContentConfig"].(map[string]any)
	if cfg2["outputDimensionality"] != float64(256) {
		t.Fatalf("cfg %v", cfg2)
	}
}

func TestGoogleEmbedTokens(t *testing.T) {
	srv, _, _ := googleServer(t, `,"usageMetadata":{"promptTokenCount":12}`)
	c, err := newTestGoogleEmbed(srv.URL).embed(context.Background(), "gemini-embedding-001", []string{"a"}, "", 0)
	if err != nil || c.tokens != 12 || c.estimated {
		t.Fatalf("%+v %v", c, err)
	}
	srv2, _, _ := googleServer(t, "")
	c, err = newTestGoogleEmbed(srv2.URL).embed(context.Background(), "gemini-embedding-2", []string{"abcd"}, EmbedQuery, 0)
	want := estimateTokens("task: search result | query: abcd")
	if err != nil || c.tokens != want || !c.estimated {
		t.Fatalf("%+v %v want %d", c, err, want)
	}
}

func TestGoogleEmbedCountMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"embeddings":[{"values":[1]},{"values":[1]},{"values":[1]}],"usageMetadata":{"promptTokenCount":5}}`)
	}))
	defer srv.Close()
	c, err := newTestGoogleEmbed(srv.URL).embed(context.Background(), "gemini-embedding-001", []string{"a", "b"}, "", 0)
	if err == nil || !strings.Contains(err.Error(), "got 3 embeddings for 2 texts") {
		t.Fatalf("err %v", err)
	}
	if c.tokens != 5 {
		t.Fatalf("tokens %d", c.tokens)
	}
}

func TestGoogleEmbedHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("retry-after-ms", "0")
		w.WriteHeader(429)
		fmt.Fprint(w, `{"error":{"message":"quota gone"}}`)
	}))
	defer srv.Close()
	_, err := newTestGoogleEmbed(srv.URL).embed(context.Background(), "gemini-embedding-001", []string{"a"}, "", 0)
	if err == nil || !strings.Contains(err.Error(), "quota gone") || !strings.Contains(err.Error(), "status 429") {
		t.Fatalf("err %v", err)
	}
}

func TestGeminiEmbedUsesTaskType(t *testing.T) {
	for m, want := range map[string]bool{
		"gemini-embedding-001":        true,
		"models/gemini-embedding-001": true,
		"text-embedding-004":          true,
		"embedding-001":               true,
		"gemini-embedding-2":          false,
		"gemini-embedding-2-preview":  false,
	} {
		if got := geminiEmbedUsesTaskType(m); got != want {
			t.Errorf("%s: %v", m, got)
		}
	}
}

func TestGoogleEmbedChunksThroughCore(t *testing.T) {
	var mu sync.Mutex
	var sizes []int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var m struct {
			Requests []struct {
				Content struct {
					Parts []struct{ Text string } `json:"parts"`
				} `json:"content"`
			} `json:"requests"`
		}
		_ = json.NewDecoder(r.Body).Decode(&m)
		mu.Lock()
		sizes = append(sizes, len(m.Requests))
		mu.Unlock()
		parts := make([]string, len(m.Requests))
		for i, q := range m.Requests {
			parts[i] = fmt.Sprintf(`{"values":[%s,1]}`, q.Content.Parts[0].Text)
		}
		fmt.Fprintf(w, `{"embeddings":[%s]}`, strings.Join(parts, ","))
	}))
	defer srv.Close()
	e := &embedder{
		backends: map[Provider]embedBackend{ProviderGoogle: newTestGoogleEmbed(srv.URL)},
		model:    ModelRef{ProviderGoogle, "gemini-embedding-001"},
	}
	texts := make([]string, 250)
	for i := range texts {
		texts[i] = fmt.Sprint(i + 1)
	}
	res, err := e.Embed(context.Background(), EmbedRequest{Texts: texts})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Vectors) != 250 {
		t.Fatalf("vectors %d", len(res.Vectors))
	}
	// Normalized [n,1] grows with n, so a growing first value means order is kept.
	for i := 1; i < len(res.Vectors); i++ {
		if res.Vectors[i][0] <= res.Vectors[i-1][0] {
			t.Fatalf("order broken at %d", i)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(sizes) != 3 || sizes[0]+sizes[1]+sizes[2] != 250 {
		t.Fatalf("sizes %v", sizes)
	}
	for _, n := range sizes {
		if n != 100 && n != 50 {
			t.Fatalf("sizes %v", sizes)
		}
	}
}

func TestGoogleEmbedBadJSONIsBilled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{not json`)
	}))
	defer srv.Close()
	c, err := newTestGoogleEmbed(srv.URL).embed(context.Background(), "gemini-embedding-001", []string{"abcdefgh", "ij"}, "", 0)
	if err == nil || !strings.Contains(err.Error(), "decode embeddings") {
		t.Fatalf("err %v", err)
	}
	if want := estimateTokens("abcdefghij"); c.tokens != want || !c.estimated {
		t.Fatalf("%+v want %d", c, want)
	}
}

func TestGoogleEmbedEscapesModelPath(t *testing.T) { // G-esc3
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.URL.EscapedPath()
		fmt.Fprint(w, `{"embeddings":[{"values":[1]}]}`)
	}))
	t.Cleanup(srv.Close)
	b := newTestGoogleEmbed(srv.URL)
	if _, err := b.embed(context.Background(), "tuned/x y", []string{"a"}, "", 0); err != nil {
		t.Fatal(err)
	}
	if got != "/models/tuned%2Fx%20y:batchEmbedContents" {
		t.Errorf("path = %q", got)
	}
}
