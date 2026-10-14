package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func newTestOpenAIEmbed(url string) *openaiEmbed {
	return &openaiEmbed{apiKey: "k", baseURL: url, http: newProviderHTTPClient(nil)}
}

func TestOpenAIEmbedBody(t *testing.T) {
	var gotPath, gotAuth string
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotAuth = r.URL.Path, r.Header.Get("Authorization")
		var m map[string]any
		_ = json.NewDecoder(r.Body).Decode(&m)
		bodies = append(bodies, m)
		fmt.Fprint(w, `{"data":[{"index":0,"embedding":[1]},{"index":1,"embedding":[2]}],"usage":{"prompt_tokens":3}}`)
	}))
	defer srv.Close()
	b := newTestOpenAIEmbed(srv.URL)
	for _, dims := range []int{0, 256} {
		if _, err := b.embed(context.Background(), "m", []string{"x", "y"}, EmbedQuery, dims); err != nil {
			t.Fatal(err)
		}
	}
	if gotPath != "/embeddings" || gotAuth != "Bearer k" {
		t.Fatalf("path %q auth %q", gotPath, gotAuth)
	}
	m := bodies[0]
	in, _ := m["input"].([]any)
	if m["model"] != "m" || m["encoding_format"] != "float" || len(in) != 2 || in[0] != "x" || in[1] != "y" {
		t.Fatalf("body %v", m)
	}
	if _, ok := m["dimensions"]; ok {
		t.Fatal("dimensions sent for 0")
	}
	if bodies[1]["dimensions"] != float64(256) {
		t.Fatalf("dimensions %v", bodies[1]["dimensions"])
	}
}

func serveOpenAIEmbed(t *testing.T, body string, status int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestOpenAIEmbedReverseOrder(t *testing.T) {
	srv := serveOpenAIEmbed(t, `{"data":[{"index":1,"embedding":[2]},{"index":0,"embedding":[1]}],"usage":{"prompt_tokens":2}}`, 200)
	c, err := newTestOpenAIEmbed(srv.URL).embed(context.Background(), "m", []string{"a", "b"}, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if c.vectors[0][0] != 1 || c.vectors[1][0] != 2 || c.tokens != 2 || c.estimated {
		t.Fatalf("%+v", c)
	}
}

func TestOpenAIEmbedBadIndex(t *testing.T) {
	for _, body := range []string{
		`{"data":[{"index":5,"embedding":[1]},{"index":0,"embedding":[1]}],"usage":{"prompt_tokens":2}}`,
		`{"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[1]}],"usage":{"prompt_tokens":2}}`,
		`{"data":[{"index":-1,"embedding":[1]},{"index":0,"embedding":[1]}],"usage":{"prompt_tokens":2}}`,
	} {
		srv := serveOpenAIEmbed(t, body, 200)
		c, err := newTestOpenAIEmbed(srv.URL).embed(context.Background(), "m", []string{"a", "b"}, "", 0)
		if err == nil || !strings.Contains(err.Error(), "out of range or repeated") {
			t.Fatalf("err %v", err)
		}
		if c.tokens != 2 || c.vectors != nil {
			t.Fatalf("%+v", c)
		}
	}
}

func TestOpenAIEmbedMissingItem(t *testing.T) {
	srv := serveOpenAIEmbed(t, `{"data":[{"index":0,"embedding":[1]}],"usage":{"prompt_tokens":7}}`, 200)
	c, err := newTestOpenAIEmbed(srv.URL).embed(context.Background(), "m", []string{"a", "b"}, "", 0)
	if err == nil || !strings.Contains(err.Error(), "got 1 embeddings for 2 texts") {
		t.Fatalf("err %v", err)
	}
	if c.tokens != 7 || c.estimated {
		t.Fatalf("%+v", c)
	}
}

func TestOpenAIEmbedNoUsage(t *testing.T) {
	srv := serveOpenAIEmbed(t, `{"data":[{"index":0,"embedding":[1]},{"index":1,"embedding":[1]}]}`, 200)
	c, err := newTestOpenAIEmbed(srv.URL).embed(context.Background(), "m", []string{"abcdefgh", "ij"}, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if want := estimateTokens("abcdefghij"); c.tokens != want || !c.estimated {
		t.Fatalf("%+v want %d", c, want)
	}
}

func TestOpenAIEmbedHTTPError(t *testing.T) {
	srv := serveOpenAIEmbed(t, `{"error":{"message":"bad dims"}}`, 400)
	_, err := newTestOpenAIEmbed(srv.URL).embed(context.Background(), "m", []string{"a"}, "", 0)
	if err == nil || !strings.Contains(err.Error(), "bad dims") || !strings.Contains(err.Error(), "status 400") {
		t.Fatalf("err %v", err)
	}
}

func TestOpenAIEmbedBaseURL(t *testing.T) {
	if b := newOpenAIEmbed("k", "http://x/v1/", nil).(*openaiEmbed); b.baseURL != "http://x/v1" {
		t.Fatalf("baseURL %q", b.baseURL)
	}
	if b := newOpenAIEmbed("k", "", nil).(*openaiEmbed); b.baseURL != openaiBaseURL {
		t.Fatalf("baseURL %q", b.baseURL)
	}
}

func TestOpenAIEmbedRetry(t *testing.T) {
	var mu sync.Mutex
	n := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		n++
		cur := n
		mu.Unlock()
		if cur == 1 {
			w.Header().Set("retry-after-ms", "0")
			w.WriteHeader(503)
			return
		}
		fmt.Fprint(w, `{"data":[{"index":0,"embedding":[1]}],"usage":{"prompt_tokens":1}}`)
	}))
	defer srv.Close()
	if _, err := newTestOpenAIEmbed(srv.URL).embed(context.Background(), "m", []string{"a"}, "", 0); err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("requests %d", n)
	}
}

func TestOpenAIEmbedThroughCore(t *testing.T) {
	var mu sync.Mutex
	reqs := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var in struct {
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&in)
		mu.Lock()
		reqs++
		mu.Unlock()
		items := make([]map[string]any, len(in.Input))
		for i := range in.Input {
			v := []float32{1, 0}
			if i%2 == 1 {
				v = []float32{0, 2}
			}
			items[i] = map[string]any{"index": i, "embedding": v}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": items, "usage": map[string]any{"prompt_tokens": len(in.Input)}})
	}))
	defer srv.Close()
	e := NewEmbedder(Config{OpenAIAPIKey: "k", OpenAIBaseURL: srv.URL,
		EmbedModel: ModelRef{Provider: ProviderOpenAI, Model: "text-embedding-3-small"}})
	texts := make([]string, 2049)
	for i := range texts {
		texts[i] = "a"
	}
	resp, err := e.Embed(context.Background(), EmbedRequest{Texts: texts})
	if err != nil {
		t.Fatal(err)
	}
	if reqs != 2 || len(resp.Vectors) != 2049 || resp.InputTokens != 2049 {
		t.Fatalf("reqs %d vectors %d tokens %d", reqs, len(resp.Vectors), resp.InputTokens)
	}
	for i, v := range resp.Vectors {
		var want []float32 = []float32{1, 0}
		if i%2 == 1 {
			want = []float32{0, 1}
		}
		if len(v) != 2 || math.Abs(float64(v[0]-want[0])) > 1e-6 || math.Abs(float64(v[1]-want[1])) > 1e-6 {
			t.Fatalf("vector %d = %v", i, v)
		}
	}
}

func TestOpenAIEmbedBadJSONIsBilled(t *testing.T) {
	srv := serveOpenAIEmbed(t, `{not json`, 200)
	c, err := newTestOpenAIEmbed(srv.URL).embed(context.Background(), "m", []string{"abcdefgh", "ij"}, "", 0)
	if err == nil || !strings.Contains(err.Error(), "decode embeddings") {
		t.Fatalf("err %v", err)
	}
	if want := estimateTokens("abcdefghij"); c.tokens != want || !c.estimated {
		t.Fatalf("%+v want %d", c, want)
	}
	// A non-2xx reply stays at 0 tokens.
	srv = serveOpenAIEmbed(t, `{not json`, 500)
	c, _ = newTestOpenAIEmbed(srv.URL).embed(context.Background(), "m", []string{"abcdefgh"}, "", 0)
	if c.tokens != 0 {
		t.Fatalf("500 tokens %d", c.tokens)
	}
}
