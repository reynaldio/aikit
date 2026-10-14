package llm

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

type fakeEmbedBackend struct {
	maxN, maxTok int
	calls        [][]string
	kinds        []EmbedKind
	dims         []int
	models       []string
	deadlines    []time.Duration
	failOn       int
	err          error
	tokensPer    int
	estimated    bool
	short        bool
}

func (f *fakeEmbedBackend) maxTexts() int  { return f.maxN }
func (f *fakeEmbedBackend) maxTokens() int { return f.maxTok }

func (f *fakeEmbedBackend) embed(ctx context.Context, model string, texts []string, kind EmbedKind, dims int) (embedChunk, error) {
	f.calls = append(f.calls, texts)
	f.kinds = append(f.kinds, kind)
	f.dims = append(f.dims, dims)
	f.models = append(f.models, model)
	if dl, ok := ctx.Deadline(); ok {
		f.deadlines = append(f.deadlines, time.Until(dl))
	} else {
		f.deadlines = append(f.deadlines, -1)
	}
	tokens := f.tokensPer * len(texts)
	if f.failOn != 0 && len(f.calls) == f.failOn {
		return embedChunk{tokens: tokens, estimated: f.estimated}, f.err
	}
	n := len(texts)
	if f.short {
		n--
	}
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		vecs[i] = []float32{float32(len(texts[i])), 1}
	}
	return embedChunk{vectors: vecs, tokens: tokens, estimated: f.estimated}, nil
}

func newTestEmbedder(b embedBackend) *embedder {
	return &embedder{
		backends: map[Provider]embedBackend{ProviderOpenAI: b},
		model:    ModelRef{ProviderOpenAI, "m"},
	}
}

func TestEmbed_NewEmbedder(t *testing.T) {
	ctx := context.Background()
	m := ModelRef{ProviderOpenAI, "m"}
	for name, cfg := range map[string]Config{
		"no keys":         {EmbedModel: m},
		"empty model":     {OpenAIAPIKey: "k", GoogleAPIKey: "k"},
		"anthropic only":  {AnthropicAPIKey: "k", EmbedModel: m},
		"google, openai?": {GoogleAPIKey: "k", EmbedModel: m},
	} {
		e := NewEmbedder(cfg)
		if e.Enabled() {
			t.Errorf("%s: Enabled true", name)
		}
		if _, err := e.Embed(ctx, EmbedRequest{Texts: []string{"a"}}); !errors.Is(err, ErrNotConfigured) {
			t.Errorf("%s: err = %v", name, err)
		}
	}
	if !NewEmbedder(Config{OpenAIAPIKey: "k", EmbedModel: m}).Enabled() {
		t.Error("openai: Enabled false")
	}
	if !NewEmbedder(Config{GoogleAPIKey: "k", EmbedModel: ModelRef{ProviderGoogle, "g"}}).Enabled() {
		t.Error("google: Enabled false")
	}
}

func TestEmbed_Unsupported(t *testing.T) {
	for _, p := range []Provider{ProviderAnthropic, ProviderDeepSeek, ProviderMoonshot} {
		fb := &fakeEmbedBackend{maxN: 10}
		e := newTestEmbedder(fb)
		e.model = ModelRef{p, "m"}
		_, err := e.Embed(context.Background(), EmbedRequest{Texts: []string{"a"}})
		if !errors.Is(err, ErrEmbedUnsupported) {
			t.Errorf("%s: err = %v", p, err)
		}
		e.model = ModelRef{ProviderOpenAI, "m"}
		_, err = e.Embed(context.Background(), EmbedRequest{Texts: []string{"a"}, Model: &ModelRef{p, "m"}})
		if !errors.Is(err, ErrEmbedUnsupported) {
			t.Errorf("%s override: err = %v", p, err)
		}
		if len(fb.calls) != 0 {
			t.Errorf("%s: backend called", p)
		}
	}
}

func TestEmbed_ModelOverride(t *testing.T) {
	fb := &fakeEmbedBackend{maxN: 10}
	gb := &fakeEmbedBackend{maxN: 10}
	e := newTestEmbedder(fb)
	e.backends[ProviderGoogle] = gb
	ctx := context.Background()
	res, err := e.Embed(ctx, EmbedRequest{Texts: []string{"a"}, Model: &ModelRef{ProviderGoogle, "g1"}})
	if err != nil || res.Provider != ProviderGoogle || res.Model != "g1" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
	if len(gb.calls) != 1 || gb.models[0] != "g1" || len(fb.calls) != 0 {
		t.Errorf("routing wrong: g=%v f=%v", gb.models, fb.calls)
	}
	res, err = e.Embed(ctx, EmbedRequest{Texts: []string{"a"}, Model: &ModelRef{}})
	if err != nil || res.Provider != ProviderOpenAI || res.Model != "m" || len(fb.calls) != 1 {
		t.Errorf("empty override not ignored: %+v %v", res, err)
	}
	// Unknown provider name.
	if _, err := e.Embed(ctx, EmbedRequest{Texts: []string{"a"}, Model: &ModelRef{"nope", "x"}}); !errors.Is(err, ErrNotConfigured) {
		t.Errorf("unknown provider: %v", err)
	}
}

func TestEmbed_EmptyTexts(t *testing.T) {
	fb := &fakeEmbedBackend{maxN: 10}
	e := newTestEmbedder(fb)
	for _, texts := range [][]string{nil, {}} {
		res, err := e.Embed(context.Background(), EmbedRequest{Texts: texts})
		if err != nil {
			t.Fatal(err)
		}
		if res.Vectors == nil || len(res.Vectors) != 0 {
			t.Errorf("Vectors = %#v", res.Vectors)
		}
		if res.Provider != ProviderOpenAI || res.Model != "m" {
			t.Errorf("res = %+v", res)
		}
	}
	if len(fb.calls) != 0 {
		t.Error("backend called")
	}
}

func TestEmbed_Validation(t *testing.T) {
	fb := &fakeEmbedBackend{maxN: 10}
	e := newTestEmbedder(fb)
	ctx := context.Background()
	_, err := e.Embed(ctx, EmbedRequest{Texts: []string{"a", "b", "", "d"}})
	if err == nil || !strings.Contains(err.Error(), "2") {
		t.Errorf("empty text: %v", err)
	}
	if _, err = e.Embed(ctx, EmbedRequest{Texts: []string{"a"}, Dimensions: -1}); err == nil {
		t.Error("negative dims: no error")
	}
	if _, err = e.Embed(ctx, EmbedRequest{Texts: []string{"a"}, Kind: "x"}); err == nil {
		t.Error("bad kind: no error")
	}
	if len(fb.calls) != 0 {
		t.Error("backend called")
	}
}

func TestEmbed_KindAndDimsReachBackend(t *testing.T) {
	fb := &fakeEmbedBackend{maxN: 2}
	e := newTestEmbedder(fb)
	_, err := e.Embed(context.Background(), EmbedRequest{Texts: []string{"a", "b", "c"}, Kind: EmbedQuery, Dimensions: 256})
	if err != nil {
		t.Fatal(err)
	}
	if len(fb.calls) != 2 {
		t.Fatalf("calls = %d", len(fb.calls))
	}
	for i := range fb.calls {
		if fb.kinds[i] != EmbedQuery || fb.dims[i] != 256 {
			t.Errorf("chunk %d: kind=%q dims=%d", i, fb.kinds[i], fb.dims[i])
		}
	}
}

func TestEmbed_Chunking(t *testing.T) {
	fb := &fakeEmbedBackend{maxN: 2, tokensPer: 3}
	e := newTestEmbedder(fb)
	texts := []string{"a", "bb", "ccc", "dddd", "eeeee"}
	res, err := e.Embed(context.Background(), EmbedRequest{Texts: texts})
	if err != nil {
		t.Fatal(err)
	}
	var sizes []int
	for _, c := range fb.calls {
		sizes = append(sizes, len(c))
	}
	if !reflect.DeepEqual(sizes, []int{2, 2, 1}) {
		t.Errorf("sizes = %v", sizes)
	}
	if len(res.Vectors) != 5 {
		t.Fatalf("vectors = %d", len(res.Vectors))
	}
	// The marker survives normalizing: x/y recovers len(text).
	for i, v := range res.Vectors {
		if got := float64(v[0]) / float64(v[1]); math.Abs(got-float64(len(texts[i]))) > 1e-4 {
			t.Errorf("vec %d: x/y = %v, want %d", i, got, len(texts[i]))
		}
	}
	if res.InputTokens != 15 {
		t.Errorf("tokens = %d", res.InputTokens)
	}
}

func TestEmbed_ChunkingOrder(t *testing.T) {
	// Use 2-d vectors whose direction encodes the input: backend returns [len, 0],
	// so order cannot be seen after normalizing. Check order through the call log.
	fb := &fakeEmbedBackend{maxN: 2}
	e := newTestEmbedder(fb)
	texts := []string{"a", "bb", "ccc", "dddd", "eeeee"}
	if _, err := e.Embed(context.Background(), EmbedRequest{Texts: texts}); err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, c := range fb.calls {
		got = append(got, c...)
	}
	if !reflect.DeepEqual(got, texts) {
		t.Errorf("order = %v", got)
	}
}

func TestEmbed_FailureInChunk(t *testing.T) {
	boom := errors.New("boom")
	fb := &fakeEmbedBackend{maxN: 1, tokensPer: 5, failOn: 2, err: boom}
	e := newTestEmbedder(fb)
	res, err := e.Embed(context.Background(), EmbedRequest{Texts: []string{"a", "b", "c"}})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
	// Chunk 2 failed and the fake still reports its tokens (billed), so 5 + 5.
	if res.Vectors != nil || res.Provider != ProviderOpenAI || res.Model != "m" || res.InputTokens != 10 {
		t.Errorf("res = %+v", res)
	}
	if len(fb.calls) != 2 {
		t.Errorf("calls = %d", len(fb.calls))
	}

	// Failure with no tokens billed (transport error).
	fb2 := &failNoTokens{err: boom, failOn: 2}
	e = newTestEmbedder(fb2)
	res, err = e.Embed(context.Background(), EmbedRequest{Texts: []string{"a", "b", "c"}})
	if !errors.Is(err, boom) || res.InputTokens != 5 || res.Vectors != nil {
		t.Errorf("res=%+v err=%v", res, err)
	}
	if fb2.n != 2 {
		t.Errorf("calls = %d", fb2.n)
	}
	fb2 = &failNoTokens{err: boom, failOn: 1}
	e = newTestEmbedder(fb2)
	res, err = e.Embed(context.Background(), EmbedRequest{Texts: []string{"a", "b"}})
	if !errors.Is(err, boom) || res.InputTokens != 0 {
		t.Errorf("chunk 1 fail: res=%+v err=%v", res, err)
	}
}

// failNoTokens succeeds with 5 tokens per chunk, and fails the failOn-th call with 0 tokens.
type failNoTokens struct {
	n, failOn int
	err       error
}

func (f *failNoTokens) maxTexts() int  { return 1 }
func (f *failNoTokens) maxTokens() int { return 0 }
func (f *failNoTokens) embed(ctx context.Context, model string, texts []string, kind EmbedKind, dims int) (embedChunk, error) {
	f.n++
	if f.n == f.failOn {
		return embedChunk{}, f.err
	}
	return embedChunk{vectors: [][]float32{{1, 0}}, tokens: 5}, nil
}

func TestEmbed_WrongVectorCount(t *testing.T) {
	fb := &fakeEmbedBackend{maxN: 2, tokensPer: 4, short: true}
	e := newTestEmbedder(fb)
	res, err := e.Embed(context.Background(), EmbedRequest{Texts: []string{"a", "b", "c"}})
	if err == nil {
		t.Fatal("no error")
	}
	if res.InputTokens != 8 || res.Vectors != nil {
		t.Errorf("res = %+v", res)
	}
	if len(fb.calls) != 1 {
		t.Errorf("calls = %d", len(fb.calls))
	}
}

func TestEmbed_ErrorWithTokens(t *testing.T) {
	boom := errors.New("boom")
	e := newTestEmbedder(&errTokens{boom: boom})
	res, err := e.Embed(context.Background(), EmbedRequest{Texts: []string{"a", "b"}})
	if !errors.Is(err, boom) || res.InputTokens != 4+7 {
		t.Errorf("res=%+v err=%v", res, err)
	}
}

type errTokens struct {
	n    int
	boom error
}

func (f *errTokens) maxTexts() int  { return 1 }
func (f *errTokens) maxTokens() int { return 0 }
func (f *errTokens) embed(ctx context.Context, model string, texts []string, kind EmbedKind, dims int) (embedChunk, error) {
	f.n++
	if f.n == 2 {
		return embedChunk{tokens: 7, estimated: true}, f.boom
	}
	return embedChunk{vectors: [][]float32{{1, 0}}, tokens: 4}, nil
}

func TestEmbed_UsageEstimated(t *testing.T) {
	for _, est := range []bool{true, false} {
		fb := &fakeEmbedBackend{maxN: 1, estimated: est}
		res, err := newTestEmbedder(fb).Embed(context.Background(), EmbedRequest{Texts: []string{"a", "b"}})
		if err != nil || res.UsageEstimated != est {
			t.Errorf("est=%v: res=%+v err=%v", est, res, err)
		}
		if len(fb.calls) != 2 {
			t.Errorf("est=%v: calls = %d, want 2", est, len(fb.calls))
		}
	}
	// Only one chunk estimated.
	res, _ := newTestEmbedder(&errTokens{boom: errors.New("x")}).Embed(context.Background(), EmbedRequest{Texts: []string{"a", "b"}})
	if !res.UsageEstimated {
		t.Error("estimated flag lost on error")
	}
}

func TestNormalizeVec(t *testing.T) {
	v := []float32{3, 4}
	normalizeVec(v)
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Errorf("v = %v", v)
	}
	z := []float32{0, 0}
	normalizeVec(z)
	if z[0] != 0 || z[1] != 0 {
		t.Errorf("z = %v", z)
	}
}

func TestEmbed_Deadline(t *testing.T) {
	check := func(name string, e *embedder, ctx context.Context, lo, hi time.Duration) {
		t.Helper()
		fb := e.backends[ProviderOpenAI].(*fakeEmbedBackend)
		fb.deadlines = nil
		if _, err := e.Embed(ctx, EmbedRequest{Texts: []string{"a"}}); err != nil {
			t.Fatal(err)
		}
		d := fb.deadlines[0]
		if d < lo || d > hi {
			t.Errorf("%s: deadline %v not in [%v, %v]", name, d, lo, hi)
		}
	}
	e := newTestEmbedder(&fakeEmbedBackend{maxN: 1})
	check("default", e, context.Background(), 59*time.Second, 60*time.Second)
	e.timeout = 5 * time.Second
	check("timeout", e, context.Background(), 4*time.Second, 5*time.Second)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	check("caller", e, ctx, 9*time.Second, 10*time.Second)
}

func TestChunkRanges(t *testing.T) {
	r8 := strings.Repeat("x", 8)
	cases := []struct {
		name  string
		texts []string
		n, tk int
		want  [][2]int
	}{
		{"nil", nil, 2, 0, nil},
		{"max texts", []string{"a", "b", "c"}, 2, 0, [][2]int{{0, 2}, {2, 3}}},
		{"token cap", []string{r8, r8, r8}, 10, 4, [][2]int{{0, 2}, {2, 3}}},
		{"oversize alone", []string{"a", strings.Repeat("x", 40), "b"}, 10, 4, [][2]int{{0, 1}, {1, 2}, {2, 3}}},
		{"oversize only", []string{strings.Repeat("x", 40)}, 10, 4, [][2]int{{0, 1}}},
		{"tokens ignored", []string{r8, r8, r8}, 10, 0, [][2]int{{0, 3}}},
	}
	for _, c := range cases {
		got := chunkRanges(c.texts, c.n, c.tk)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: got %v want %v", c.name, got, c.want)
		}
	}
}

func TestEmbed_GoogleModelPrefixStripped(t *testing.T) {
	gb := &fakeEmbedBackend{maxN: 10}
	e := newTestEmbedder(&fakeEmbedBackend{maxN: 10})
	e.backends[ProviderGoogle] = gb
	res, err := e.Embed(context.Background(), EmbedRequest{Texts: []string{"a"}, Model: &ModelRef{ProviderGoogle, "models/gemini-embedding-001"}})
	if err != nil || res.Model != "gemini-embedding-001" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
	if len(gb.models) != 1 || gb.models[0] != "gemini-embedding-001" {
		t.Errorf("backend model = %v", gb.models)
	}
	// Other providers keep their model name as given.
	res, _ = e.Embed(context.Background(), EmbedRequest{Texts: []string{"a"}, Model: &ModelRef{ProviderOpenAI, "models/x"}})
	if res.Model != "models/x" {
		t.Errorf("openai model = %q", res.Model)
	}
}

// shapeBackend returns the given vectors per call (one chunk per text).
type shapeBackend struct {
	n    int
	vecs [][]float32
}

func (f *shapeBackend) maxTexts() int  { return 1 }
func (f *shapeBackend) maxTokens() int { return 0 }
func (f *shapeBackend) embed(ctx context.Context, model string, texts []string, kind EmbedKind, dims int) (embedChunk, error) {
	v := f.vecs[f.n]
	f.n++
	return embedChunk{vectors: [][]float32{v}, tokens: 3}, nil
}

func TestEmbed_BadVectorShape(t *testing.T) {
	// Empty vector in the first chunk.
	e := newTestEmbedder(&shapeBackend{vecs: [][]float32{{}}})
	res, err := e.Embed(context.Background(), EmbedRequest{Texts: []string{"a"}})
	if err == nil || !strings.Contains(err.Error(), "empty or uneven vector") || res.InputTokens != 3 || res.Vectors != nil {
		t.Errorf("empty: res=%+v err=%v", res, err)
	}
	// Chunk 2 has another length than chunk 1.
	e = newTestEmbedder(&shapeBackend{vecs: [][]float32{{1, 0}, {1, 0, 0}}})
	res, err = e.Embed(context.Background(), EmbedRequest{Texts: []string{"a", "b"}})
	if err == nil || !strings.Contains(err.Error(), "empty or uneven vector") || res.InputTokens != 6 || res.Vectors != nil {
		t.Errorf("uneven: res=%+v err=%v", res, err)
	}
}
