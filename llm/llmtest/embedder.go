package llmtest

import (
	"context"
	"hash/fnv"
	"math"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/reynaldio/aikit/llm"
)

// Embedder is a fake llm.Embedder. It makes the same unit-length vector for the
// same text and kind, and records every request. It is safe for concurrent use.
//
//	e := llmtest.NewEmbedder(8)
//	doc, _ := e.Embed(ctx, llm.EmbedRequest{Texts: []string{"cats"}, Kind: llm.EmbedDocument})
//	q, _ := e.Embed(ctx, llm.EmbedRequest{Texts: []string{"cats"}, Kind: llm.EmbedQuery})
//	// compare with a dot product of doc.Vectors[0] and q.Vectors[0]
//	reqs := e.Requests() // one entry per Embed call, in order
type Embedder struct {
	mu       sync.Mutex
	dims     int
	requests []llm.EmbedRequest
	failNext error
}

var _ llm.Embedder = (*Embedder)(nil)

// NewEmbedder returns an Embedder with vectors of size dims. dims <= 0 means 8.
func NewEmbedder(dims int) *Embedder {
	if dims <= 0 {
		dims = 8
	}
	return &Embedder{dims: dims}
}

// Enabled always reports true.
func (e *Embedder) Enabled() bool { return true }

// Requests returns a copy of the requests received so far, in order.
func (e *Embedder) Requests() []llm.EmbedRequest {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]llm.EmbedRequest(nil), e.requests...)
}

// Fail makes the next Embed call return err. Later calls work again.
func (e *Embedder) Fail(err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.failNext = err
}

// Embed records req, then returns one vector per text. A cancelled ctx returns
// ctx.Err(). A pending Fail error is returned once.
func (e *Embedder) Embed(ctx context.Context, req llm.EmbedRequest) (llm.EmbedResponse, error) {
	e.mu.Lock()
	rec := req
	rec.Texts = append([]string(nil), req.Texts...)
	e.requests = append(e.requests, rec)
	if err := ctx.Err(); err != nil {
		e.mu.Unlock()
		return llm.EmbedResponse{}, err
	}
	if err := e.failNext; err != nil {
		e.failNext = nil
		e.mu.Unlock()
		return llm.EmbedResponse{}, err
	}
	d := e.dims
	e.mu.Unlock()
	if req.Dimensions > 0 {
		d = req.Dimensions
	}

	vecs := make([][]float32, len(req.Texts))
	for i, t := range req.Texts {
		vecs[i] = fakeVector(req.Kind, t, d)
	}
	runes := utf8.RuneCountInString(strings.Join(req.Texts, ""))
	return llm.EmbedResponse{
		Vectors:     vecs,
		Provider:    "llmtest",
		Model:       "fake",
		InputTokens: (runes + 3) / 4,
	}, nil
}

// fakeVector makes a unit-length vector of size d from the kind and text.
func fakeVector(kind llm.EmbedKind, text string, d int) []float32 {
	v := make([]float32, d)
	var sum float64
	for i := range v {
		h := fnv.New64a()
		_, _ = h.Write([]byte(string(kind) + "\x00" + text + "\x00" + strconv.Itoa(i)))
		x := float64(h.Sum64()>>11)/float64(1<<53)*2 - 1
		v[i] = float32(x)
		sum += float64(v[i]) * float64(v[i])
	}
	if sum == 0 {
		return v
	}
	norm := math.Sqrt(sum)
	for i, x := range v {
		v[i] = float32(float64(x) / norm)
	}
	return v
}
