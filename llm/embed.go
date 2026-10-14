package llm

import (
	"context"
	"fmt"
	"math"
	"time"
)

// EmbedKind is a hint about how the vectors will be used. It is optional.
// OpenAI ignores it. Gemini maps it to a task type (query or document).
type EmbedKind string

const (
	// EmbedQuery marks a search query.
	EmbedQuery EmbedKind = "query"
	// EmbedDocument marks a text that will be searched.
	EmbedDocument EmbedKind = "document"
)

// EmbedRequest asks for one vector per text.
type EmbedRequest struct {
	// Texts to embed. Each must be non-empty. Any number is fine: Embed splits
	// them into chunks the provider accepts.
	Texts []string
	// Kind is an optional hint ("" = none). See EmbedKind.
	Kind EmbedKind
	// Dimensions asks for shorter vectors. 0 = the model's default size.
	Dimensions int
	// Model overrides the configured embed model for this call. Nil or empty = no override.
	Model *ModelRef
}

// EmbedResponse is the result of an Embed call.
type EmbedResponse struct {
	// Vectors has one vector per text, in the same order as Texts. Each has
	// unit length (a zero vector stays zero).
	Vectors [][]float32
	// Provider and Model say which model made the vectors.
	Provider Provider
	Model    string
	// InputTokens is what the provider billed, or a guess when UsageEstimated is true.
	// On an error it still counts the chunks that were billed.
	InputTokens    int
	UsageEstimated bool
}

// Embedder turns text into vectors.
//
// There is no failover: if the provider fails, Embed returns the error. Vectors
// from different models cannot be compared, so store Model with each vector.
type Embedder interface {
	Embed(ctx context.Context, req EmbedRequest) (EmbedResponse, error)
	// Enabled reports whether the configured embed model has a backend with a key.
	Enabled() bool
}

// embedBackend is one provider's embedding endpoint.
type embedBackend interface {
	// embed sends ONE chunk. texts is never empty and never longer than maxTexts().
	embed(ctx context.Context, model string, texts []string, kind EmbedKind, dims int) (embedChunk, error)
	maxTexts() int  // texts per chunk, > 0
	maxTokens() int // guessed tokens per chunk; 0 = no token cap
}

// embedChunk is the result of one backend call.
type embedChunk struct {
	vectors   [][]float32 // one per text, in the order of texts
	tokens    int
	estimated bool
}

// defaultEmbedTimeout bounds one chunk when the context has no deadline and
// Config.RequestTimeout is 0.
const defaultEmbedTimeout = 60 * time.Second

// NewEmbedder builds an Embedder from cfg. It reads the key fields, OpenAIBaseURL,
// RequestTimeout, Logger and EmbedModel. If no OpenAI or Google key is set, or
// EmbedModel is empty, it returns an Embedder that is not enabled and whose
// Embed returns ErrNotConfigured.
func NewEmbedder(cfg Config) Embedder {
	lg := cfg.Logger
	backends := map[Provider]embedBackend{}
	if cfg.OpenAIAPIKey != "" {
		backends[ProviderOpenAI] = newOpenAIEmbed(cfg.OpenAIAPIKey, cfg.OpenAIBaseURL, lg)
	}
	if cfg.GoogleAPIKey != "" {
		backends[ProviderGoogle] = newGoogleEmbed(cfg.GoogleAPIKey, lg)
	}
	if len(backends) == 0 || cfg.EmbedModel.empty() {
		return noopEmbedder{}
	}
	return &embedder{backends: backends, model: cfg.EmbedModel, timeout: cfg.RequestTimeout}
}

type noopEmbedder struct{}

func (noopEmbedder) Embed(context.Context, EmbedRequest) (EmbedResponse, error) {
	return EmbedResponse{}, ErrNotConfigured
}
func (noopEmbedder) Enabled() bool { return false }

type embedder struct {
	backends map[Provider]embedBackend // only OpenAI and Google, only with a key
	model    ModelRef                  // Config.EmbedModel
	timeout  time.Duration             // Config.RequestTimeout; 0 = defaultEmbedTimeout
}

func (e *embedder) Enabled() bool {
	_, ok := e.backends[e.model.Provider]
	return ok
}

func (e *embedder) Embed(ctx context.Context, req EmbedRequest) (EmbedResponse, error) {
	ref := e.model
	if req.Model != nil && !req.Model.empty() {
		ref = *req.Model
	}
	switch ref.Provider {
	case ProviderAnthropic, ProviderDeepSeek, ProviderMoonshot:
		return EmbedResponse{}, fmt.Errorf("%w: %s", ErrEmbedUnsupported, ref.Provider)
	}
	b, ok := e.backends[ref.Provider]
	if !ok {
		return EmbedResponse{}, ErrNotConfigured
	}
	base := EmbedResponse{Provider: ref.Provider, Model: ref.Model}
	if len(req.Texts) == 0 {
		base.Vectors = [][]float32{}
		return base, nil
	}
	for i, t := range req.Texts {
		if t == "" {
			return EmbedResponse{}, fmt.Errorf("llm: embed text %d is empty", i)
		}
	}
	if req.Dimensions < 0 {
		return EmbedResponse{}, fmt.Errorf("llm: embed Dimensions is negative")
	}
	switch req.Kind {
	case "", EmbedQuery, EmbedDocument:
	default:
		return EmbedResponse{}, fmt.Errorf("llm: unknown EmbedKind %q", req.Kind)
	}

	vectors := make([][]float32, len(req.Texts))
	total := 0
	estimated := false
	for _, r := range chunkRanges(req.Texts, b.maxTexts(), b.maxTokens()) {
		s, t := r[0], r[1]
		cctx, cancel := e.chunkContext(ctx)
		chunk, err := b.embed(cctx, ref.Model, req.Texts[s:t], req.Kind, req.Dimensions)
		cancel()
		// Count billed tokens first, even on error.
		total += chunk.tokens
		estimated = estimated || chunk.estimated
		partial := EmbedResponse{Provider: ref.Provider, Model: ref.Model, InputTokens: total, UsageEstimated: estimated}
		if err != nil {
			return partial, err
		}
		if len(chunk.vectors) != t-s {
			return partial, fmt.Errorf("llm: %s/%s returned %d vectors for %d texts",
				ref.Provider, ref.Model, len(chunk.vectors), t-s)
		}
		copy(vectors[s:t], chunk.vectors)
	}
	for _, v := range vectors {
		normalizeVec(v)
	}
	base.Vectors = vectors
	base.InputTokens = total
	base.UsageEstimated = estimated
	return base, nil
}

// chunkContext gives one chunk a deadline. A caller deadline always wins.
func (e *embedder) chunkContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, func() {}
	}
	d := e.timeout
	if d == 0 {
		d = defaultEmbedTimeout
	}
	return context.WithTimeout(ctx, d)
}

// chunkRanges splits texts into [start, end) ranges, in order. A range holds at
// most maxTexts texts and, when maxTokens > 0, at most maxTokens guessed tokens.
// One text over maxTokens is a range on its own.
func chunkRanges(texts []string, maxTexts, maxTokens int) [][2]int {
	var out [][2]int
	start, tokens := 0, 0
	for i, t := range texts {
		n := estimateTokens(t)
		if i > start && (i-start >= maxTexts || (maxTokens > 0 && tokens+n > maxTokens)) {
			out = append(out, [2]int{start, i})
			start, tokens = i, 0
		}
		tokens += n
	}
	if len(texts) > 0 {
		out = append(out, [2]int{start, len(texts)})
	}
	return out
}

// normalizeVec scales v to unit length in place. A zero vector is left as is.
func normalizeVec(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	norm := math.Sqrt(sum)
	for i, x := range v {
		v[i] = float32(float64(x) / norm)
	}
}
