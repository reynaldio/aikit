package llm

import (
	"context"
	"time"
)

// Usage is the tokens and cost of one Complete call.
type Usage struct {
	Provider         Provider
	Model            string
	Task             Task // from the request
	InputTokens      int
	OutputTokens     int
	CachedTokens     int
	CacheWriteTokens int
	CostUSD          float64       // book.Cost(Model, the four counts)
	Estimated        bool          // the counts are a guess (Response.UsageEstimated)
	Duration         time.Duration // time spent in Complete
	Err              error         // nil on success
}

// WithUsage wraps inner and calls record once for each Complete call that used
// tokens. That is a success, ErrRefused, ErrSchemaViolation, or a stream that
// stopped early. A call with no tokens (a transport error, ErrNotConfigured, a
// guard refusal) is not reported.
//
// book prices the call. A nil book uses DefaultPrices.
//
// record runs on the calling goroutine, before Complete returns. The ctx it
// gets may already be cancelled (the user pressed Stop). To save the record,
// use context.WithoutCancel(ctx).
//
// Known gap: if the primary model fails and the fallback answers, only the
// fallback's tokens are reported. Tokens the primary already charged (a
// refusal, or a stream that broke before any text) are not reported. This
// reports too little, but never charges twice. It also happens when the
// primary stops with tokens and the fallback then fails with tokens: the
// primary's reply is returned and recorded, and the fallback's tokens are
// dropped.
//
// It works in either order with guard.Wrap.
//
// WithUsage panics if inner or record is nil.
func WithUsage(inner Client, book *PriceBook, record func(ctx context.Context, u Usage)) Client {
	if inner == nil {
		panic("llm: WithUsage needs an inner Client")
	}
	if record == nil {
		panic("llm: WithUsage needs a record func")
	}
	return &usageClient{inner: inner, book: book, record: record}
}

type usageClient struct {
	inner  Client
	book   *PriceBook
	record func(ctx context.Context, u Usage)
}

func (c *usageClient) Enabled() bool { return c.inner.Enabled() }

func (c *usageClient) CompareTargets() []ModelRef { return c.inner.CompareTargets() }

func (c *usageClient) Complete(ctx context.Context, req Request) (Response, error) {
	start := time.Now()
	resp, err := c.inner.Complete(ctx, req)
	if hasUsage(resp) {
		c.record(ctx, Usage{
			Provider:         resp.Provider,
			Model:            resp.Model,
			Task:             req.Task,
			InputTokens:      resp.InputTokens,
			OutputTokens:     resp.OutputTokens,
			CachedTokens:     resp.CachedTokens,
			CacheWriteTokens: resp.CacheWriteTokens,
			CostUSD:          c.book.Cost(resp.Model, resp.InputTokens, resp.OutputTokens, resp.CachedTokens, resp.CacheWriteTokens),
			Estimated:        resp.UsageEstimated,
			Duration:         time.Since(start),
			Err:              err,
		})
	}
	return resp, err
}

// WithEmbedUsage wraps inner and calls record once for each Embed call that used
// tokens. That includes a call that failed after some chunks were already billed:
// the record carries those tokens and the error. A call with no tokens is not
// reported.
//
// book prices the call (input tokens only). A nil book uses DefaultPrices.
//
// record runs on the calling goroutine, before Embed returns. The ctx it gets may
// already be cancelled. To save the record, use context.WithoutCancel(ctx).
//
// WithEmbedUsage panics if inner or record is nil.
func WithEmbedUsage(inner Embedder, book *PriceBook, record func(ctx context.Context, u Usage)) Embedder {
	if inner == nil {
		panic("llm: WithEmbedUsage needs an inner Embedder")
	}
	if record == nil {
		panic("llm: WithEmbedUsage needs a record func")
	}
	return &embedUsage{inner: inner, book: book, record: record}
}

type embedUsage struct {
	inner  Embedder
	book   *PriceBook
	record func(ctx context.Context, u Usage)
}

func (c *embedUsage) Enabled() bool { return c.inner.Enabled() }

func (c *embedUsage) Embed(ctx context.Context, req EmbedRequest) (EmbedResponse, error) {
	start := time.Now()
	resp, err := c.inner.Embed(ctx, req)
	if resp.InputTokens > 0 {
		c.record(ctx, Usage{
			Provider:    resp.Provider,
			Model:       resp.Model,
			Task:        TaskEmbed,
			InputTokens: resp.InputTokens,
			CostUSD:     c.book.Cost(resp.Model, resp.InputTokens, 0, 0, 0),
			Estimated:   resp.UsageEstimated,
			Duration:    time.Since(start),
			Err:         err,
		})
	}
	return resp, err
}
