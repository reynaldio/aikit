package llm

import (
	"context"
	"iter"
	"time"
)

// Usage is the tokens and cost of one Complete or Embed call, or one batch item.
type Usage struct {
	Provider         Provider
	Model            string
	Task             Task // from the request (TaskEmbed for Embed)
	InputTokens      int
	OutputTokens     int
	CachedTokens     int
	CacheWriteTokens int
	CostUSD          float64       // book.Cost(Model, the four counts) × the batch factor for a batch item
	Estimated        bool          // the counts are a guess (Response.UsageEstimated)
	Duration         time.Duration // time spent in the call
	Err              error         // nil on success
	Batch            bool          // true for a batch item
	BatchID          BatchID       // the batch it came from
	CustomID         string        // the item's CustomID
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
// primary fails with tokens and the fallback then also fails: the
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

// WithBatchUsage wraps inner and calls record once for each batch item that used
// tokens, when you read the results. Failed, expired and canceled items have no
// tokens and are not reported. A refused item that used tokens is reported with
// Err set.
//
// Each Usage has Batch true, BatchID and CustomID set, and Duration 0. CostUSD is
// book.Cost times the provider's batch factor (see batchFactor). Task is TaskEmbed
// for an embed batch. For a chat batch Task is empty: the batch ID does not carry
// the Task and aikit keeps no state. Your app knows what it submitted; BatchID is
// in the record.
//
// book prices the items. A nil book uses DefaultPrices.
//
// Reading the results twice records twice. Save the record with an upsert on
// (BatchID, CustomID). record runs on the reading goroutine, before the item is
// yielded. If the reader stops early, items not yielded are not recorded. The ctx
// may already be cancelled; use context.WithoutCancel(ctx) to save.
//
// Submit, Status, Cancel and Enabled pass straight through.
//
// WithBatchUsage panics if inner or record is nil.
func WithBatchUsage(inner Batcher, book *PriceBook, record func(ctx context.Context, u Usage)) Batcher {
	if inner == nil {
		panic("llm: WithBatchUsage needs an inner Batcher")
	}
	if record == nil {
		panic("llm: WithBatchUsage needs a record func")
	}
	return &batchUsage{inner: inner, book: book, record: record}
}

type batchUsage struct {
	inner  Batcher
	book   *PriceBook
	record func(ctx context.Context, u Usage)
}

func (c *batchUsage) SubmitChat(ctx context.Context, b ChatBatch) (BatchID, error) {
	return c.inner.SubmitChat(ctx, b)
}

func (c *batchUsage) SubmitEmbed(ctx context.Context, b EmbedBatch) (BatchID, error) {
	return c.inner.SubmitEmbed(ctx, b)
}

func (c *batchUsage) Status(ctx context.Context, id BatchID) (BatchStatus, error) {
	return c.inner.Status(ctx, id)
}

func (c *batchUsage) Cancel(ctx context.Context, id BatchID) error { return c.inner.Cancel(ctx, id) }

func (c *batchUsage) Enabled() bool { return c.inner.Enabled() }

func (c *batchUsage) Results(ctx context.Context, id BatchID) iter.Seq2[BatchResult, error] {
	return func(yield func(BatchResult, error) bool) {
		var task Task
		if _, kind, _, _, err := parseBatchID(id); err == nil && kind == batchEmbed {
			task = TaskEmbed
		}
		for r, err := range c.inner.Results(ctx, id) {
			if err == nil && hasUsage(r.Response) {
				resp := r.Response
				c.record(ctx, Usage{
					Provider:         resp.Provider,
					Model:            resp.Model,
					Task:             task,
					InputTokens:      resp.InputTokens,
					OutputTokens:     resp.OutputTokens,
					CachedTokens:     resp.CachedTokens,
					CacheWriteTokens: resp.CacheWriteTokens,
					CostUSD:          c.book.Cost(resp.Model, resp.InputTokens, resp.OutputTokens, resp.CachedTokens, resp.CacheWriteTokens) * batchFactor(resp.Provider),
					Estimated:        resp.UsageEstimated,
					Err:              r.Err,
					Batch:            true,
					BatchID:          id,
					CustomID:         r.CustomID,
				})
			}
			if !yield(r, err) {
				return
			}
		}
	}
}
