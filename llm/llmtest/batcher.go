package llmtest

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"regexp"
	"sync"

	"github.com/reynaldio/aikit/llm"
)

var customIDRe = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

type fakeBatch struct {
	kind    string
	chat    []llm.ChatItem
	embed   llm.EmbedBatch
	total   int
	done    bool
	results []llm.BatchResult
}

// Batcher is a fake llm.Batcher. It runs batch items through a fake Client
// (chat) or Embedder (embed). It is safe for concurrent use.
//
//	b := llmtest.NewBatcher(llmtest.New(llmtest.Reply{...}), llmtest.NewEmbedder(8))
//	id, _ := b.SubmitChat(ctx, batch) // done at once, unless b.Hold() was called
type Batcher struct {
	mu       sync.Mutex
	chat     *Client
	emb      *Embedder
	hold     bool
	n        int
	batches  map[llm.BatchID]*fakeBatch
	submits  []any
	failNext error
}

var _ llm.Batcher = (*Batcher)(nil)

// NewBatcher returns a Batcher. chat or emb may be nil: a submit of that kind
// then returns an error.
func NewBatcher(chat *Client, emb *Embedder) *Batcher {
	return &Batcher{chat: chat, emb: emb, batches: map[llm.BatchID]*fakeBatch{}}
}

// Enabled always reports true.
func (b *Batcher) Enabled() bool { return true }

// Hold makes later submits stay running until Release.
func (b *Batcher) Hold() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.hold = true
}

// Release runs a held batch. An unknown or finished ID is ignored.
func (b *Batcher) Release(id llm.BatchID) {
	b.mu.Lock()
	fb := b.batches[id]
	if fb == nil || fb.done {
		b.mu.Unlock()
		return
	}
	// Mark as taken so a second Release does not run it twice.
	fb.done = true
	b.mu.Unlock()
	res := b.run(context.Background(), fb)
	b.mu.Lock()
	fb.results = res
	b.mu.Unlock()
}

// Submits returns a copy of the batches received so far, as llm.ChatBatch or
// llm.EmbedBatch values, in order.
func (b *Batcher) Submits() []any {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]any, len(b.submits))
	for i, s := range b.submits {
		switch v := s.(type) {
		case llm.ChatBatch:
			v.Items = append([]llm.ChatItem(nil), v.Items...)
			out[i] = v
		case llm.EmbedBatch:
			v.Items = append([]llm.EmbedItem(nil), v.Items...)
			out[i] = v
		}
	}
	return out
}

// Fail makes the next method call return err (for Results, the first pull).
// Later calls work again.
func (b *Batcher) Fail(err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failNext = err
}

// takeFail returns and clears the pending error. Caller holds b.mu.
func (b *Batcher) takeFail() error {
	err := b.failNext
	b.failNext = nil
	return err
}

func checkIDs(ids []string) error {
	if len(ids) == 0 {
		return errors.New("llmtest: batch has no items")
	}
	seen := make(map[string]bool, len(ids))
	for _, id := range ids {
		if !customIDRe.MatchString(id) {
			return fmt.Errorf("llmtest: bad CustomID %q", id)
		}
		if seen[id] {
			return fmt.Errorf("llmtest: duplicate CustomID %q", id)
		}
		seen[id] = true
	}
	return nil
}

// SubmitChat records cb and runs it, unless Hold is on.
func (b *Batcher) SubmitChat(ctx context.Context, cb llm.ChatBatch) (llm.BatchID, error) {
	items := append([]llm.ChatItem(nil), cb.Items...)
	ids := make([]string, len(items))
	for i, it := range items {
		ids[i] = it.CustomID
	}
	rec := cb
	rec.Items = append([]llm.ChatItem(nil), items...)
	return b.submit(ctx, "chat", b.chat != nil, "no fake Client for chat batches", ids, rec,
		&fakeBatch{kind: "chat", chat: items, total: len(items)})
}

// SubmitEmbed records eb and runs it, unless Hold is on.
func (b *Batcher) SubmitEmbed(ctx context.Context, eb llm.EmbedBatch) (llm.BatchID, error) {
	items := append([]llm.EmbedItem(nil), eb.Items...)
	ids := make([]string, len(items))
	for i, it := range items {
		ids[i] = it.CustomID
	}
	eb.Items = items
	rec := eb
	rec.Items = append([]llm.EmbedItem(nil), items...)
	return b.submit(ctx, "embed", b.emb != nil, "no fake Embedder for embed batches", ids, rec,
		&fakeBatch{kind: "embed", embed: eb, total: len(items)})
}

func (b *Batcher) submit(ctx context.Context, kind string, ok bool, noFake string, ids []string, rec any, fb *fakeBatch) (llm.BatchID, error) {
	b.mu.Lock()
	if err := b.takeFail(); err != nil {
		b.mu.Unlock()
		return "", err
	}
	if err := ctx.Err(); err != nil {
		b.mu.Unlock()
		return "", err
	}
	if !ok {
		b.mu.Unlock()
		return "", errors.New("llmtest: " + noFake)
	}
	if err := checkIDs(ids); err != nil {
		b.mu.Unlock()
		return "", err
	}
	b.n++
	id := llm.BatchID(fmt.Sprintf("llmtest:%s:fake:%d", kind, b.n))
	b.batches[id] = fb
	b.submits = append(b.submits, rec)
	held := b.hold
	fb.done = !held
	b.mu.Unlock()

	if !held {
		res := b.run(ctx, fb)
		b.mu.Lock()
		fb.results = res
		b.mu.Unlock()
	}
	return id, nil
}

// run executes every item in order. It does not touch b.mu.
func (b *Batcher) run(ctx context.Context, fb *fakeBatch) []llm.BatchResult {
	var out []llm.BatchResult
	if fb.kind == "chat" {
		for _, it := range fb.chat {
			resp, err := b.chat.Complete(ctx, it.Request)
			out = append(out, llm.BatchResult{CustomID: it.CustomID, Response: resp, Err: err})
		}
		return out
	}
	for _, it := range fb.embed.Items {
		resp, err := b.emb.Embed(ctx, llm.EmbedRequest{
			Texts: []string{it.Text}, Kind: fb.embed.Kind,
			Dimensions: fb.embed.Dimensions, Model: fb.embed.Model,
		})
		r := llm.BatchResult{CustomID: it.CustomID, Err: err}
		if err == nil && len(resp.Vectors) > 0 {
			r.Vector = resp.Vectors[0]
			r.Response = llm.Response{Provider: resp.Provider, Model: resp.Model, InputTokens: resp.InputTokens}
		}
		out = append(out, r)
	}
	return out
}

// statusLocked builds the status. Caller holds b.mu.
func statusOf(fb *fakeBatch) llm.BatchStatus {
	st := llm.BatchStatus{State: llm.BatchRunning, Total: fb.total}
	// A released batch is marked done before its results are stored; it still
	// reads as running until they are.
	if !fb.done || fb.results == nil {
		return st
	}
	st.State = llm.BatchDone
	for _, r := range fb.results {
		switch {
		case r.Err == nil:
			st.Succeeded++
		case errors.Is(r.Err, llm.ErrBatchItemCanceled):
			st.Canceled++
		default:
			st.Failed++
		}
	}
	return st
}

// Status reports the batch state and counts.
func (b *Batcher) Status(ctx context.Context, id llm.BatchID) (llm.BatchStatus, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.takeFail(); err != nil {
		return llm.BatchStatus{}, err
	}
	fb := b.batches[id]
	if fb == nil {
		return llm.BatchStatus{}, llm.ErrBadBatchID
	}
	return statusOf(fb), nil
}

// Cancel ends a held batch with every item canceled. A done batch is left as is.
func (b *Batcher) Cancel(ctx context.Context, id llm.BatchID) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.takeFail(); err != nil {
		return err
	}
	fb := b.batches[id]
	if fb == nil {
		return llm.ErrBadBatchID
	}
	if fb.done {
		return nil
	}
	var ids []string
	if fb.kind == "chat" {
		for _, it := range fb.chat {
			ids = append(ids, it.CustomID)
		}
	} else {
		for _, it := range fb.embed.Items {
			ids = append(ids, it.CustomID)
		}
	}
	fb.results = make([]llm.BatchResult, len(ids))
	for i, cid := range ids {
		fb.results[i] = llm.BatchResult{CustomID: cid, Err: llm.ErrBatchItemCanceled}
	}
	fb.done = true
	return nil
}

// Results yields the items of a done batch, in item order.
func (b *Batcher) Results(ctx context.Context, id llm.BatchID) iter.Seq2[llm.BatchResult, error] {
	return func(yield func(llm.BatchResult, error) bool) {
		b.mu.Lock()
		if err := b.takeFail(); err != nil {
			b.mu.Unlock()
			yield(llm.BatchResult{}, err)
			return
		}
		fb := b.batches[id]
		if fb == nil {
			b.mu.Unlock()
			yield(llm.BatchResult{}, llm.ErrBadBatchID)
			return
		}
		if statusOf(fb).State != llm.BatchDone {
			b.mu.Unlock()
			yield(llm.BatchResult{}, llm.ErrBatchNotDone)
			return
		}
		res := append([]llm.BatchResult(nil), fb.results...)
		b.mu.Unlock()
		for _, r := range res {
			if !yield(r, nil) {
				return
			}
		}
	}
}
