package guard

import (
	"context"
	"slices"

	"github.com/reynaldio/aikit/llm"
)

type batcher struct {
	llm.Batcher
	own   []string
	audit AuditSink
}

// WrapBatcher returns inner with the background-class guardrails applied to chat
// batches: DataRule is added to every item's system message, and every item is
// scanned for fenced prompt injection (audited through o.Audit, once per source
// per item). A batch has no Turn, no size limit and no rate limit. Only
// OwnSources and Audit of o are used. A disabled inner is returned as is.
// SubmitEmbed, Status, Results, Cancel and Enabled pass through.
func WrapBatcher(inner llm.Batcher, o Options) llm.Batcher {
	if inner == nil {
		panic("guard: WrapBatcher needs an inner Batcher")
	}
	if !inner.Enabled() {
		return inner
	}
	own := o.OwnSources
	if own == nil {
		own = []string{SourceTranscript, SourceMemory}
	}
	return &batcher{Batcher: inner, own: slices.Clone(own), audit: o.Audit}
}

// SubmitChat guards a copy of the batch; the caller's batch is never changed.
func (b *batcher) SubmitChat(ctx context.Context, cb llm.ChatBatch) (llm.BatchID, error) {
	items := make([]llm.ChatItem, len(cb.Items))
	for i, it := range cb.Items {
		it.Request = withPolicies(it.Request, DataRule)
		scanRequest(ctx, it.Request, b.own, b.audit, nil)
		items[i] = it
	}
	cb.Items = items
	return b.Batcher.SubmitChat(ctx, cb)
}
