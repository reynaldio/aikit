package llm

import (
	"context"
	"errors"
	"iter"
	"math"
	"testing"
)

type fakeBatcher struct {
	items   []BatchResult
	iterErr error // yielded after items
	calls   []string
}

func (f *fakeBatcher) SubmitChat(ctx context.Context, b ChatBatch) (BatchID, error) {
	f.calls = append(f.calls, "SubmitChat")
	return "openai:chat:m:1", nil
}
func (f *fakeBatcher) SubmitEmbed(ctx context.Context, b EmbedBatch) (BatchID, error) {
	f.calls = append(f.calls, "SubmitEmbed")
	return "openai:embed:m:1", nil
}
func (f *fakeBatcher) Status(ctx context.Context, id BatchID) (BatchStatus, error) {
	f.calls = append(f.calls, "Status")
	return BatchStatus{State: BatchDone, Total: 7}, nil
}
func (f *fakeBatcher) Cancel(ctx context.Context, id BatchID) error {
	f.calls = append(f.calls, "Cancel")
	return nil
}
func (f *fakeBatcher) Enabled() bool { f.calls = append(f.calls, "Enabled"); return true }
func (f *fakeBatcher) Results(ctx context.Context, id BatchID) iter.Seq2[BatchResult, error] {
	return func(yield func(BatchResult, error) bool) {
		for _, r := range f.items {
			if !yield(r, nil) {
				return
			}
		}
		if f.iterErr != nil {
			yield(BatchResult{}, f.iterErr)
		}
	}
}

func usageBook() *PriceBook {
	return NewPriceBook(nil, map[string]Rate{
		"m": {Input: 1, Output: 2, CachedRead: 0.5, CacheWrite: 4},
	})
}

func okItem(id string, p Provider) BatchResult {
	return BatchResult{CustomID: id, Response: Response{
		Provider: p, Model: "m", InputTokens: 1_000_000, OutputTokens: 1_000_000,
		CachedTokens: 1_000_000, CacheWriteTokens: 1_000_000,
	}}
}

// full price: 1 + 2 + 0.5 + 4 = 7.5
const fullPrice = 7.5

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func readAll(t *testing.T, b Batcher, id BatchID) []BatchResult {
	t.Helper()
	var out []BatchResult
	for r, err := range b.Results(context.Background(), id) {
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		out = append(out, r)
	}
	return out
}

func TestBatchUsageRecordsEachBilledItem(t *testing.T) { // U1
	f := &fakeBatcher{items: []BatchResult{okItem("a", ProviderAnthropic), okItem("b", ProviderAnthropic)}}
	f.items[1].Response.UsageEstimated = true
	var got []Usage
	b := WithBatchUsage(f, usageBook(), func(ctx context.Context, u Usage) { got = append(got, u) })
	id := BatchID("anthropic:chat:m:xyz")
	if n := len(readAll(t, b, id)); n != 2 {
		t.Fatalf("yielded %d", n)
	}
	if len(got) != 2 {
		t.Fatalf("recorded %d", len(got))
	}
	u := got[0]
	if u.Provider != ProviderAnthropic || u.Model != "m" || u.Task != "" ||
		u.InputTokens != 1_000_000 || u.OutputTokens != 1_000_000 || u.CachedTokens != 1_000_000 || u.CacheWriteTokens != 1_000_000 ||
		!near(u.CostUSD, fullPrice*0.5) || u.Estimated || u.Duration != 0 || u.Err != nil ||
		!u.Batch || u.BatchID != id || u.CustomID != "a" {
		t.Fatalf("bad usage: %+v", u)
	}
	if !got[1].Estimated || got[1].CustomID != "b" {
		t.Fatalf("bad second: %+v", got[1])
	}
}

func TestBatchFactorCost(t *testing.T) { // U2
	for p, f := range map[Provider]float64{
		ProviderAnthropic: 0.5, ProviderOpenAI: 0.5, ProviderGoogle: 0.5, ProviderMoonshot: 0.6,
		ProviderDeepSeek: 1, "other": 1,
	} {
		if got := batchFactor(p); got != f {
			t.Errorf("batchFactor(%s)=%v want %v", p, got, f)
		}
		fb := &fakeBatcher{items: []BatchResult{okItem("a", p)}}
		var got []Usage
		b := WithBatchUsage(fb, usageBook(), func(ctx context.Context, u Usage) { got = append(got, u) })
		readAll(t, b, BatchID(string(p)+":chat:m:1"))
		if len(got) != 1 || !near(got[0].CostUSD, fullPrice*f) {
			t.Errorf("%s: cost %+v want %v", p, got, fullPrice*f)
		}
	}
}

func TestBatchUsageSkipsUnbilledItems(t *testing.T) { // U3
	f := &fakeBatcher{items: []BatchResult{
		{CustomID: "f", Err: errors.New("failed")},
		{CustomID: "e", Err: ErrBatchItemExpired},
		{CustomID: "c", Err: ErrBatchItemCanceled},
	}}
	n := 0
	b := WithBatchUsage(f, usageBook(), func(ctx context.Context, u Usage) { n++ })
	if got := len(readAll(t, b, "openai:chat:m:1")); got != 3 {
		t.Fatalf("yielded %d", got)
	}
	if n != 0 {
		t.Fatalf("recorded %d", n)
	}
}

func TestBatchUsageRefusedRecordedWithErr(t *testing.T) { // U4
	it := okItem("a", ProviderOpenAI)
	it.Err = ErrRefused
	f := &fakeBatcher{items: []BatchResult{it}}
	var got []Usage
	b := WithBatchUsage(f, usageBook(), func(ctx context.Context, u Usage) { got = append(got, u) })
	readAll(t, b, "openai:chat:m:1")
	if len(got) != 1 || !errors.Is(got[0].Err, ErrRefused) {
		t.Fatalf("got %+v", got)
	}
}

func TestBatchUsageEmbedTask(t *testing.T) { // U5
	it := BatchResult{CustomID: "a", Response: Response{Provider: ProviderOpenAI, Model: "m", InputTokens: 1_000_000}}
	f := &fakeBatcher{items: []BatchResult{it}}
	var got []Usage
	b := WithBatchUsage(f, usageBook(), func(ctx context.Context, u Usage) { got = append(got, u) })
	readAll(t, b, "openai:embed:m:1")
	if len(got) != 1 || got[0].Task != TaskEmbed || !near(got[0].CostUSD, 0.5) {
		t.Fatalf("got %+v", got)
	}
}

func TestBatchUsageEarlyBreak(t *testing.T) { // U6
	f := &fakeBatcher{items: []BatchResult{okItem("a", ProviderOpenAI), okItem("b", ProviderOpenAI), okItem("c", ProviderOpenAI)}}
	var got []Usage
	b := WithBatchUsage(f, usageBook(), func(ctx context.Context, u Usage) { got = append(got, u) })
	for r := range b.Results(context.Background(), "openai:chat:m:1") {
		if r.CustomID == "b" {
			break
		}
	}
	if len(got) != 2 || got[1].CustomID != "b" {
		t.Fatalf("got %+v", got)
	}
}

func TestBatchUsageIterErrorPassedOn(t *testing.T) { // U7
	boom := errors.New("boom")
	f := &fakeBatcher{iterErr: boom}
	n := 0
	b := WithBatchUsage(f, usageBook(), func(ctx context.Context, u Usage) { n++ })
	var gotErr error
	for _, err := range b.Results(context.Background(), "openai:chat:m:1") {
		gotErr = err
	}
	if !errors.Is(gotErr, boom) || n != 0 {
		t.Fatalf("err=%v n=%d", gotErr, n)
	}
}

func TestBatchUsagePanics(t *testing.T) { // U8
	check := func(want string, fn func()) {
		t.Helper()
		defer func() {
			if r := recover(); r != want {
				t.Fatalf("panic %v want %q", r, want)
			}
		}()
		fn()
	}
	check("llm: WithBatchUsage needs an inner Batcher", func() {
		WithBatchUsage(nil, nil, func(context.Context, Usage) {})
	})
	check("llm: WithBatchUsage needs a record func", func() {
		WithBatchUsage(&fakeBatcher{}, nil, nil)
	})
}

func TestBatchUsageNilBookUsesDefaults(t *testing.T) { // U9
	it := BatchResult{CustomID: "a", Response: Response{Provider: ProviderOpenAI, Model: "text-embedding-3-small", InputTokens: 1_000_000}}
	f := &fakeBatcher{items: []BatchResult{it}}
	var got []Usage
	b := WithBatchUsage(f, nil, func(ctx context.Context, u Usage) { got = append(got, u) })
	readAll(t, b, "openai:embed:text-embedding-3-small:1")
	if len(got) != 1 || !near(got[0].CostUSD, 0.01) {
		t.Fatalf("got %+v", got)
	}
}

func TestBatchUsagePassThrough(t *testing.T) { // U10
	f := &fakeBatcher{}
	b := WithBatchUsage(f, nil, func(context.Context, Usage) {})
	ctx := context.Background()
	if id, err := b.SubmitChat(ctx, ChatBatch{}); err != nil || id != "openai:chat:m:1" {
		t.Fatal(id, err)
	}
	if id, err := b.SubmitEmbed(ctx, EmbedBatch{}); err != nil || id != "openai:embed:m:1" {
		t.Fatal(id, err)
	}
	if st, err := b.Status(ctx, "x"); err != nil || st.Total != 7 {
		t.Fatal(st, err)
	}
	if err := b.Cancel(ctx, "x"); err != nil {
		t.Fatal(err)
	}
	if !b.Enabled() {
		t.Fatal("not enabled")
	}
	if len(f.calls) != 5 {
		t.Fatalf("calls %v", f.calls)
	}
}
