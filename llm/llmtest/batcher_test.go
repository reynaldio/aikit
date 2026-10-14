package llmtest

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/reynaldio/aikit/llm"
)

func chatBatch(ids ...string) llm.ChatBatch {
	var cb llm.ChatBatch
	for _, id := range ids {
		cb.Items = append(cb.Items, llm.ChatItem{CustomID: id, Request: llm.Request{SystemCacheable: "p-" + id}})
	}
	return cb
}

func TestBatcherNotHeld(t *testing.T) { // F1
	ctx := context.Background()
	c := New(Reply{Response: llm.Response{Text: "a"}}, Reply{Err: errors.New("boom")})
	b := NewBatcher(c, nil)
	id, err := b.SubmitChat(ctx, chatBatch("x", "y"))
	if err != nil {
		t.Fatal(err)
	}
	if id != "llmtest:chat:fake:1" {
		t.Fatalf("id %q", id)
	}
	st, err := b.Status(ctx, id)
	if err != nil || st.State != llm.BatchDone || st.Total != 2 || st.Succeeded != 1 || st.Failed != 1 {
		t.Fatalf("status %+v %v", st, err)
	}
	res, err := llm.CollectResults(b.Results(ctx, id))
	if err != nil || len(res) != 2 {
		t.Fatal(res, err)
	}
	if res[0].CustomID != "x" || res[0].Response.Text != "a" || res[1].CustomID != "y" || res[1].Err == nil {
		t.Fatalf("results %+v", res)
	}
	if !b.Enabled() {
		t.Fatal("not enabled")
	}
}

func TestBatcherEmbed(t *testing.T) { // F2
	ctx := context.Background()
	b := NewBatcher(nil, NewEmbedder(6))
	id, err := b.SubmitEmbed(ctx, llm.EmbedBatch{Kind: llm.EmbedDocument, Items: []llm.EmbedItem{{CustomID: "a", Text: "cats"}}})
	if err != nil {
		t.Fatal(err)
	}
	if id != "llmtest:embed:fake:1" {
		t.Fatalf("id %q", id)
	}
	res, err := llm.CollectResults(b.Results(ctx, id))
	if err != nil || len(res) != 1 {
		t.Fatal(res, err)
	}
	want, _ := NewEmbedder(6).Embed(ctx, llm.EmbedRequest{Texts: []string{"cats"}, Kind: llm.EmbedDocument})
	if !reflect.DeepEqual(res[0].Vector, want.Vectors[0]) {
		t.Fatal("vector differs")
	}
	var sum float64
	for _, x := range res[0].Vector {
		sum += float64(x) * float64(x)
	}
	if math.Abs(sum-1) > 1e-5 {
		t.Fatalf("norm %v", sum)
	}
	if res[0].Response.Model != "fake" || res[0].Response.Provider != "llmtest" {
		t.Fatalf("resp %+v", res[0].Response)
	}
}

func TestBatcherHoldRelease(t *testing.T) { // F3
	ctx := context.Background()
	b := NewBatcher(New(Reply{Response: llm.Response{Text: "a"}}), nil)
	b.Hold()
	id, _ := b.SubmitChat(ctx, chatBatch("x"))
	st, _ := b.Status(ctx, id)
	if st.State != llm.BatchRunning || st.Total != 1 {
		t.Fatalf("status %+v", st)
	}
	b.Release("nope")
	b.Release(id)
	st, _ = b.Status(ctx, id)
	if st.State != llm.BatchDone || st.Succeeded != 1 {
		t.Fatalf("status %+v", st)
	}
	b.Release(id) // no-op
	res, err := llm.CollectResults(b.Results(ctx, id))
	if err != nil || len(res) != 1 || res[0].Response.Text != "a" {
		t.Fatal(res, err)
	}
}

func TestBatcherCancel(t *testing.T) { // F4
	ctx := context.Background()
	b := NewBatcher(New(), nil)
	b.Hold()
	id, _ := b.SubmitChat(ctx, chatBatch("x", "y"))
	if err := b.Cancel(ctx, id); err != nil {
		t.Fatal(err)
	}
	st, _ := b.Status(ctx, id)
	if st.State != llm.BatchDone || st.Canceled != 2 {
		t.Fatalf("status %+v", st)
	}
	res, err := llm.CollectResults(b.Results(ctx, id))
	if err != nil || len(res) != 2 || !errors.Is(res[0].Err, llm.ErrBatchItemCanceled) {
		t.Fatal(res, err)
	}
	if err := b.Cancel(ctx, id); err != nil {
		t.Fatal(err)
	}
	if err := b.Cancel(ctx, "llmtest:chat:fake:99"); !errors.Is(err, llm.ErrBadBatchID) {
		t.Fatal(err)
	}
	if _, err := b.Status(ctx, "zzz"); !errors.Is(err, llm.ErrBadBatchID) {
		t.Fatal(err)
	}
	if _, err := llm.CollectResults(b.Results(ctx, "zzz")); !errors.Is(err, llm.ErrBadBatchID) {
		t.Fatal(err)
	}
}

func TestBatcherFail(t *testing.T) { // F5
	ctx := context.Background()
	b := NewBatcher(New(Reply{}), nil)
	boom := errors.New("boom")
	b.Fail(boom)
	if _, err := b.SubmitChat(ctx, chatBatch("x")); !errors.Is(err, boom) {
		t.Fatal(err)
	}
	id, err := b.SubmitChat(ctx, chatBatch("x"))
	if err != nil {
		t.Fatal(err)
	}
	b.Fail(boom)
	if _, err := llm.CollectResults(b.Results(ctx, id)); !errors.Is(err, boom) {
		t.Fatal(err)
	}
	if _, err := llm.CollectResults(b.Results(ctx, id)); err != nil {
		t.Fatal(err)
	}
	b.Fail(boom)
	if _, err := b.Status(ctx, id); !errors.Is(err, boom) {
		t.Fatal(err)
	}
	b.Fail(boom)
	if err := b.Cancel(ctx, id); !errors.Is(err, boom) {
		t.Fatal(err)
	}
}

func TestBatcherSubmitsCopy(t *testing.T) { // F6
	ctx := context.Background()
	b := NewBatcher(New(Reply{}), NewEmbedder(4))
	cb := chatBatch("x")
	if _, err := b.SubmitChat(ctx, cb); err != nil {
		t.Fatal(err)
	}
	cb.Items[0].CustomID = "changed"
	if _, err := b.SubmitEmbed(ctx, llm.EmbedBatch{Items: []llm.EmbedItem{{CustomID: "e", Text: "t"}}}); err != nil {
		t.Fatal(err)
	}
	s := b.Submits()
	if len(s) != 2 {
		t.Fatal(len(s))
	}
	got := s[0].(llm.ChatBatch)
	if got.Items[0].CustomID != "x" {
		t.Fatal("recorded batch changed with caller's slice")
	}
	got.Items[0].CustomID = "mut"
	if b.Submits()[0].(llm.ChatBatch).Items[0].CustomID != "x" {
		t.Fatal("Submits is not a copy")
	}
	if _, ok := s[1].(llm.EmbedBatch); !ok {
		t.Fatal("second is not EmbedBatch")
	}
}

func TestBatcherBadIDs(t *testing.T) { // F7
	ctx := context.Background()
	b := NewBatcher(New(), nil)
	for _, bad := range []string{"", "a b", "a.b", string(make([]byte, 65))} {
		if _, err := b.SubmitChat(ctx, chatBatch(bad)); err == nil {
			t.Fatalf("accepted %q", bad)
		}
	}
	if _, err := b.SubmitChat(ctx, chatBatch("a", "a")); err == nil {
		t.Fatal("accepted duplicate")
	}
	if _, err := b.SubmitChat(ctx, llm.ChatBatch{}); err == nil {
		t.Fatal("accepted empty")
	}
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := b.SubmitChat(cctx, chatBatch("a")); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if len(b.Submits()) != 0 {
		t.Fatal("failed submits were recorded")
	}
}

func TestBatcherRunningResults(t *testing.T) { // F8
	ctx := context.Background()
	b := NewBatcher(New(), nil)
	b.Hold()
	id, _ := b.SubmitChat(ctx, chatBatch("x"))
	if _, err := llm.CollectResults(b.Results(ctx, id)); !errors.Is(err, llm.ErrBatchNotDone) {
		t.Fatal(err)
	}
}

func TestBatcherNilFakes(t *testing.T) { // F9
	ctx := context.Background()
	b := NewBatcher(nil, nil)
	if _, err := b.SubmitChat(ctx, chatBatch("x")); err == nil || err.Error() != "llmtest: no fake Client for chat batches" {
		t.Fatal(err)
	}
	if _, err := b.SubmitEmbed(ctx, llm.EmbedBatch{Items: []llm.EmbedItem{{CustomID: "x", Text: "t"}}}); err == nil {
		t.Fatal("no error")
	}
}

func TestBatcherConcurrent(t *testing.T) { // F10
	ctx := context.Background()
	replies := make([]Reply, 50)
	b := NewBatcher(New(replies...), nil)
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id, err := b.SubmitChat(ctx, chatBatch(fmt.Sprintf("i%d", i)))
			if err != nil {
				t.Error(err)
				return
			}
			if _, err := b.Status(ctx, id); err != nil {
				t.Error(err)
			}
			if _, err := llm.CollectResults(b.Results(ctx, id)); err != nil {
				t.Error(err)
			}
			_ = b.Submits()
		}(i)
	}
	wg.Wait()
	if len(b.Submits()) != 50 {
		t.Fatal(len(b.Submits()))
	}
}
