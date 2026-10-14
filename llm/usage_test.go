package llm_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/reynaldio/aikit/llm"
	"github.com/reynaldio/aikit/llm/llmtest"
)

type usageRec struct{ got []llm.Usage }

func (r *usageRec) record(_ context.Context, u llm.Usage) { r.got = append(r.got, u) }

func TestWithUsageSuccess(t *testing.T) {
	book := llm.NewPriceBook(nil, map[string]llm.Rate{"m": {Input: 1, Output: 2, CachedRead: 0.5, CacheWrite: 4}})
	inner := llmtest.New(llmtest.Reply{Response: llm.Response{
		Text: "hi", Provider: llm.ProviderAnthropic, Model: "m",
		InputTokens: 1_000_000, OutputTokens: 1_000_000, CachedTokens: 1_000_000, CacheWriteTokens: 1_000_000,
	}})
	var rec usageRec
	c := llm.WithUsage(inner, book, rec.record)
	resp, err := c.Complete(context.Background(), llm.Request{Task: llm.TaskChat})
	if err != nil || resp.Text != "hi" {
		t.Fatalf("resp = %+v err = %v", resp, err)
	}
	if len(rec.got) != 1 {
		t.Fatalf("records = %d, want 1", len(rec.got))
	}
	u := rec.got[0]
	if u.Provider != llm.ProviderAnthropic || u.Model != "m" || u.Task != llm.TaskChat ||
		u.InputTokens != 1_000_000 || u.OutputTokens != 1_000_000 || u.CachedTokens != 1_000_000 || u.CacheWriteTokens != 1_000_000 {
		t.Fatalf("usage = %+v", u)
	}
	if math.Abs(u.CostUSD-7.5) > 1e-9 {
		t.Fatalf("cost = %v, want 7.5", u.CostUSD)
	}
	if u.Err != nil || u.Estimated || u.Duration < 0 {
		t.Fatalf("usage = %+v", u)
	}
}

func TestWithUsageRefusedWithTokens(t *testing.T) {
	inner := llmtest.New(llmtest.Reply{
		Response: llm.Response{Model: "m", InputTokens: 7},
		Err:      llm.ErrRefused,
	})
	var rec usageRec
	c := llm.WithUsage(inner, nil, rec.record)
	_, err := c.Complete(context.Background(), llm.Request{})
	if !errors.Is(err, llm.ErrRefused) {
		t.Fatalf("err = %v", err)
	}
	if len(rec.got) != 1 || !errors.Is(rec.got[0].Err, llm.ErrRefused) || rec.got[0].InputTokens != 7 {
		t.Fatalf("records = %+v", rec.got)
	}
}

func TestWithUsageTransportErrorNoRecord(t *testing.T) {
	boom := errors.New("boom")
	inner := llmtest.New(llmtest.Reply{Err: boom})
	var rec usageRec
	c := llm.WithUsage(inner, nil, rec.record)
	if _, err := c.Complete(context.Background(), llm.Request{}); !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
	if len(rec.got) != 0 {
		t.Fatalf("records = %+v, want none", rec.got)
	}
}

func TestWithUsageEstimatedCopied(t *testing.T) {
	inner := llmtest.New(llmtest.Reply{Response: llm.Response{Model: "m", OutputTokens: 3, UsageEstimated: true}})
	var rec usageRec
	c := llm.WithUsage(inner, nil, rec.record)
	if _, err := c.Complete(context.Background(), llm.Request{}); err != nil {
		t.Fatal(err)
	}
	if len(rec.got) != 1 || !rec.got[0].Estimated {
		t.Fatalf("records = %+v", rec.got)
	}
}

func TestWithUsageNilBookUsesDefaultPrices(t *testing.T) {
	inner := llmtest.New(llmtest.Reply{Response: llm.Response{Model: "claude-haiku-4-5", InputTokens: 1_000_000, OutputTokens: 1_000_000}})
	var rec usageRec
	c := llm.WithUsage(inner, nil, rec.record)
	if _, err := c.Complete(context.Background(), llm.Request{}); err != nil {
		t.Fatal(err)
	}
	want := llm.DefaultPrices["claude-haiku-4-5"].Cost(1_000_000, 1_000_000, 0, 0)
	if want <= 0 {
		t.Fatalf("want = %v, test is meaningless", want)
	}
	if len(rec.got) != 1 || math.Abs(rec.got[0].CostUSD-want) > 1e-9 {
		t.Fatalf("records = %+v, want cost %v", rec.got, want)
	}
}

func TestWithUsageReturnsReplyAndKeepsEvents(t *testing.T) {
	boom := errors.New("boom")
	inner := llmtest.New(llmtest.Reply{
		Response: llm.Response{Text: "hello", Model: "m", InputTokens: 1},
		Err:      boom,
		Events:   []llm.StreamEvent{{Kind: llm.StreamText, Text: "hello"}},
	})
	var rec usageRec
	c := llm.WithUsage(inner, nil, rec.record)
	var text string
	resp, err := c.Complete(context.Background(), llm.Request{OnEvent: func(ev llm.StreamEvent) {
		if ev.Kind == llm.StreamText {
			text += ev.Text
		}
	}})
	if !errors.Is(err, boom) || resp.Text != "hello" || resp.InputTokens != 1 {
		t.Fatalf("resp = %+v err = %v", resp, err)
	}
	if text != "hello" {
		t.Fatalf("events text = %q, want hello", text)
	}
}

func TestWithUsagePassThroughAndPanics(t *testing.T) {
	inner := llmtest.New()
	c := llm.WithUsage(inner, nil, func(context.Context, llm.Usage) {})
	if !c.Enabled() || c.CompareTargets() != nil {
		t.Fatal("Enabled/CompareTargets not passed through")
	}
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: want panic", name)
			}
		}()
		f()
	}
	mustPanic("nil record", func() { llm.WithUsage(inner, nil, nil) })
	mustPanic("nil inner", func() { llm.WithUsage(nil, nil, func(context.Context, llm.Usage) {}) })
}
