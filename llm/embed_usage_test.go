package llm_test

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/reynaldio/aikit/llm"
)

// fakeEmbedder returns a canned reply.
type fakeEmbedder struct {
	resp    llm.EmbedResponse
	err     error
	enabled bool
}

func (f *fakeEmbedder) Embed(context.Context, llm.EmbedRequest) (llm.EmbedResponse, error) {
	return f.resp, f.err
}
func (f *fakeEmbedder) Enabled() bool { return f.enabled }

func TestWithEmbedUsageSuccess(t *testing.T) {
	book := llm.NewPriceBook(nil, map[string]llm.Rate{"em": {Input: 0.5}})
	inner := &fakeEmbedder{resp: llm.EmbedResponse{
		Vectors: [][]float32{{1}}, Provider: llm.ProviderOpenAI, Model: "em", InputTokens: 2_000_000,
	}}
	var rec usageRec
	e := llm.WithEmbedUsage(inner, book, rec.record)
	resp, err := e.Embed(context.Background(), llm.EmbedRequest{Texts: []string{"a"}})
	if err != nil || len(resp.Vectors) != 1 || resp.InputTokens != 2_000_000 {
		t.Fatalf("resp = %+v err = %v", resp, err)
	}
	if len(rec.got) != 1 {
		t.Fatalf("records = %d, want 1", len(rec.got))
	}
	u := rec.got[0]
	if u.Task != llm.TaskEmbed || u.Provider != llm.ProviderOpenAI || u.Model != "em" || u.InputTokens != 2_000_000 {
		t.Errorf("usage = %+v", u)
	}
	if u.OutputTokens != 0 || u.CachedTokens != 0 || u.CacheWriteTokens != 0 {
		t.Errorf("non-input counts = %+v", u)
	}
	if want := book.Cost("em", 2_000_000, 0, 0, 0); u.CostUSD != want || want != 1.0 {
		t.Errorf("cost = %v, want %v", u.CostUSD, want)
	}
	if u.Err != nil || u.Estimated {
		t.Errorf("err/estimated = %v/%v", u.Err, u.Estimated)
	}
	if resp.Provider != llm.ProviderOpenAI || resp.Model != "em" {
		t.Errorf("reply provider/model = %v/%v", resp.Provider, resp.Model)
	}
}

func TestWithEmbedUsagePartialFailure(t *testing.T) {
	boom := errors.New("boom")
	inner := &fakeEmbedder{resp: llm.EmbedResponse{Model: "m", InputTokens: 5}, err: boom}
	var rec usageRec
	e := llm.WithEmbedUsage(inner, nil, rec.record)
	resp, err := e.Embed(context.Background(), llm.EmbedRequest{})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
	if resp.Vectors != nil || resp.Model != "m" || resp.InputTokens != 5 {
		t.Fatalf("reply = %+v", resp)
	}
	if len(rec.got) != 1 || rec.got[0].InputTokens != 5 || !errors.Is(rec.got[0].Err, boom) {
		t.Fatalf("records = %+v", rec.got)
	}
}

func TestWithEmbedUsageNoTokensNoRecord(t *testing.T) {
	inner := &fakeEmbedder{err: errors.New("down")}
	var rec usageRec
	e := llm.WithEmbedUsage(inner, nil, rec.record)
	if _, err := e.Embed(context.Background(), llm.EmbedRequest{}); err == nil {
		t.Fatal("want error")
	}
	if len(rec.got) != 0 {
		t.Fatalf("records = %d, want 0", len(rec.got))
	}
}

func TestWithEmbedUsageNilBookUsesDefaults(t *testing.T) {
	inner := &fakeEmbedder{resp: llm.EmbedResponse{Model: "text-embedding-3-small", InputTokens: 1_000_000}}
	var rec usageRec
	e := llm.WithEmbedUsage(inner, nil, rec.record)
	if _, err := e.Embed(context.Background(), llm.EmbedRequest{}); err != nil {
		t.Fatal(err)
	}
	if len(rec.got) != 1 || math.Abs(rec.got[0].CostUSD-0.02) > 1e-9 {
		t.Fatalf("records = %+v", rec.got)
	}
}

func TestWithEmbedUsageEstimatedCopied(t *testing.T) {
	inner := &fakeEmbedder{resp: llm.EmbedResponse{Model: "m", InputTokens: 3, UsageEstimated: true}}
	var rec usageRec
	e := llm.WithEmbedUsage(inner, nil, rec.record)
	if _, err := e.Embed(context.Background(), llm.EmbedRequest{}); err != nil {
		t.Fatal(err)
	}
	if len(rec.got) != 1 || !rec.got[0].Estimated {
		t.Fatalf("records = %+v", rec.got)
	}
}

func TestWithEmbedUsageEnabledAndPanics(t *testing.T) {
	var rec usageRec
	if llm.WithEmbedUsage(&fakeEmbedder{enabled: false}, nil, rec.record).Enabled() {
		t.Error("Enabled should pass through false")
	}
	if !llm.WithEmbedUsage(&fakeEmbedder{enabled: true}, nil, rec.record).Enabled() {
		t.Error("Enabled should pass through true")
	}
	for name, tc := range map[string]struct {
		inner  llm.Embedder
		record func(context.Context, llm.Usage)
		msg    string
	}{
		"nil inner":  {nil, rec.record, "needs an inner Embedder"},
		"nil record": {&fakeEmbedder{}, nil, "needs a record func"},
	} {
		func() {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("%s: want panic", name)
				}
				if s, _ := r.(string); !strings.Contains(s, tc.msg) {
					t.Errorf("%s: panic = %v", name, r)
				}
			}()
			llm.WithEmbedUsage(tc.inner, nil, tc.record)
		}()
	}
}

func TestEmbedDefaultPrices(t *testing.T) {
	for model, want := range map[string]float64{
		"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13,
		"gemini-embedding-2": 0.20, "gemini-embedding-001": 0.15,
	} {
		if got := llm.DefaultPrices[model].Input; got != want {
			t.Errorf("%s input = %v, want %v", model, got, want)
		}
	}
}
