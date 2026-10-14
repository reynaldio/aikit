package llmtest

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/reynaldio/aikit/llm"
)

func embedOne(t *testing.T, e *Embedder, req llm.EmbedRequest) []float32 {
	t.Helper()
	resp, err := e.Embed(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	return resp.Vectors[0]
}

func TestEmbedderE1Deterministic(t *testing.T) {
	e := NewEmbedder(0)
	a := embedOne(t, e, llm.EmbedRequest{Texts: []string{"hello"}, Kind: llm.EmbedQuery})
	b := embedOne(t, e, llm.EmbedRequest{Texts: []string{"hello"}, Kind: llm.EmbedQuery})
	if !reflect.DeepEqual(a, b) {
		t.Fatal("same text and kind gave different vectors")
	}
	c := embedOne(t, e, llm.EmbedRequest{Texts: []string{"world"}, Kind: llm.EmbedQuery})
	if reflect.DeepEqual(a, c) {
		t.Fatal("different text gave the same vector")
	}
	d := embedOne(t, e, llm.EmbedRequest{Texts: []string{"hello"}, Kind: llm.EmbedDocument})
	if reflect.DeepEqual(a, d) {
		t.Fatal("different kind gave the same vector")
	}
}

func TestEmbedderE2UnitLengthAndDims(t *testing.T) {
	check := func(e *Embedder, dims int, req llm.EmbedRequest) {
		t.Helper()
		resp, err := e.Embed(context.Background(), req)
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Vectors) != len(req.Texts) {
			t.Fatalf("got %d vectors", len(resp.Vectors))
		}
		for _, v := range resp.Vectors {
			if len(v) != dims {
				t.Fatalf("len = %d, want %d", len(v), dims)
			}
			var s float64
			for _, x := range v {
				s += float64(x) * float64(x)
			}
			if math.Abs(math.Sqrt(s)-1) > 1e-5 {
				t.Fatalf("norm = %v", math.Sqrt(s))
			}
		}
	}
	texts := []string{"a", "bb", "ccc"}
	check(NewEmbedder(0), 8, llm.EmbedRequest{Texts: texts})
	check(NewEmbedder(4), 4, llm.EmbedRequest{Texts: texts})
	check(NewEmbedder(4), 16, llm.EmbedRequest{Texts: texts, Dimensions: 16})
}

func TestEmbedderResponseFields(t *testing.T) {
	e := NewEmbedder(0)
	resp, err := e.Embed(context.Background(), llm.EmbedRequest{Texts: []string{"abcde", "fg"}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Provider != "llmtest" || resp.Model != "fake" {
		t.Fatalf("provider/model = %q/%q", resp.Provider, resp.Model)
	}
	if resp.InputTokens != (7+3)/4 {
		t.Fatalf("tokens = %d", resp.InputTokens)
	}
	empty, err := e.Embed(context.Background(), llm.EmbedRequest{})
	if err != nil || empty.Vectors == nil || len(empty.Vectors) != 0 {
		t.Fatalf("empty = %#v, %v", empty, err)
	}
	if !e.Enabled() {
		t.Fatal("not enabled")
	}
}

func TestEmbedderE3Fail(t *testing.T) {
	e := NewEmbedder(0)
	boom := errors.New("boom")
	e.Fail(boom)
	_, err := e.Embed(context.Background(), llm.EmbedRequest{Texts: []string{"x"}})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
	if len(e.Requests()) != 1 {
		t.Fatalf("recorded %d", len(e.Requests()))
	}
	if _, err := e.Embed(context.Background(), llm.EmbedRequest{Texts: []string{"x"}}); err != nil {
		t.Fatal(err)
	}
	if len(e.Requests()) != 2 {
		t.Fatalf("recorded %d", len(e.Requests()))
	}
}

func TestEmbedderE4CancelledContext(t *testing.T) {
	e := NewEmbedder(0)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := e.Embed(ctx, llm.EmbedRequest{Texts: []string{"x"}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
}

func TestEmbedderE5RequestsCopy(t *testing.T) {
	e := NewEmbedder(0)
	texts := []string{"one", "two"}
	_, _ = e.Embed(context.Background(), llm.EmbedRequest{Texts: texts})
	texts[0] = "changed"
	_, _ = e.Embed(context.Background(), llm.EmbedRequest{Texts: []string{"second"}})
	reqs := e.Requests()
	if len(reqs) != 2 || reqs[0].Texts[0] != "one" || reqs[1].Texts[0] != "second" {
		t.Fatalf("reqs = %#v", reqs)
	}
	reqs[0] = llm.EmbedRequest{}
	if e.Requests()[0].Texts[0] != "one" {
		t.Fatal("Requests is not a copy")
	}
}

func TestEmbedderE6Concurrent(t *testing.T) {
	e := NewEmbedder(0)
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = e.Embed(context.Background(), llm.EmbedRequest{Texts: []string{"x"}})
		}()
	}
	wg.Wait()
	if n := len(e.Requests()); n != 10 {
		t.Fatalf("recorded %d", n)
	}
}
