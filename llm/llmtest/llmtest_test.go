package llmtest

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/reynaldio/aikit/llm"
)

func collect(req *llm.Request) *[]llm.StreamEvent {
	var evs []llm.StreamEvent
	req.OnEvent = func(e llm.StreamEvent) { evs = append(evs, e) }
	return &evs
}

func TestRepliesInOrderThenErrNoReplies(t *testing.T) {
	c := New(Reply{Response: llm.Response{Text: "a"}}, Reply{Response: llm.Response{Text: "b"}}, Reply{Err: errors.New("boom")})
	for _, want := range []string{"a", "b"} {
		got, err := c.Complete(context.Background(), llm.Request{})
		if err != nil || got.Text != want {
			t.Fatalf("got %q, %v; want %q", got.Text, err, want)
		}
	}
	if _, err := c.Complete(context.Background(), llm.Request{}); err == nil || err.Error() != "boom" {
		t.Fatalf("third: %v", err)
	}
	if _, err := c.Complete(context.Background(), llm.Request{}); !errors.Is(err, ErrNoReplies) {
		t.Fatalf("fourth: %v", err)
	}
	if !c.Enabled() || c.CompareTargets() != nil {
		t.Fatal("Enabled/CompareTargets wrong")
	}
}

func TestRequestsCopy(t *testing.T) {
	c := New(Reply{}, Reply{})
	c.Complete(context.Background(), llm.Request{SystemCacheable: "one"})
	c.Complete(context.Background(), llm.Request{SystemCacheable: "two"})
	got := c.Requests()
	if len(got) != 2 || got[0].SystemCacheable != "one" || got[1].SystemCacheable != "two" {
		t.Fatalf("order: %+v", got)
	}
	got[0].SystemCacheable = "changed"
	_ = append(got, llm.Request{})
	again := c.Requests()
	if len(again) != 2 || again[0].SystemCacheable != "one" {
		t.Fatalf("not a copy: %+v", again)
	}
}

func TestCancelledContextKeepsReply(t *testing.T) {
	c := New(Reply{Response: llm.Response{Text: "kept"}})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.Complete(ctx, llm.Request{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	got, err := c.Complete(context.Background(), llm.Request{})
	if err != nil || got.Text != "kept" {
		t.Fatalf("got %q, %v", got.Text, err)
	}
	if len(c.Requests()) != 2 {
		t.Fatal("cancelled call should still be recorded")
	}
}

func TestTextPiecesJoin(t *testing.T) {
	for _, text := range []string{"", "a", "ab", "abc", "héllo wörld", "日本語のテキストです"} {
		c := New(Reply{Response: llm.Response{Text: text}})
		req := llm.Request{}
		evs := collect(&req)
		if _, err := c.Complete(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		var b strings.Builder
		for _, e := range *evs {
			if e.Kind != llm.StreamText {
				t.Fatalf("%q: unexpected kind %q", text, e.Kind)
			}
			b.WriteString(e.Text)
		}
		if b.String() != text {
			t.Fatalf("joined %q, want %q", b.String(), text)
		}
		if n := len(*evs); n > 3 || (text == "" && n != 0) {
			t.Fatalf("%q: %d events", text, n)
		}
	}
	if got := splitText("héllo wörld"); len(got) != 3 {
		t.Fatalf("pieces = %q", got)
	}
}

func TestToolEvents(t *testing.T) {
	calls := []llm.ToolCall{{ID: "1", Name: "x"}, {ID: "2", Name: "y"}}
	tool := func(k llm.StreamKind, id, name string) llm.StreamEvent {
		return llm.StreamEvent{Kind: k, ToolCallID: id, ToolName: name}
	}
	cases := map[llm.ToolEvents][]llm.StreamEvent{
		"":                       nil,
		"bogus":                  nil,
		llm.ToolEventsStart:      {tool(llm.StreamToolStart, "1", "x"), tool(llm.StreamToolStart, "2", "y")},
		llm.ToolEventsStartReady: {tool(llm.StreamToolStart, "1", "x"), tool(llm.StreamToolReady, "1", "x"), tool(llm.StreamToolStart, "2", "y"), tool(llm.StreamToolReady, "2", "y")},
	}
	for mode, want := range cases {
		c := New(Reply{Response: llm.Response{ToolCalls: calls}})
		req := llm.Request{ToolEvents: mode}
		evs := collect(&req)
		c.Complete(context.Background(), req)
		if !reflect.DeepEqual(*evs, want) && !(len(*evs) == 0 && len(want) == 0) {
			t.Fatalf("mode %q: got %+v, want %+v", mode, *evs, want)
		}
	}
}

func TestTextThenToolEvents(t *testing.T) {
	c := New(Reply{Response: llm.Response{Text: "hello", ToolCalls: []llm.ToolCall{{ID: "1", Name: "x"}}}})
	req := llm.Request{ToolEvents: llm.ToolEventsStart}
	evs := collect(&req)
	c.Complete(context.Background(), req)
	last := (*evs)[len(*evs)-1]
	if last.Kind != llm.StreamToolStart || (*evs)[0].Kind != llm.StreamText {
		t.Fatalf("order wrong: %+v", *evs)
	}
}

func TestJSONSchemaAndErr(t *testing.T) {
	c := New(Reply{Response: llm.Response{Text: "{\"a\":1}", ToolCalls: []llm.ToolCall{{ID: "1", Name: "x"}}}})
	req := llm.Request{JSONSchema: map[string]any{"type": "object"}, ToolEvents: llm.ToolEventsStartReady}
	evs := collect(&req)
	c.Complete(context.Background(), req)
	if len(*evs) != 1 || (*evs)[0].Kind != llm.StreamText || (*evs)[0].Text != "{\"a\":1}" {
		t.Fatalf("schema events: %+v", *evs)
	}

	c = New(Reply{Response: llm.Response{Text: "partial"}, Err: errors.New("x")})
	req = llm.Request{}
	evs = collect(&req)
	c.Complete(context.Background(), req)
	if len(*evs) != 0 {
		t.Fatalf("err events: %+v", *evs)
	}
}

func TestExplicitEvents(t *testing.T) {
	want := []llm.StreamEvent{{Kind: llm.StreamText, Text: "q"}, {Kind: llm.StreamToolReady, ToolCallID: "z"}}
	for _, err := range []error{nil, errors.New("bad")} {
		c := New(Reply{Response: llm.Response{Text: "ignored"}, Err: err, Events: want})
		req := llm.Request{JSONSchema: map[string]any{"a": 1}, ToolEvents: llm.ToolEventsStartReady}
		evs := collect(&req)
		c.Complete(context.Background(), req)
		if !reflect.DeepEqual(*evs, want) {
			t.Fatalf("err=%v got %+v", err, *evs)
		}
	}
}

func TestConcurrent(t *testing.T) {
	var rs []Reply
	for i := 0; i < 10; i++ {
		rs = append(rs, Reply{Response: llm.Response{InputTokens: i}})
	}
	c := New(rs...)
	var mu sync.Mutex
	seen := map[int]int{}
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := c.Complete(context.Background(), llm.Request{})
			if err != nil {
				t.Error(err)
				return
			}
			mu.Lock()
			seen[r.InputTokens]++
			mu.Unlock()
		}()
	}
	wg.Wait()
	if len(seen) != 10 {
		t.Fatalf("seen = %v", seen)
	}
	for k, v := range seen {
		if v != 1 {
			t.Fatalf("reply %d returned %d times", k, v)
		}
	}
	if len(c.Requests()) != 10 {
		t.Fatal("requests not recorded")
	}
}

func TestAdd(t *testing.T) {
	c := New(Reply{Response: llm.Response{Text: "a"}})
	c.Add(Reply{Response: llm.Response{Text: "b"}}, Reply{Response: llm.Response{Text: "c"}})
	for _, want := range []string{"a", "b", "c"} {
		got, _ := c.Complete(context.Background(), llm.Request{})
		if got.Text != want {
			t.Fatalf("got %q want %q", got.Text, want)
		}
	}
}
