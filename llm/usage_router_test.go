package llm

import (
	"context"
	"errors"
	"testing"
)

func TestWithUsageOverCancelledStream(t *testing.T) {
	g := &streamFake{
		pieces: []string{"Hi"},
		err:    context.Canceled,
		resp:   Response{Text: "Hi", InputTokens: 10, OutputTokens: 1, UsageEstimated: true},
	}
	r := newStreamRouter(&streamFake{}, g)
	var got []Usage
	c := WithUsage(r, nil, func(_ context.Context, u Usage) { got = append(got, u) })
	_, err := c.Complete(context.Background(), Request{Task: TaskChat, OnEvent: func(StreamEvent) {}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("records = %d, want 1", len(got))
	}
	u := got[0]
	if !u.Estimated || u.Provider != ProviderGoogle || u.Model == "" || !errors.Is(u.Err, context.Canceled) ||
		u.InputTokens != 10 || u.OutputTokens != 1 {
		t.Fatalf("usage = %+v", u)
	}
}
