// Package llmtest is a scripted fake llm.Client for tests.
//
// Give it replies; each Complete call uses the next one. It records every
// request, and it streams events to Request.OnEvent like a real provider.
//
//	c := llmtest.New(
//		llmtest.Reply{Response: llm.Response{ToolCalls: []llm.ToolCall{{ID: "1", Name: "lookup"}}}},
//		llmtest.Reply{Response: llm.Response{Text: "done"}},
//	)
//	// ... run the code under test with c ...
//	reqs := c.Requests() // one entry per Complete call, in order
package llmtest

import (
	"context"
	"errors"
	"sync"

	"github.com/reynaldio/aikit/llm"
)

// ErrNoReplies is returned by Complete when every reply has been used.
var ErrNoReplies = errors.New("llmtest: no replies left")

// Reply is one scripted answer.
type Reply struct {
	Response llm.Response
	Err      error
	// Events, when set, are sent to OnEvent exactly as given, instead of
	// events made from Response. They are sent even when Err is set.
	Events []llm.StreamEvent
}

// Client is a fake llm.Client. It is safe for concurrent use.
type Client struct {
	mu       sync.Mutex
	replies  []Reply
	requests []llm.Request
}

var _ llm.Client = (*Client)(nil)

// New returns a Client that answers with replies, in order.
func New(replies ...Reply) *Client {
	return &Client{replies: append([]Reply(nil), replies...)}
}

// Add appends replies after the ones already queued.
func (c *Client) Add(replies ...Reply) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.replies = append(c.replies, replies...)
}

// Requests returns a copy of the requests received so far, in order.
func (c *Client) Requests() []llm.Request {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]llm.Request(nil), c.requests...)
}

// Enabled always reports true.
func (c *Client) Enabled() bool { return true }

// CompareTargets always returns nil.
func (c *Client) CompareTargets() []llm.ModelRef { return nil }

// Complete records req, then answers with the next reply. A cancelled ctx
// returns ctx.Err() and uses no reply. With no replies left it returns
// ErrNoReplies.
func (c *Client) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	c.mu.Lock()
	c.requests = append(c.requests, req)
	if err := ctx.Err(); err != nil {
		c.mu.Unlock()
		return llm.Response{}, err
	}
	if len(c.replies) == 0 {
		c.mu.Unlock()
		return llm.Response{}, ErrNoReplies
	}
	r := c.replies[0]
	c.replies = c.replies[1:]
	c.mu.Unlock()

	if req.OnEvent != nil {
		for _, ev := range eventsFor(r, req) {
			req.OnEvent(ev)
		}
	}
	return r.Response, r.Err
}

// eventsFor builds the events one reply sends for req.
func eventsFor(r Reply, req llm.Request) []llm.StreamEvent {
	if r.Events != nil {
		return r.Events
	}
	if r.Err != nil {
		return nil
	}
	text := r.Response.Text
	var evs []llm.StreamEvent
	if req.JSONSchema != nil {
		if text != "" {
			evs = append(evs, llm.StreamEvent{Kind: llm.StreamText, Text: text})
		}
		return evs
	}
	for _, p := range splitText(text) {
		evs = append(evs, llm.StreamEvent{Kind: llm.StreamText, Text: p})
	}
	start := req.ToolEvents == llm.ToolEventsStart || req.ToolEvents == llm.ToolEventsStartReady
	ready := req.ToolEvents == llm.ToolEventsStartReady
	for _, call := range r.Response.ToolCalls {
		if start {
			evs = append(evs, llm.StreamEvent{Kind: llm.StreamToolStart, ToolCallID: call.ID, ToolName: call.Name})
		}
		if ready {
			evs = append(evs, llm.StreamEvent{Kind: llm.StreamToolReady, ToolCallID: call.ID, ToolName: call.Name})
		}
	}
	return evs
}

// splitText cuts s at rune boundaries. Empty text gives no pieces, 1-2 runes
// give one piece per rune, and longer text gives 3 pieces of about equal size.
func splitText(s string) []string {
	rs := []rune(s)
	n := len(rs)
	if n == 0 {
		return nil
	}
	if n <= 2 {
		out := make([]string, n)
		for i, r := range rs {
			out[i] = string(r)
		}
		return out
	}
	a, b := n/3, 2*n/3
	if a == 0 {
		a = 1
	}
	return []string{string(rs[:a]), string(rs[a:b]), string(rs[b:])}
}
