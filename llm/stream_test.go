package llm

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"strings"
	"testing"
)

// streamFake is a provider that streams: it sends each piece through
// req.OnEvent, then returns err (with resp) or the joined text.
type streamFake struct {
	pieces []string
	err    error
	resp   Response
	calls  int

	gotOnEvent bool // the last request carried an OnEvent
}

func (s *streamFake) streams() bool { return true }

func (s *streamFake) complete(_ context.Context, _ string, _ int, req Request) (Response, error) {
	s.calls++
	s.gotOnEvent = req.OnEvent != nil
	if req.OnEvent != nil {
		for _, p := range s.pieces {
			req.OnEvent(StreamEvent{Kind: StreamText, Text: p})
		}
	}
	resp := s.resp
	if resp.Text == "" {
		resp.Text = strings.Join(s.pieces, "")
	}
	return resp, s.err
}

type streamRec struct{ texts []string }

func (c *streamRec) on(ev StreamEvent) { c.texts = append(c.texts, ev.Text) }
func (c *streamRec) joined() string    { return strings.Join(c.texts, "") }

func newStreamRouter(a, g provider) *router {
	r := newTestRouter(&fakeProvider{}, &fakeProvider{})
	r.providers[ProviderAnthropic] = a
	r.providers[ProviderGoogle] = g
	return r
}

func eq(got []string, want ...string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

var errOverloaded = errors.New("503 overloaded")

func TestStreamErrorBeforeTextFallsBack(t *testing.T) {
	g := &streamFake{err: errOverloaded}
	a := &streamFake{pieces: []string{"Hel", "lo"}}
	r := newStreamRouter(a, g)
	var rec streamRec
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat, OnEvent: rec.on})
	if err != nil {
		t.Fatalf("expected fallback to answer: %v", err)
	}
	if !eq(rec.texts, "Hel", "lo") || rec.joined() != resp.Text {
		t.Fatalf("events %v, resp.Text %q", rec.texts, resp.Text)
	}
	if resp.Model != "haiku" {
		t.Fatalf("model = %q, want haiku", resp.Model)
	}
}

func TestStreamErrorAfterTextDoesNotFallBack(t *testing.T) {
	g := &streamFake{pieces: []string{"Par"}, err: errOverloaded}
	a := &streamFake{}
	r := newStreamRouter(a, g)
	var buf bytes.Buffer
	r.log = slog.New(slog.NewTextHandler(&buf, nil))
	var rec streamRec
	_, err := r.Complete(context.Background(), Request{Task: TaskChat, OnEvent: rec.on})
	if !errors.Is(err, errOverloaded) {
		t.Fatalf("err = %v, want the primary's error", err)
	}
	if !eq(rec.texts, "Par") || a.calls != 0 {
		t.Fatalf("events %v, fallback calls %d", rec.texts, a.calls)
	}
	if !strings.Contains(buf.String(), "llm: stream failed after text was sent; not falling back") {
		t.Fatalf("missing warn log, got %q", buf.String())
	}
}

func TestStreamRefusalMidwayDoesNotFallBack(t *testing.T) {
	g := &streamFake{
		pieces: []string{"I can"},
		err:    &RefusalError{Provider: ProviderGoogle, Model: "flash", Category: "safety"},
		resp:   Response{OutputTokens: 5},
	}
	a := &streamFake{}
	r := newStreamRouter(a, g)
	var rec streamRec
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat, OnEvent: rec.on})
	if !errors.Is(err, ErrRefused) {
		t.Fatalf("err = %v, want ErrRefused", err)
	}
	if resp.Text != "I can" || resp.Model != "flash" || resp.OutputTokens != 5 {
		t.Fatalf("resp = %+v", resp)
	}
	if a.calls != 0 {
		t.Fatalf("fallback called %d times", a.calls)
	}
}

func TestStreamRefusalBeforeTextFallsBack(t *testing.T) {
	g := &streamFake{err: &RefusalError{Provider: ProviderGoogle, Model: "flash", Category: "safety"}}
	a := &streamFake{pieces: []string{"fine"}}
	r := newStreamRouter(a, g)
	var rec streamRec
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat, OnEvent: rec.on})
	if err != nil || resp.Text != "fine" || !eq(rec.texts, "fine") {
		t.Fatalf("err %v, resp %+v, events %v", err, resp, rec.texts)
	}
}

func TestStreamEmptyTextEventDoesNotCountAsSent(t *testing.T) {
	g := &streamFake{pieces: []string{""}, err: errOverloaded}
	a := &streamFake{pieces: []string{"ok"}}
	r := newStreamRouter(a, g)
	var rec streamRec
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat, OnEvent: rec.on})
	if err != nil || resp.Text != "ok" {
		t.Fatalf("err %v, resp %+v", err, resp)
	}
	// The empty piece is passed through as the fake sent it; what matters is that
	// it did not stop the fallback, whose text follows and completes the reply.
	if !eq(rec.texts, "", "ok") || rec.joined() != resp.Text {
		t.Fatalf("events %q, resp.Text %q", rec.texts, resp.Text)
	}
}

func TestStreamJSONSchemaSendsOneEventAfterCheck(t *testing.T) {
	g := &streamFake{pieces: []string{"```json\n{\"a\":1}\n```"}}
	r := newStreamRouter(&streamFake{}, g)
	var rec streamRec
	resp, err := r.Complete(context.Background(), Request{
		Task: TaskChat, JSONSchema: map[string]any{"type": "object"}, OnEvent: rec.on,
	})
	if err != nil {
		t.Fatal(err)
	}
	if g.gotOnEvent {
		t.Fatal("provider must not see OnEvent for a schema request")
	}
	if !eq(rec.texts, `{"a":1}`) || resp.Text != `{"a":1}` {
		t.Fatalf("events %v, resp.Text %q", rec.texts, resp.Text)
	}
}

func TestStreamJSONSchemaViolationSendsNoEvent(t *testing.T) {
	g := &streamFake{pieces: []string{"not json"}}
	r := newStreamRouter(&streamFake{}, g)
	var rec streamRec
	_, err := r.Complete(context.Background(), Request{
		Task: TaskChat, JSONSchema: map[string]any{"type": "object"}, OnEvent: rec.on,
	})
	if !errors.Is(err, ErrSchemaViolation) {
		t.Fatalf("err = %v, want ErrSchemaViolation", err)
	}
	if len(rec.texts) != 0 {
		t.Fatalf("events %v, want none", rec.texts)
	}
}

func TestStreamNonStreamingProviderSendsOneEvent(t *testing.T) {
	g := &fakeProvider{}
	r := newStreamRouter(&fakeProvider{}, g)
	var rec streamRec
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat, OnEvent: rec.on})
	if err != nil {
		t.Fatal(err)
	}
	if !eq(rec.texts, "ok:flash") || resp.Text != "ok:flash" {
		t.Fatalf("events %v, resp %q", rec.texts, resp.Text)
	}
	if g.lastReq.OnEvent != nil {
		t.Fatal("a non-streaming provider must get OnEvent == nil")
	}
}

func TestStreamFallsBackToNonStreamingProvider(t *testing.T) {
	g := &streamFake{err: errOverloaded}
	a := &fakeProvider{}
	r := newStreamRouter(a, g)
	var rec streamRec
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat, OnEvent: rec.on})
	if err != nil {
		t.Fatal(err)
	}
	if !eq(rec.texts, "ok:haiku") || resp.Text != "ok:haiku" {
		t.Fatalf("events %v, resp %q", rec.texts, resp.Text)
	}
	if a.lastReq.OnEvent != nil {
		t.Fatal("a non-streaming fallback must get OnEvent == nil")
	}
}

func TestStreamExplicitModelStreamsAndNeverFallsBack(t *testing.T) {
	g := &streamFake{pieces: []string{"Par"}, err: errOverloaded}
	a := &streamFake{}
	r := newStreamRouter(a, g)
	var buf bytes.Buffer
	r.log = slog.New(slog.NewTextHandler(&buf, nil))
	var rec streamRec
	_, err := r.Complete(context.Background(), Request{
		Model:   &ModelRef{Provider: ProviderGoogle, Model: "flash"},
		OnEvent: rec.on,
	})
	if !errors.Is(err, errOverloaded) {
		t.Fatalf("err = %v", err)
	}
	if !eq(rec.texts, "Par") || a.calls != 0 || buf.Len() != 0 {
		t.Fatalf("events %v, fallback calls %d, log %q", rec.texts, a.calls, buf.String())
	}
}

func TestStreamDoesNotMutateCallersRequest(t *testing.T) {
	g := &streamFake{pieces: []string{"x"}}
	r := newStreamRouter(&streamFake{}, g)
	var rec streamRec
	req := Request{Task: TaskChat, OnEvent: rec.on}
	if _, err := r.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	n := len(rec.texts)
	req.OnEvent(StreamEvent{Kind: StreamText, Text: "again"})
	if len(rec.texts) != n+1 || rec.texts[n] != "again" {
		t.Fatalf("caller's OnEvent was replaced: %v", rec.texts)
	}
}
