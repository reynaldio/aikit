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
	script []StreamEvent // when set, sent in place of pieces
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
		for _, ev := range s.script {
			req.OnEvent(ev)
		}
	}
	resp := s.resp
	if resp.Text == "" {
		resp.Text = strings.Join(s.pieces, "")
		for _, ev := range s.script {
			if ev.Kind == StreamText {
				resp.Text += ev.Text
			}
		}
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

// evRec records whole events, so tests can see kinds and tool fields.
type evRec struct{ evs []StreamEvent }

func (c *evRec) on(ev StreamEvent) { c.evs = append(c.evs, ev) }

// desc renders events as "text:Hi", "start:c1/search", "ready:c1/search".
func (c *evRec) desc() []string {
	var out []string
	for _, ev := range c.evs {
		switch ev.Kind {
		case StreamText:
			out = append(out, "text:"+ev.Text)
		case StreamToolStart:
			out = append(out, "start:"+ev.ToolCallID+"/"+ev.ToolName)
		case StreamToolReady:
			out = append(out, "ready:"+ev.ToolCallID+"/"+ev.ToolName)
		}
	}
	return out
}

var toolScript = []StreamEvent{
	{Kind: StreamToolStart, ToolCallID: "c1", ToolName: "search"},
	{Kind: StreamText, Text: "Hi"},
	{Kind: StreamToolReady, ToolCallID: "c1", ToolName: "search"},
}

func TestToolEventsFiltering(t *testing.T) {
	cases := []struct {
		name string
		mode ToolEvents
		want []string
	}{
		{"unset", "", []string{"text:Hi"}},
		{"start", ToolEventsStart, []string{"start:c1/search", "text:Hi"}},
		{"start_ready", ToolEventsStartReady, []string{"start:c1/search", "text:Hi", "ready:c1/search"}},
		{"unknown", ToolEvents("yes"), []string{"text:Hi"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := newStreamRouter(&streamFake{}, &streamFake{script: toolScript})
			var rec evRec
			resp, err := r.Complete(context.Background(), Request{Task: TaskChat, OnEvent: rec.on, ToolEvents: tc.mode})
			if err != nil || resp.Text != "Hi" {
				t.Fatalf("err %v, resp %+v", err, resp)
			}
			if got := rec.desc(); !eq(got, tc.want...) {
				t.Fatalf("events %v, want %v", got, tc.want)
			}
		})
	}
}

func TestToolEventThenFailureStillFallsBack(t *testing.T) {
	g := &streamFake{
		script: []StreamEvent{{Kind: StreamToolStart, ToolCallID: "c1", ToolName: "search"}},
		err:    errOverloaded,
	}
	a := &streamFake{pieces: []string{"ok"}, resp: Response{ToolCalls: []ToolCall{{ID: "f1", Name: "lookup"}}}}
	r := newStreamRouter(a, g)
	var rec evRec
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat, OnEvent: rec.on, ToolEvents: ToolEventsStart})
	if err != nil {
		t.Fatalf("expected fallback to answer: %v", err)
	}
	if got := rec.desc(); !eq(got, "start:c1/search", "text:ok") {
		t.Fatalf("events %v", got)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "f1" {
		t.Fatalf("ToolCalls = %+v, want the fallback's", resp.ToolCalls)
	}
}

func TestToolEventsFromProviderThatCannotStream(t *testing.T) {
	reply := &Response{Text: "Let me check", ToolCalls: []ToolCall{{ID: "1", Name: "a"}, {ID: "2", Name: "b"}}}
	cases := []struct {
		name string
		mode ToolEvents
		want []string
	}{
		{"start_ready", ToolEventsStartReady, []string{
			"text:Let me check", "start:1/a", "ready:1/a", "start:2/b", "ready:2/b"}},
		{"start", ToolEventsStart, []string{"text:Let me check", "start:1/a", "start:2/b"}},
		{"unset", "", []string{"text:Let me check"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := newTestRouter(&fakeProvider{reply: reply}, &fakeProvider{reply: reply})
			var rec evRec
			_, err := r.Complete(context.Background(), Request{Task: TaskChat, OnEvent: rec.on, ToolEvents: tc.mode})
			if err != nil {
				t.Fatal(err)
			}
			if got := rec.desc(); !eq(got, tc.want...) {
				t.Fatalf("events %v, want %v", got, tc.want)
			}
		})
	}
}

func TestToolEventsNotSentWithJSONSchema(t *testing.T) {
	script := []StreamEvent{
		{Kind: StreamToolStart, ToolCallID: "c1", ToolName: "search"},
		{Kind: StreamText, Text: "{}"},
		{Kind: StreamToolReady, ToolCallID: "c1", ToolName: "search"},
	}
	g := &streamFake{script: script}
	r := newStreamRouter(&streamFake{}, g)
	var rec evRec
	_, err := r.Complete(context.Background(), Request{
		Task: TaskChat, JSONSchema: map[string]any{"type": "object"}, OnEvent: rec.on, ToolEvents: ToolEventsStartReady,
	})
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if g.gotOnEvent {
		t.Fatal("provider received OnEvent despite JSONSchema")
	}
	if got := rec.desc(); !eq(got, "text:{}") {
		t.Fatalf("events %v, want [text:{}]", got)
	}
}

func TestToolEventsWithoutOnEvent(t *testing.T) {
	g := &streamFake{script: toolScript}
	r := newStreamRouter(&streamFake{}, g)
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat, ToolEvents: ToolEventsStartReady})
	if err != nil || resp.Text != "Hi" || g.gotOnEvent {
		t.Fatalf("err %v, resp %+v, gotOnEvent %v", err, resp, g.gotOnEvent)
	}
}
