package guard

import (
	"context"
	"encoding/base64"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/reynaldio/aikit/llm"
)

// Event kinds reported through Options.OnEvent.
const (
	EventRequestTooLarge = "request_too_large"
	EventRateLimited     = "rate_limited"
	EventRefusal         = "refusal"
)

// Event is one guardrail decision, for the app's logs.
type Event struct {
	Kind  string // EventRequestTooLarge | EventRateLimited | EventRefusal
	Class Class
}

// AuditSink records a suspected prompt injection. It must never store the
// matched text — b carries it only so the sink can locate the source.
type AuditSink interface {
	InjectionSuspected(ctx context.Context, b Block)
}

// Options configures Wrap. Only Inner is required.
type Options struct {
	Inner llm.Client
	// Classify decides how a call is guarded, typically from a usage label the
	// app puts on ctx. Nil → every call is ClassOneShot.
	Classify func(ctx context.Context) Class
	// Identity returns the rate-limit key (the user) for a call. !ok, or an empty
	// key, means the call is not rate limited. Nil → nothing is rate limited.
	Identity func(ctx context.Context) (key string, ok bool)
	Limits   Limits
	// Limiter enforces the per-user windows. Nil → a built-in in-process
	// limiter sized from Limits.
	Limiter Limiter
	// TopicPolicy is appended to conversational calls after DataRule: what the
	// assistant covers and declines. "" → none.
	TopicPolicy string
	// RefusalMarker is the token a conversational reply starts with when the
	// model declines. With a TopicPolicy, the instruction to use it is appended
	// after the policy; the wrapper strips it from conversational replies and
	// reports EventRefusal. "" → no marker handling.
	RefusalMarker string
	// OwnSources are fence sources holding the user's OWN words: still fenced,
	// never scanned ("forget my previous instructions" typed by the user is not
	// a third-party injection). Nil → {SourceTranscript, SourceMemory}; an empty
	// non-nil slice scans every source.
	OwnSources []string
	Audit      AuditSink                          // nil → no audit
	OnEvent    func(ctx context.Context, e Event) // nil → nothing reported
	Now        func() time.Time                   // nil → time.Now
}

type client struct {
	o Options
}

// Wrap returns inner with the guardrails applied. A nil or disabled Inner is
// returned as is, so an app without AI keeps its noop client.
func Wrap(o Options) llm.Client {
	if o.Inner == nil || !o.Inner.Enabled() {
		return o.Inner
	}
	if o.Classify == nil {
		o.Classify = func(context.Context) Class { return ClassOneShot }
	}
	if o.Limiter == nil {
		o.Limiter = newWindowLimiter(o.Limits)
	}
	if o.OwnSources == nil {
		o.OwnSources = []string{SourceTranscript, SourceMemory}
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	return &client{o: o}
}

func (c *client) Enabled() bool                  { return c.o.Inner.Enabled() }
func (c *client) CompareTargets() []llm.ModelRef { return c.o.Inner.CompareTargets() }

// Complete runs, in order: the size check (not for background calls), the rate
// limit, the policies, the scan, the inner call, and — for conversational calls
// — the marker strip. A refused call never reaches the inner client, so it
// spends no tokens.
func (c *client) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	class := c.o.Classify(ctx)
	if class != ClassBackground && tooLarge(req, c.o.Limits) {
		c.event(ctx, EventRequestTooLarge, class)
		return llm.Response{}, ErrRequestTooLarge
	}
	if !c.allow(ctx, class) {
		c.event(ctx, EventRateLimited, class)
		return llm.Response{}, ErrRateLimited
	}
	req = withPolicies(req, c.policyFor(class))
	c.scan(ctx, req)
	resp, err := c.o.Inner.Complete(ctx, req)
	if err != nil {
		return resp, err
	}
	if class == ClassConversational {
		if text, declined := StripMarker(resp.Text, c.o.RefusalMarker); declined {
			resp.Text = text
			c.event(ctx, EventRefusal, class)
		}
	}
	return resp, nil
}

func (c *client) event(ctx context.Context, kind string, class Class) {
	if c.o.OnEvent != nil {
		c.o.OnEvent(ctx, Event{Kind: kind, Class: class})
	}
}

// allow applies the per-user windows. One user action counts once: inside a
// Turn only the first call is checked and counted, and its outcome (allowed
// or refused) is reused for every later call in that turn.
func (c *client) allow(ctx context.Context, class Class) bool {
	if class == ClassBackground || c.o.Identity == nil {
		return true
	}
	key, ok := c.o.Identity(ctx)
	if !ok || key == "" {
		return true
	}
	check := func() bool { return c.o.Limiter.CheckAndRecord(key, class, c.o.Now()) }
	if t := TurnFrom(ctx); t != nil {
		return t.decide(check)
	}
	return check()
}

// policyFor is the text appended to a call's system message: DataRule always;
// the topic policy (and the marker instruction, when there is a marker) for
// conversational calls.
func (c *client) policyFor(class Class) string {
	add := DataRule
	if class != ClassConversational || c.o.TopicPolicy == "" {
		return add
	}
	add += "\n\n" + c.o.TopicPolicy
	if c.o.RefusalMarker != "" {
		add += "\n" + markerInstruction(c.o.RefusalMarker)
	}
	return add
}

// tooLarge counts characters of SystemCacheable + every message's content +
// every tool result, and decoded bytes of documents + images.
func tooLarge(req llm.Request, l Limits) bool {
	chars := utf8.RuneCountInString(req.SystemCacheable)
	docs := 0
	for _, m := range req.Messages {
		chars += utf8.RuneCountInString(m.Content)
		for _, tr := range m.ToolResults {
			chars += utf8.RuneCountInString(tr.Content)
		}
		for _, d := range m.Documents {
			docs += base64.StdEncoding.DecodedLen(len(d.Base64))
		}
		for _, im := range m.Images {
			docs += base64.StdEncoding.DecodedLen(len(im.Base64))
		}
	}
	return (l.MaxRequestChars > 0 && chars > l.MaxRequestChars) || (l.MaxDocumentBytes > 0 && docs > l.MaxDocumentBytes)
}

// withPolicies appends add to the first system message, else to
// SystemCacheable, else prepends a new system message — always on a copy, so
// the caller's request is never mutated.
func withPolicies(req llm.Request, add string) llm.Request {
	msgs := append([]llm.Message(nil), req.Messages...)
	for i := range msgs {
		if msgs[i].Role == "system" {
			msgs[i].Content += "\n\n" + add
			req.Messages = msgs
			return req
		}
	}
	if req.SystemCacheable != "" {
		req.SystemCacheable += "\n\n" + add
		return req
	}
	req.Messages = append([]llm.Message{{Role: "system", Content: add}}, msgs...)
	return req
}

// scan audits fenced third-party blocks that look like prompt injection — once
// per source per request, and once per source per Turn across tool-loop
// rounds. Sources in OwnSources are skipped.
func (c *client) scan(ctx context.Context, req llm.Request) {
	turn := TurnFrom(ctx)
	seen := map[string]bool{}
	check := func(s string) {
		if !strings.Contains(s, "<untrusted-data") {
			return
		}
		for _, b := range FindBlocks(s) {
			if slices.Contains(c.o.OwnSources, b.Source) || len(ScanInjection(b.Text)) == 0 {
				continue
			}
			key := b.Source + "|" + b.Ref + "|" + b.Name
			if seen[key] {
				continue
			}
			seen[key] = true
			if turn != nil {
				turn.markSuspected()
				if !turn.firstSight(key) {
					continue
				}
			}
			if c.o.Audit != nil {
				c.o.Audit.InjectionSuspected(ctx, b)
			}
		}
	}
	check(req.SystemCacheable)
	for _, m := range req.Messages {
		check(m.Content)
		for _, tr := range m.ToolResults {
			check(tr.Content)
		}
	}
}
