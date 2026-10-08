package guard

import (
	"errors"
	"sync"
	"time"
	"unicode/utf8"
)

// ErrRequestTooLarge refuses a request over the size caps before the model is
// called. ErrRateLimited refuses a user over their usage limits. Both are plain
// sentinels: an app maps them onto its own (localized) errors with errors.Is.
var (
	ErrRequestTooLarge = errors.New("guard: request too large")
	ErrRateLimited     = errors.New("guard: rate limited")
)

// Limits are the per-user usage limits and per-request size caps. A value ≤ 0
// disables that limit.
type Limits struct {
	// Per-user windows, applied by the built-in Limiter (a custom
	// Options.Limiter brings its own and ignores these three).
	ConversationalPerHour   int
	ConversationalPerMinute int
	OneShotPerHour          int
	// Per-request size caps, applied by the wrapper to every call except
	// ClassBackground: characters of text, and decoded bytes of documents +
	// images.
	MaxRequestChars  int
	MaxDocumentBytes int
	// Caps on one typed message, applied by CheckMessage at an app's entry
	// points (the wrapper never sees "a message", only whole requests).
	MaxMessageRunes int
	MaxAttachments  int
}

// DefaultLimits are dossio's production defaults.
func DefaultLimits() Limits {
	return Limits{
		ConversationalPerHour:   60,
		ConversationalPerMinute: 10,
		OneShotPerHour:          120,
		MaxRequestChars:         600_000,
		MaxDocumentBytes:        50 << 20,
		MaxMessageRunes:         20_000,
		MaxAttachments:          10,
	}
}

// CheckMessage refuses an over-long typed message or too many attachments with
// ErrRequestTooLarge. Call it at every entry point that takes free-typed text,
// before anything else runs.
func CheckMessage(text string, attachments int, l Limits) error {
	if l.MaxMessageRunes > 0 && utf8.RuneCountInString(text) > l.MaxMessageRunes {
		return ErrRequestTooLarge
	}
	if l.MaxAttachments > 0 && attachments > l.MaxAttachments {
		return ErrRequestTooLarge
	}
	return nil
}

// Limiter enforces the per-user usage limits. Implement it to share limits
// across replicas (Redis, Postgres); the built-in one is in-process.
type Limiter interface {
	// CheckAndRecord atomically refuses when any window for the class is full,
	// otherwise records one use in each. Only ClassOneShot and
	// ClassConversational are ever passed. It must be atomic: concurrent calls
	// for one key must not both pass a nearly-full window.
	CheckAndRecord(key string, class Class, now time.Time) bool
}

// windowLimiter is the built-in Limiter: sliding windows per key, one set for
// conversational calls (per minute and per hour) and one for one-shot calls
// (per hour). Deliberately in-process — each replica enforces its own budget.
type windowLimiter struct {
	// mu makes check-then-record atomic across the windows, so concurrent
	// calls of one user can't both pass a nearly-full window.
	mu                         sync.Mutex
	convMin, convHour, oneHour *window
}

func newWindowLimiter(l Limits) *windowLimiter {
	return &windowLimiter{
		convMin:  newWindow(l.ConversationalPerMinute, time.Minute),
		convHour: newWindow(l.ConversationalPerHour, time.Hour),
		oneHour:  newWindow(l.OneShotPerHour, time.Hour),
	}
}

func (w *windowLimiter) CheckAndRecord(key string, class Class, now time.Time) bool {
	ws := []*window{w.oneHour}
	if class == ClassConversational {
		ws = []*window{w.convMin, w.convHour}
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, win := range ws {
		if win != nil && win.exceeded(key, now) {
			return false
		}
	}
	for _, win := range ws {
		if win != nil {
			win.record(key, now)
		}
	}
	return true
}

// window is a sliding-window counter keyed by string: at most max hits per key
// within any span of length per. Not safe for concurrent use on its own; the
// windowLimiter's mutex guards it. Ported from dossio's pkg/ratelimit.Window.
type window struct {
	max  int
	per  time.Duration
	hits map[string][]time.Time
	// lastSweep bounds memory: keys with no recent hits are dropped every
	// 10 × per, opportunistically on record, so no background goroutine is needed.
	lastSweep time.Time
}

// newWindow returns nil for a disabled limit (max ≤ 0).
func newWindow(max int, per time.Duration) *window {
	if max <= 0 {
		return nil
	}
	return &window{max: max, per: per, hits: map[string][]time.Time{}}
}

func (w *window) exceeded(key string, now time.Time) bool {
	hits := pruneBefore(w.hits[key], now.Add(-w.per))
	w.hits[key] = hits
	return len(hits) >= w.max
}

func (w *window) record(key string, now time.Time) {
	w.sweep(now)
	w.hits[key] = append(pruneBefore(w.hits[key], now.Add(-w.per)), now)
}

func (w *window) sweep(now time.Time) {
	if w.lastSweep.IsZero() {
		w.lastSweep = now
		return
	}
	if now.Sub(w.lastSweep) < 10*w.per {
		return
	}
	w.lastSweep = now
	for k, hits := range w.hits {
		if fresh := pruneBefore(hits, now.Add(-w.per)); len(fresh) == 0 {
			delete(w.hits, k)
		} else {
			w.hits[k] = fresh
		}
	}
}

func pruneBefore(hits []time.Time, cutoff time.Time) []time.Time {
	kept := hits[:0]
	for _, t := range hits {
		if t.After(cutoff) {
			kept = append(kept, t)
		}
	}
	return kept
}
