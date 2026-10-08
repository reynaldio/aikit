package guard

import (
	"context"
	"sync"
)

type turnKey struct{}

// Turn is one user action that may span several AI calls (a chat message whose
// tool loop runs several rounds). It is counted once against the rate limit —
// the outcome of that one count (allowed or refused) applies to every call in
// the turn — and it remembers whether any call in it read content that
// tripped the scanner.
type Turn struct {
	mu        sync.Mutex
	counted   bool // the turn's one rate-limit decision has been made
	allowed   bool // … and its outcome
	suspected bool
	seen      map[string]bool // audited sources (dedupe across rounds)
}

// BeginTurn marks the start of a user action on ctx.
func BeginTurn(ctx context.Context) context.Context {
	return context.WithValue(ctx, turnKey{}, &Turn{seen: map[string]bool{}})
}

// TurnFrom returns the turn on ctx, or nil.
func TurnFrom(ctx context.Context) *Turn {
	t, _ := ctx.Value(turnKey{}).(*Turn)
	return t
}

// Suspected reports whether a call in this turn read suspicious content.
func (t *Turn) Suspected() bool {
	if t == nil {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.suspected
}

func (t *Turn) markSuspected() {
	t.mu.Lock()
	t.suspected = true
	t.mu.Unlock()
}

// decide returns the turn's single rate-limit outcome: the first call runs
// check (which counts the action) and the answer is remembered, so every later
// call in the turn gets the same answer — a refused turn stays refused, an
// allowed one isn't counted again.
func (t *Turn) decide(check func() bool) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.counted {
		t.allowed = check()
		t.counted = true
	}
	return t.allowed
}

// firstSight returns true the first time key is seen in this turn.
func (t *Turn) firstSight(key string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.seen[key] {
		return false
	}
	t.seen[key] = true
	return true
}
