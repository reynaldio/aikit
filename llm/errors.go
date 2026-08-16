package llm

import (
	"errors"
	"fmt"
)

// ErrNotConfigured is returned when no provider is configured (no API key wired),
// or when a resolved profile points at a provider that isn't configured.
var ErrNotConfigured = errors.New("llm: no provider configured")

// ErrRefused reports that a provider's safety classifiers declined the request.
//
// This is deliberately an ERROR even though the transport succeeded. A refusal comes
// back as a normal 200 with an empty or partial body, so a client that only checks
// `err != nil` would otherwise hand its caller a silent empty string — the failure
// mode this sentinel exists to prevent. Match with errors.Is; read the category off
// a *RefusalError via errors.As.
var ErrRefused = errors.New("llm: refused by provider safety policy")

// RefusalError carries the provider's refusal category so callers can tell a policy
// decline apart from an outage, and tell the categories apart from each other.
//
// Category is the provider's own label, passed through unmapped (Anthropic today
// emits "cyber", "bio", "frontier_llm", "reasoning_extraction"; the set is open and
// differs per provider). Explanation is human-readable, may be empty, and is not
// stable enough to branch on — log it, don't parse it.
//
// A refusal is routed through the router's normal failover path: classifiers differ
// per model, so a profile's configured fallback is often exactly the model that will
// answer. Domains that legitimately trip a classifier — security tooling against the
// cyber category, life sciences against bio — should configure that fallback rather
// than treat the refusal as a defect.
type RefusalError struct {
	Provider    Provider
	Model       string
	Category    string
	Explanation string
}

func (e *RefusalError) Error() string {
	s := fmt.Sprintf("llm: %s/%s refused by provider safety policy", e.Provider, e.Model)
	if e.Category != "" {
		s += " (category " + e.Category + ")"
	}
	if e.Explanation != "" {
		s += ": " + e.Explanation
	}
	return s
}

// Unwrap makes errors.Is(err, ErrRefused) match.
func (e *RefusalError) Unwrap() error { return ErrRefused }
