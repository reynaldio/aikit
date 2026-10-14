package llm

import (
	"errors"
	"fmt"
)

// ErrNotConfigured is returned when no provider is configured (no API key wired),
// or when a resolved profile points at a provider that isn't configured.
var ErrNotConfigured = errors.New("llm: no provider configured")

// ErrEmbedUnsupported is returned when an embed names a provider with no embedding
// endpoint: Anthropic, DeepSeek and Moonshot. Match with errors.Is.
var ErrEmbedUnsupported = errors.New("llm: provider has no embedding endpoint")

// ErrRefused reports that a provider's safety classifiers declined the request.
//
// This is deliberately an ERROR even though the transport succeeded. A refusal comes
// back as a normal 200 with an empty or partial body, so a client that only checks
// `err != nil` would otherwise hand its caller a silent empty string — the failure
// mode this sentinel exists to prevent. Match with errors.Is; read the category off
// a *RefusalError via errors.As.
var ErrRefused = errors.New("llm: refused by provider safety policy")

// ErrToolResultMismatch reports that a turn's ToolResults do not answer the
// preceding assistant turn's ToolCalls exactly one-to-one — a result naming a
// call that isn't there, or a call left without one.
//
// It is only produced by the Google adapter, and only because it must be: Gemini
// has no tool-call IDs and matches responses to calls BY POSITION, so a missing
// or unrecognized result would silently shift every later response onto the
// wrong call rather than fail. Anthropic and OpenAI resolve by ID and need no
// such check.
//
// Match with errors.Is. The wrapped message names the offending ID, but it is
// prose — do not pattern-match it.
var ErrToolResultMismatch = errors.New("llm: tool results do not match the preceding tool calls")

// ErrSchemaViolation reports that a request with JSONSchema got back a reply that is
// not valid JSON: the output was cut off at the token ceiling (StopTruncated), or a
// provider that cannot enforce a schema answered with prose. Like a refusal it is an
// error on a transport success, and the Response comes back POPULATED alongside it —
// the raw text and the billed tokens — so a caller can log what came back and meter
// the cost. Match with errors.Is; the wrapped message is prose.
//
// It does not trigger failover: the same request on another model is a decision for
// the caller (raise MaxTokens, pin a schema-enforcing model), not a provider outage.
var ErrSchemaViolation = errors.New("llm: reply does not satisfy the requested JSON schema")

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

// ErrBatchUnsupported is returned when a provider cannot batch this kind of
// request: DeepSeek, an OpenAI-compatible endpoint (OpenAIBaseURL set), an unknown
// provider, or embeddings on Anthropic, Moonshot and DeepSeek.
var ErrBatchUnsupported = errors.New("llm: provider cannot batch this kind of request")

// ErrBatchNotDone is returned by Results when the batch is still running.
var ErrBatchNotDone = errors.New("llm: batch has not finished")

// ErrBatchTooLarge is returned by a Submit call when the batch is over the
// provider's item or size limit. Nothing was sent.
var ErrBatchTooLarge = errors.New("llm: batch is over the provider's size limit")

// ErrBadBatchID is returned when a BatchID is not one this library made.
var ErrBadBatchID = errors.New("llm: malformed batch ID")

// ErrBatchItemExpired is the Err of a batch item that the provider dropped
// because the batch ran out of time before the item ran.
var ErrBatchItemExpired = errors.New("llm: batch item expired before it ran")

// ErrBatchItemCanceled is the Err of a batch item that was canceled before it ran.
var ErrBatchItemCanceled = errors.New("llm: batch item canceled before it ran")
