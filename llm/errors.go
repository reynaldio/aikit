package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
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
// not valid JSON: the output was incomplete (StopTruncated: cut off at the token
// ceiling, or an Anthropic search turn still paused), or a
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

// ProviderError is the typed form of a failure the provider reported or a
// transport failure on the way to it. Every provider wraps one in the errors it
// returns from Complete (plain, streaming, tool loop), Embed and Batch calls; find
// it with errors.As through any wrapping.
//
// Status is the HTTP status of the failure. It is 0 when there is none: the
// request failed in transport (no response, or the connection dropped mid-body),
// or the provider reported the error inside a 2xx reply (mid-stream) without a
// status of its own.
//
// Retryable says the failure is likely the provider's, not the request's: Status
// 0, 408, 409, 429 and every 5xx (Anthropic's 529 overloaded among them). Other
// 4xx — a bad request, a page limit, a bad key — are false.
//
// A context that ended (the caller canceled, or a deadline passed) is never a
// ProviderError: match it with errors.Is(err, context.Canceled) or
// context.DeadlineExceeded. Errors the reply caused on a transport success —
// ErrRefused, ErrSchemaViolation — are not ProviderErrors either.
//
// Error returns the underlying message unchanged.
type ProviderError struct {
	Provider  Provider
	Status    int  // HTTP status; 0 for a transport error with no response
	Retryable bool // true: transport, 408, 409, 429, 5xx; false: other 4xx
	Err       error
}

func (e *ProviderError) Error() string { return e.Err.Error() }

// Unwrap returns the underlying error, so errors.Is still matches what it wraps
// (io.ErrUnexpectedEOF, a *net.OpError, the SDK's *anthropic.Error).
func (e *ProviderError) Unwrap() error { return e.Err }

// retryableStatusCode is the Retryable rule. It is wider than retryableStatus,
// which picks what the transport retries on its own.
func retryableStatusCode(status int) bool {
	return status == 0 || status == 408 || status == 409 || status == 429 || status >= 500
}

// statusError wraps err, a failure with HTTP status status, in a *ProviderError.
func statusError(p Provider, status int, err error) error {
	return &ProviderError{Provider: p, Status: status, Retryable: retryableStatusCode(status), Err: err}
}

// transportError wraps err, a request that got no complete response, as Status 0.
// When ctx has ended, err is returned as is: the caller's cancel or deadline is
// not a provider failure.
func transportError(ctx context.Context, p Provider, err error) error {
	if err == nil || ctx.Err() != nil {
		return err
	}
	return statusError(p, 0, err)
}

// isTransport reports whether err is a connection-level failure: a network error,
// or a body cut off part way.
func isTransport(err error) bool {
	var ue *url.Error
	var ne net.Error
	return errors.As(err, &ue) || errors.As(err, &ne) ||
		errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF)
}

// anthropicTypeStatus maps an Anthropic error type to its HTTP status. It is used
// where the error came without a usable status: mid-stream and batch items. An
// unknown type gives 0.
func anthropicTypeStatus(t string) int {
	switch t {
	case "invalid_request_error":
		return 400
	case "authentication_error":
		return 401
	case "billing_error":
		return 402
	case "permission_error":
		return 403
	case "not_found_error":
		return 404
	case "request_too_large":
		return 413
	case "rate_limit_error":
		return 429
	case "api_error":
		return 500
	case "timeout_error":
		return 504
	case "overloaded_error":
		return 529
	}
	return 0
}

// grpcStatus maps a google.rpc.Status code (Gemini batch items) to an HTTP status.
// A code that is already an HTTP status passes through; an unknown one gives 0.
func grpcStatus(code int) int {
	if code >= 100 {
		return code
	}
	switch code {
	case 3, 9, 11: // INVALID_ARGUMENT, FAILED_PRECONDITION, OUT_OF_RANGE
		return 400
	case 16: // UNAUTHENTICATED
		return 401
	case 7: // PERMISSION_DENIED
		return 403
	case 5: // NOT_FOUND
		return 404
	case 10: // ABORTED
		return 409
	case 8: // RESOURCE_EXHAUSTED
		return 429
	case 2, 13, 15: // UNKNOWN, INTERNAL, DATA_LOSS
		return 500
	case 12: // UNIMPLEMENTED
		return 501
	case 14: // UNAVAILABLE
		return 503
	case 4: // DEADLINE_EXCEEDED
		return 504
	}
	return 0
}
