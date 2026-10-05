package decide

import (
	"errors"
	"fmt"
)

// ErrNotConfigured is returned when no backend is configured (no API key wired).
var ErrNotConfigured = errors.New("decide: no backend configured")

// APIError is a non-2xx reply from the backend. StatusCode is the HTTP status;
// Message is the backend's prose, for logging only — do not pattern-match it.
//
// The client does not retry. TypeSafe asks for exponential backoff on 429 (rate
// limited) and 529 (overloaded) rather than an immediate retry; Retryable reports
// exactly those plus other 5xx, so a caller's backoff loop can branch on it.
type APIError struct {
	StatusCode int
	Message    string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("decide: %s (status %d)", e.Message, e.StatusCode)
}

// Retryable reports whether the same request may succeed later. A 401 (bad key) or
// 422 (invalid request) will not.
func (e *APIError) Retryable() bool {
	return e.StatusCode == 429 || e.StatusCode >= 500
}
