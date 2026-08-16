package llm

import "errors"

// ErrNotConfigured is returned when no provider is configured (no API key wired),
// or when a resolved profile points at a provider that isn't configured.
var ErrNotConfigured = errors.New("llm: no provider configured")
