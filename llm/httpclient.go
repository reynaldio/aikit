package llm

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"time"
)

// newProviderHTTPClient is the HTTP client the hand-rolled providers (Google,
// OpenAI-compatible) share. It deliberately has NO whole-request Timeout: that
// covers reading the body too, so it would cut off any reply that takes longer to
// generate than the cap, whatever deadline the caller gave. The caller's context
// bounds the request instead (see withDefaultDeadline).
//
// The transport only guards against a dead connection: dial and TLS timeouts, and
// TCP keepalives that notice a peer gone silent mid-request. It sets no
// ResponseHeaderTimeout on purpose: a non-streaming completion sends its headers
// only once generation has FINISHED, so a header timeout would reimpose the same
// cap under another name.
func newProviderHTTPClient() *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.DialContext = (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	t.TLSHandshakeTimeout = 10 * time.Second
	return &http.Client{Transport: t}
}

// minRequestTimeout is the floor of the default deadline: what every request got
// before the default started scaling with MaxTokens.
const minRequestTimeout = 120 * time.Second

// defaultRequestTimeout is the deadline for a caller that set none: override when
// non-zero, else max(120s, 1h × maxTokens / 128000), the same estimate the
// Anthropic SDK uses, so a long reply gets room and a short one behaves as before.
func defaultRequestTimeout(override time.Duration, maxTokens int) time.Duration {
	if override > 0 {
		return override
	}
	scaled := time.Duration(float64(time.Hour) * float64(maxTokens) / 128000)
	return max(minRequestTimeout, scaled)
}

// withDefaultDeadline returns ctx unchanged when the caller set a deadline — the
// caller decides how long a request may take — and otherwise bounds it by
// defaultRequestTimeout, so a deadline-less caller can't hang forever. Either way
// the error on expiry is context.DeadlineExceeded. Always call the CancelFunc.
func withDefaultDeadline(ctx context.Context, override time.Duration, maxTokens int) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, defaultRequestTimeout(override, maxTokens))
}

// readError reports a failure reading a response body. When the context ended
// meanwhile — the deadline passed mid-body — the context's error is what is
// wrapped, so errors.Is(err, context.DeadlineExceeded) holds instead of the
// failure reading as a garbled reply.
func readError(ctx context.Context, provider string, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return fmt.Errorf("%s: read response: %w", provider, ctxErr)
	}
	return fmt.Errorf("%s: read response: %w", provider, err)
}
