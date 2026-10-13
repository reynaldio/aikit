package llm

import (
	"context"
	"io"
	"log/slog"
	"math"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// retryTransport retries a request on the same endpoint when it failed in a way
// that is likely to pass on its own: 429/5xx responses and connection errors. It
// returns the final outcome exactly as one attempt would, so callers see the same
// *http.Response or error as before. It never reads a 2xx body.
type retryTransport struct {
	base       http.RoundTripper
	log        *slog.Logger                                     // nil = no log
	maxRetries int                                              // 2
	maxWait    time.Duration                                    // 10 * time.Second
	sleep      func(ctx context.Context, d time.Duration) error // default sleepCtx; tests swap in a fake
	now        func() time.Time                                 // default time.Now
}

func newRetryTransport(base http.RoundTripper, lg *slog.Logger) *retryTransport {
	return &retryTransport{
		base:       base,
		log:        lg,
		maxRetries: 2,
		maxWait:    10 * time.Second,
		sleep:      sleepCtx,
		now:        time.Now,
	}
}

// RoundTrip follows a fixed rule order; the numbers match the design. Every
// return in steps 2-7 happens before anything is drained, closed or fetched, so a
// response returned "as is" always has its body unread and open.
func (t *retryTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	ctx := req.Context()
	r := req
	for retry := 0; ; retry++ {
		// 1. Attempt.
		res, err := t.base.RoundTrip(r)

		// 2. Success or not retryable: hand back untouched (a 2xx body is never read).
		if err == nil && !retryableStatus(res.StatusCode) {
			return res, nil
		}

		// 3. Context ended: report it, even on the last attempt.
		if ctx.Err() != nil {
			drainClose(res)
			return nil, ctx.Err()
		}

		// 4. Out of retries.
		if retry == t.maxRetries {
			return res, err
		}

		// 5. A body without GetBody cannot be replayed.
		if req.Body != nil && req.Body != http.NoBody && req.GetBody == nil {
			return res, err
		}

		// 6. Pick the wait: the server's hint if usable, else jittered backoff.
		var d time.Duration
		useBackoff := true
		if res != nil {
			if sd, ok := serverWait(res.Header, t.now()); ok {
				if sd > t.maxWait {
					return res, nil // too long to wait; the router fails over
				}
				d, useBackoff = sd, false
			}
		}
		if useBackoff {
			d = backoff(retry + 1)
		}

		// 7. Deadline check: do not wait past the caller's deadline.
		if deadline, has := ctx.Deadline(); has && t.now().Add(d).After(deadline) {
			return res, err
		}

		// 8. Next body.
		var body io.ReadCloser
		if req.GetBody != nil {
			b, gerr := req.GetBody()
			if gerr != nil {
				return res, err
			}
			body = b
		}

		// 9. Log, then release the response.
		if t.log != nil {
			attrs := []any{"host", req.URL.Host}
			if res != nil {
				attrs = append(attrs, "status", res.StatusCode)
			} else {
				attrs = append(attrs, "err", err)
			}
			attrs = append(attrs, "attempt", retry+1, "wait", d)
			t.log.Warn("llm: retrying", attrs...)
		}
		drainClose(res)

		// 10. Wait.
		if serr := t.sleep(ctx, d); serr != nil {
			if body != nil {
				body.Close()
			}
			return nil, serr
		}

		// 11. Next request.
		r = req.Clone(ctx)
		if req.Body != nil && req.Body != http.NoBody {
			r.Body = body
		}
	}
}

// retryableStatus reports whether a status is worth another try.
func retryableStatus(code int) bool {
	switch code {
	case 429, 500, 502, 503, 504, 529:
		return true
	}
	return false
}

// serverWait reads the wait the server asked for. retry-after-ms wins over
// Retry-After (seconds or an HTTP date); an unusable header falls through to the
// next one. false means no usable header.
func serverWait(h http.Header, now time.Time) (time.Duration, bool) {
	if v := strings.TrimSpace(h.Get("retry-after-ms")); v != "" {
		if ms, err := strconv.ParseFloat(v, 64); err == nil && !math.IsNaN(ms) && !math.IsInf(ms, 0) && ms >= 0 {
			if ms >= float64(math.MaxInt64)/float64(time.Millisecond) {
				return time.Duration(math.MaxInt64), true // saturate: the caller sees "too long"
			}
			return time.Duration(ms * float64(time.Millisecond)), true
		}
	}
	if v := strings.TrimSpace(h.Get("Retry-After")); v != "" {
		if secs, err := strconv.Atoi(v); err == nil {
			if int64(secs) > math.MaxInt64/int64(time.Second) {
				return time.Duration(math.MaxInt64), true
			}
			if secs >= 0 {
				return time.Duration(secs) * time.Second, true
			}
		} else if date, err := http.ParseTime(v); err == nil {
			return max(date.Sub(now), 0), true
		}
	}
	return 0, false
}

// backoff is the wait before retry n (1-based): 500ms, then 1s, less up to 25%
// jitter, so the result is in (0.75*base, base].
func backoff(retry int) time.Duration {
	base := 500 * time.Millisecond << (retry - 1)
	return base - time.Duration(rand.Float64()*0.25*float64(base))
}

// sleepCtx waits d, or returns ctx.Err() as soon as ctx ends. When d <= 0 it
// returns ctx.Err() (nil while ctx is alive) without a timer.
func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(d)
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		timer.Stop()
		return ctx.Err()
	}
}

// drainClose reads up to 4 KiB of the body so the connection can be reused, then
// closes it. nil res or nil Body is fine.
func drainClose(res *http.Response) {
	if res == nil || res.Body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(res.Body, 4<<10))
	res.Body.Close()
}
