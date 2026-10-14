package llm

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// roundTripFunc adapts a function to http.RoundTripper.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// fakeSleep records every wait and returns nil; hook (optional) runs first.
type fakeSleep struct {
	mu    sync.Mutex
	waits []time.Duration
	hook  func(ctx context.Context) error
}

func (f *fakeSleep) sleep(ctx context.Context, d time.Duration) error {
	f.mu.Lock()
	f.waits = append(f.waits, d)
	f.mu.Unlock()
	if f.hook != nil {
		return f.hook(ctx)
	}
	return nil
}

type countingSrv struct {
	*httptest.Server
	mu       sync.Mutex
	attempts int
	bodies   []string
}

func (c *countingSrv) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.attempts
}

// countingServer answers request n with statuses[n] (the last entry repeats) and
// headers[n] (same rule; may be nil). The 2xx body is "ok"; others are "fail-<n>".
func countingServer(t *testing.T, statuses []int, headers []http.Header) *countingSrv {
	t.Helper()
	c := &countingSrv{}
	c.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		c.mu.Lock()
		n := c.attempts
		c.attempts++
		c.bodies = append(c.bodies, string(b))
		c.mu.Unlock()
		st := statuses[min(n, len(statuses)-1)]
		if len(headers) > 0 {
			for k, vs := range headers[min(n, len(headers)-1)] {
				for _, v := range vs {
					w.Header().Add(k, v)
				}
			}
		}
		w.WriteHeader(st)
		if st >= 200 && st < 300 {
			_, _ = w.Write([]byte("ok"))
		} else {
			_, _ = w.Write([]byte("fail-" + strconv.Itoa(n)))
		}
	}))
	t.Cleanup(c.Close)
	return c
}

func newTestTransport(lg *slog.Logger) (*retryTransport, *fakeSleep) {
	tr := newRetryTransport(http.DefaultTransport, lg)
	fs := &fakeSleep{}
	tr.sleep = fs.sleep
	return tr, fs
}

func hdr(kv ...string) http.Header {
	h := http.Header{}
	for i := 0; i+1 < len(kv); i += 2 {
		h.Set(kv[i], kv[i+1])
	}
	return h
}

func get(t *testing.T, tr http.RoundTripper, url string) (*http.Response, error) {
	t.Helper()
	return (&http.Client{Transport: tr}).Get(url)
}

func inRange(d, lo, hi time.Duration) bool { return d >= lo && d <= hi }

func TestRetry503Then200(t *testing.T) {
	srv := countingServer(t, []int{503, 200}, nil)
	var buf bytes.Buffer
	tr, fs := newTestTransport(slog.New(slog.NewTextHandler(&buf, nil)))
	res, err := get(t, tr, srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 || srv.count() != 2 {
		t.Fatalf("status=%d attempts=%d", res.StatusCode, srv.count())
	}
	if len(fs.waits) != 1 || !inRange(fs.waits[0], 375*time.Millisecond, 500*time.Millisecond) {
		t.Fatalf("waits = %v", fs.waits)
	}
	lines := 0
	for _, l := range strings.Split(buf.String(), "\n") {
		if strings.Contains(l, "llm: retrying") {
			lines++
			if !strings.Contains(l, "status=503") || !strings.Contains(l, "attempt=1") {
				t.Fatalf("log line = %q", l)
			}
		}
	}
	if lines != 1 {
		t.Fatalf("retry log lines = %d in %q", lines, buf.String())
	}
}

func TestRetry503ThreeTimesReturnsLast(t *testing.T) {
	srv := countingServer(t, []int{503}, nil)
	tr, fs := newTestTransport(nil)
	res, err := get(t, tr, srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	b, _ := io.ReadAll(res.Body)
	if res.StatusCode != 503 || string(b) != "fail-2" || srv.count() != 3 {
		t.Fatalf("status=%d body=%q attempts=%d", res.StatusCode, b, srv.count())
	}
	if len(fs.waits) != 2 ||
		!inRange(fs.waits[0], 375*time.Millisecond, 500*time.Millisecond) ||
		!inRange(fs.waits[1], 750*time.Millisecond, time.Second) {
		t.Fatalf("waits = %v", fs.waits)
	}
}

func TestRetryAfterHugeNoRetry(t *testing.T) {
	srv := countingServer(t, []int{429, 200}, []http.Header{hdr("Retry-After", "99999999999")})
	tr, fs := newTestTransport(nil)
	res, err := get(t, tr, srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 429 || srv.count() != 1 || len(fs.waits) != 0 {
		t.Fatalf("status=%d attempts=%d waits=%v", res.StatusCode, srv.count(), fs.waits)
	}
}

func TestRetryAfterTooLongNoRetry(t *testing.T) {
	srv := countingServer(t, []int{429, 200}, []http.Header{hdr("Retry-After", "30")})
	tr, fs := newTestTransport(nil)
	res, err := get(t, tr, srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 429 || srv.count() != 1 || len(fs.waits) != 0 {
		t.Fatalf("status=%d attempts=%d waits=%v", res.StatusCode, srv.count(), fs.waits)
	}
}

func TestRetryAfterSeconds(t *testing.T) {
	srv := countingServer(t, []int{429, 200}, []http.Header{hdr("Retry-After", "3")})
	tr, fs := newTestTransport(nil)
	res, err := get(t, tr, srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 || len(fs.waits) != 1 || fs.waits[0] != 3*time.Second {
		t.Fatalf("status=%d waits=%v", res.StatusCode, fs.waits)
	}
}

func TestRetryAfterHTTPDate(t *testing.T) {
	now := time.Date(2026, 10, 13, 12, 0, 0, 0, time.UTC)
	srv := countingServer(t, []int{429, 200},
		[]http.Header{hdr("Retry-After", now.Add(5*time.Second).Format(http.TimeFormat))})
	tr, fs := newTestTransport(nil)
	tr.now = func() time.Time { return now }
	res, err := get(t, tr, srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 || len(fs.waits) != 1 || fs.waits[0] != 5*time.Second {
		t.Fatalf("status=%d waits=%v", res.StatusCode, fs.waits)
	}
}

func TestRetryAfterMsWins(t *testing.T) {
	for name, h := range map[string]http.Header{
		"ms alone": hdr("retry-after-ms", "1500"),
		"ms+secs":  hdr("retry-after-ms", "1500", "Retry-After", "7"),
	} {
		t.Run(name, func(t *testing.T) {
			srv := countingServer(t, []int{429, 200}, []http.Header{h})
			tr, fs := newTestTransport(nil)
			res, err := get(t, tr, srv.URL)
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			if res.StatusCode != 200 || len(fs.waits) != 1 || fs.waits[0] != 1500*time.Millisecond {
				t.Fatalf("status=%d waits=%v", res.StatusCode, fs.waits)
			}
		})
	}
}

func TestNoRetryOnClientErrors(t *testing.T) {
	for _, code := range []int{400, 401, 404} {
		t.Run(strconv.Itoa(code), func(t *testing.T) {
			srv := countingServer(t, []int{code, 200}, nil)
			tr, fs := newTestTransport(nil)
			res, err := get(t, tr, srv.URL)
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			if res.StatusCode != code || srv.count() != 1 || len(fs.waits) != 0 {
				t.Fatalf("status=%d attempts=%d waits=%v", res.StatusCode, srv.count(), fs.waits)
			}
		})
	}
}

func TestRetryConnectionRefused(t *testing.T) {
	srv := countingServer(t, []int{200}, nil)
	calls := 0
	base := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		if calls == 1 {
			return nil, &net.OpError{Op: "dial", Err: syscall.ECONNREFUSED}
		}
		return http.DefaultTransport.RoundTrip(r)
	})
	var buf bytes.Buffer
	tr := newRetryTransport(base, slog.New(slog.NewTextHandler(&buf, nil)))
	fs := &fakeSleep{}
	tr.sleep = fs.sleep
	res, err := get(t, tr, srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 || calls != 2 || srv.count() != 1 {
		t.Fatalf("status=%d calls=%d", res.StatusCode, calls)
	}
	if len(fs.waits) != 1 || !inRange(fs.waits[0], 375*time.Millisecond, 500*time.Millisecond) {
		t.Fatalf("waits = %v", fs.waits)
	}
	out := buf.String()
	if !strings.Contains(out, "err=") || strings.Contains(out, "status=") {
		t.Fatalf("log = %q", out)
	}
}

func TestRetryCancelledDuringWait(t *testing.T) {
	srv := countingServer(t, []int{503, 200}, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	tr, fs := newTestTransport(nil)
	fs.hook = func(ctx context.Context) error {
		cancel()
		return ctx.Err()
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
	res, err := (&http.Client{Transport: tr}).Do(req)
	if err == nil {
		res.Body.Close()
		t.Fatal("want error")
	}
	if !errors.Is(err, context.Canceled) || srv.count() != 1 {
		t.Fatalf("err=%v attempts=%d", err, srv.count())
	}
}

func TestRetryCancelledDuringWaitClosesReplayBody(t *testing.T) {
	srv := countingServer(t, []int{503, 200}, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	tr, fs := newTestTransport(nil)
	fs.hook = func(ctx context.Context) error {
		cancel()
		return ctx.Err()
	}
	var replayClosed bool
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL, strings.NewReader("x"))
	req.GetBody = func() (io.ReadCloser, error) {
		return closeRecorder{strings.NewReader("x"), &replayClosed}, nil
	}
	res, err := (&http.Client{Transport: tr}).Do(req)
	if err == nil {
		res.Body.Close()
		t.Fatal("want error")
	}
	if !errors.Is(err, context.Canceled) || srv.count() != 1 || !replayClosed {
		t.Fatalf("err=%v attempts=%d replayClosed=%v", err, srv.count(), replayClosed)
	}
}

func TestRetryDeadlineTooClose(t *testing.T) {
	cases := []struct {
		name    string
		timeout time.Duration
		h       []http.Header
	}{
		{"retry-after past deadline", time.Second, []http.Header{hdr("Retry-After", "3")}},
		{"backoff past deadline", time.Second, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv := countingServer(t, []int{503, 200}, c.h)
			ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
			defer cancel()
			tr, fs := newTestTransport(nil)
			// A fixed clock 10ms before the deadline: every wait overshoots it,
			// however slow the round trip is.
			deadline, _ := ctx.Deadline()
			tr.now = func() time.Time { return deadline.Add(-10 * time.Millisecond) }
			req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
			res, err := (&http.Client{Transport: tr}).Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			if res.StatusCode != 503 || srv.count() != 1 || len(fs.waits) != 0 {
				t.Fatalf("status=%d attempts=%d waits=%v", res.StatusCode, srv.count(), fs.waits)
			}
		})
	}
}

func TestRetryReplaysBody(t *testing.T) {
	srv := countingServer(t, []int{503, 503, 200}, nil)
	tr, _ := newTestTransport(nil)
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, srv.URL,
		bytes.NewReader([]byte(`{"n":1}`)))
	res, err := (&http.Client{Transport: tr}).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	srv.mu.Lock()
	bodies := append([]string(nil), srv.bodies...)
	srv.mu.Unlock()
	if res.StatusCode != 200 || len(bodies) != 3 {
		t.Fatalf("status=%d bodies=%v", res.StatusCode, bodies)
	}
	for i, b := range bodies {
		if b != `{"n":1}` {
			t.Fatalf("body %d = %q", i, b)
		}
	}
}

func TestNoRetryWithoutGetBody(t *testing.T) {
	srv := countingServer(t, []int{503, 200}, nil)
	tr, fs := newTestTransport(nil)
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, srv.URL,
		strings.NewReader("x"))
	req.GetBody = nil
	res, err := (&http.Client{Transport: tr}).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 503 || srv.count() != 1 || len(fs.waits) != 0 {
		t.Fatalf("status=%d attempts=%d waits=%v", res.StatusCode, srv.count(), fs.waits)
	}
}

type closeRecorder struct {
	io.Reader
	closed *bool
}

func (c closeRecorder) Close() error { *c.closed = true; return nil }

func TestRetryClosesPreviousBody(t *testing.T) {
	var firstClosed, closedBeforeSecond bool
	calls := 0
	base := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		if calls == 2 {
			closedBeforeSecond = firstClosed
			return &http.Response{StatusCode: 200, Header: http.Header{},
				Body: io.NopCloser(strings.NewReader("ok")), Request: r}, nil
		}
		return &http.Response{StatusCode: 503, Header: http.Header{},
			Body: closeRecorder{strings.NewReader("busy"), &firstClosed}, Request: r}, nil
	})
	tr := newRetryTransport(base, nil)
	tr.sleep = (&fakeSleep{}).sleep
	req, _ := http.NewRequest(http.MethodGet, "http://example.invalid/", nil)
	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 || !closedBeforeSecond {
		t.Fatalf("status=%d closedBeforeSecond=%v", res.StatusCode, closedBeforeSecond)
	}
}

func TestRetryNilLogger(t *testing.T) {
	srv := countingServer(t, []int{503, 200}, nil)
	tr, _ := newTestTransport(nil)
	res, err := get(t, tr, srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("status = %d", res.StatusCode)
	}
}

func TestSleepCtx(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	start := time.Now()
	if err := sleepCtx(ctx, time.Hour); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if time.Since(start) > time.Second {
		t.Fatal("did not return at once")
	}
	if err := sleepCtx(context.Background(), time.Millisecond); err != nil {
		t.Fatalf("err = %v", err)
	}
	if err := sleepCtx(context.Background(), 0); err != nil {
		t.Fatalf("zero wait err = %v", err)
	}
}

type tripBody struct {
	read bool
	r    io.Reader
}

func (b *tripBody) Read(p []byte) (int, error) { b.read = true; return b.r.Read(p) }
func (b *tripBody) Close() error               { return nil }

func TestRetry2xxBodyUntouched(t *testing.T) {
	body := &tripBody{r: strings.NewReader("stream")}
	base := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: body, Request: r}, nil
	})
	tr := newRetryTransport(base, nil)
	req, _ := http.NewRequest(http.MethodGet, "http://example.invalid/", nil)
	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	if body.read {
		t.Fatal("transport read a 2xx body")
	}
	b, _ := io.ReadAll(res.Body)
	if string(b) != "stream" {
		t.Fatalf("body = %q", b)
	}
}

func TestServerWait(t *testing.T) {
	now := time.Date(2026, 10, 13, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name string
		h    http.Header
		want time.Duration
		ok   bool
	}{
		{"none", hdr(), 0, false},
		{"junk", hdr("Retry-After", "abc"), 0, false},
		{"negative secs", hdr("Retry-After", "-1"), 0, false},
		{"negative ms", hdr("retry-after-ms", "-5"), 0, false},
		{"bad ms falls through", hdr("retry-after-ms", "x", "Retry-After", "2"), 2 * time.Second, true},
		{"zero ms", hdr("retry-after-ms", "0"), 0, true},
		{"NaN ms", hdr("retry-after-ms", "NaN"), 0, false},
		{"past date", hdr("Retry-After", now.Add(-time.Minute).Format(http.TimeFormat)), 0, true},
		{"spaces", hdr("Retry-After", " 4 "), 4 * time.Second, true},
		{"empty", hdr("Retry-After", ""), 0, false},
		{"huge ms saturates", hdr("retry-after-ms", "1e300"), time.Duration(math.MaxInt64), true},
		{"huge secs saturates", hdr("Retry-After", "99999999999"), time.Duration(math.MaxInt64), true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, ok := serverWait(c.h, now)
			if d != c.want || ok != c.ok {
				t.Fatalf("got (%v,%v), want (%v,%v)", d, ok, c.want, c.ok)
			}
		})
	}
}

func TestBackoff(t *testing.T) {
	cases := []struct {
		retry  int
		lo, hi time.Duration
	}{
		{1, 375 * time.Millisecond, 500 * time.Millisecond},
		{2, 750 * time.Millisecond, time.Second},
	}
	for _, c := range cases {
		for i := 0; i < 1000; i++ {
			d := backoff(c.retry)
			if d <= c.lo || d > c.hi {
				t.Fatalf("backoff(%d) = %v, want in (%v, %v]", c.retry, d, c.lo, c.hi)
			}
		}
	}
}
