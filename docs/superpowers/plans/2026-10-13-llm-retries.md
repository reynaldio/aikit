# Same-Model Retries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retry a call on the same model, at most twice with short waits, when a provider answers 429/5xx or
the connection fails. A long throttle still fails over at once, as it does today.

**Architecture:** One `http.RoundTripper` (`retryTransport`, new file `llm/retry.go`) wraps the transport in
`newProviderHTTPClient`, and all three providers use that client. The Anthropic SDK's own retries are turned
off, so the policy lives in one place. The router does not change. Two deferred OpenAI follow-ups are
folded in, because they touch the same code: one shared request helper in `openai.go`, and the missing OpenAI
stream tests.

**Tech Stack:** Go 1.25, `net/http`, `log/slog`, `math/rand/v2`, `github.com/anthropics/anthropic-sdk-go`
v1.56.0. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-10-13-llm-retries-design.md](../specs/2026-10-13-llm-retries-design.md)
(commit 7a6f505). Read it before your task. The decisions in it are final. Do not add `Config` fields, do not
make the policy configurable, and do not retry in the router.

This is sub-project 1 of 4 (retries → helpers → Embed → Batch). Nothing here is for the later ones.

## Global Constraints

- Branch: `feat/llm-retries` (already created; this plan is its first commit). Commit once per task. End every
  commit message with a blank line, then
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` on its own line.
- No new module dependencies. No new `Config` fields. No exported names: everything new is package-private.
- Policy constants: at most **2** retries (3 attempts per provider call). No wait longer than **10s**.
- Retryable statuses: exactly **429, 500, 502, 503, 504, 529**. Everything else, including every other 4xx,
  is returned after one attempt.
- The transport never reads a 2xx body. A 2xx response, streaming or not, is returned untouched.
- The final failure is returned exactly as one attempt would return it: the same `*http.Response` (with its
  body still unread) or the same error. Provider error text, `shouldFailover` and every existing test stay
  the same.
- The shared `http.Client` keeps having **no** whole-request `Timeout` (see the comment in
  `llm/httpclient.go`). Do not add one.
- A nil `*slog.Logger` anywhere in the new code means "do not log". It must never panic.
- Tests never sleep for real policy waits. Use the fake `sleep`, or a server that sends `retry-after-ms: 1`.
- Gate commands (run from the repo root):
  - Targeted, during a task: `go test -race ./llm -run '<pattern>'`
  - Before each commit: `go vet ./...`
  - Once, at the end (Task 5): `go vet ./...` and `go test -race ./...`

---

## File map

| File | Change | Responsibility after this plan |
| --- | --- | --- |
| `llm/retry.go` | new | `retryTransport` and its helpers: which outcomes are retried, how long to wait, the attempt loop |
| `llm/retry_test.go` | new | Transport tests (spec cases 1–12, plus a few extra edge cases) |
| `llm/retry_provider_test.go` | new | Provider and router tests (spec cases 13–17) |
| `llm/httpclient.go` | modify | `newProviderHTTPClient(lg *slog.Logger)` wraps its transport in `retryTransport` |
| `llm/google.go` | modify | `newGoogle` takes the logger and passes it to `newProviderHTTPClient` |
| `llm/openai.go` | modify | `newOpenAI` passes its logger to `newProviderHTTPClient`. One shared `post` helper for the request, headers and non-2xx handling (follow-up 1) |
| `llm/anthropic.go` | modify | `newAnthropic(apiKey, lg, opts...)` uses the shared client and `option.WithMaxRetries(0)` |
| `llm/llm.go` | modify | `New` passes the logger to `newAnthropic` and `newGoogle` |
| `llm/openai_stream_test.go` | modify | Missing branch tests, and the plain half of `TestOpenAIStreamRequestShape` uses the shared fake (follow-up 2) |
| `README.md` | modify | A short `### Retries` subsection, including the Anthropic behavior change |

## Shared names (exact)

All in package `llm`, file `llm/retry.go` unless noted.

```go
type retryTransport struct {
    base       http.RoundTripper
    log        *slog.Logger                                     // nil = no log
    maxRetries int                                              // 2
    maxWait    time.Duration                                    // 10 * time.Second
    sleep      func(ctx context.Context, d time.Duration) error // default sleepCtx; tests swap in a fake
    now        func() time.Time                                 // default time.Now; for HTTP-date Retry-After and the deadline check
}

func newRetryTransport(base http.RoundTripper, lg *slog.Logger) *retryTransport // fills the defaults above
func (t *retryTransport) RoundTrip(req *http.Request) (*http.Response, error)

func retryableStatus(code int) bool
func serverWait(h http.Header, now time.Time) (time.Duration, bool) // false = no usable header
func backoff(retry int) time.Duration                                // retry is 1-based: 1 or 2
func sleepCtx(ctx context.Context, d time.Duration) error           // waits d; returns ctx.Err() at once if ctx ends first
func drainClose(res *http.Response)                                  // reads up to 4 KiB of the body, then closes it

// llm/httpclient.go
func newProviderHTTPClient(lg *slog.Logger) *http.Client

// llm/google.go
func newGoogle(apiKey string, timeout time.Duration, lg *slog.Logger) provider

// llm/anthropic.go
func newAnthropic(apiKey string, lg *slog.Logger, opts ...option.RequestOption) provider

// llm/openai.go (Task 3)
func (o *openaiProvider) post(ctx context.Context, oaiReq oaiRequest) (*http.Response, error)
```

Log line (one per retry, level Warn): message `llm: retrying`, then the fields in this order: `host`
(`req.URL.Host`), then `status` (int) when there was a response or `err` (error) when there was not, then
`attempt` (1 or 2, meaning which retry this is), then `wait` (`time.Duration`).

---

### Task 1: `retryTransport` and its helpers

**Files:**
- Create: `llm/retry.go`
- Test: `llm/retry_test.go`

**Interfaces:**
- Consumes: nothing new.
- Produces: everything in `llm/retry.go` listed under "Shared names". Task 2 calls
  `newRetryTransport(base, lg)`. The tests in Task 2 rely on the transport honouring `retry-after-ms`.

**What to build**

`retryableStatus`: true for 429, 500, 502, 503, 504 and 529 only.

`serverWait(h, now)`. The first match wins:
1. `retry-after-ms`: parse with `strconv.ParseFloat(v, 64)`. It is usable only when the parse succeeds and the
   value is finite and `>= 0`. The wait is `time.Duration(ms * float64(time.Millisecond))`. If it is not usable,
   go on to step 2. Do not return false yet.
2. `Retry-After`: if `strconv.Atoi` succeeds and the value is `>= 0`, the wait is that many seconds. Otherwise,
   if `http.ParseTime` succeeds, the wait is `date.Sub(now)`, raised to 0 if it is negative. Otherwise it is
   not usable.
3. If neither header is usable, return `(0, false)`.

Trim spaces before parsing. An empty header counts as absent.

`backoff(retry)`: the base is 500ms for retry 1 and 1s for retry 2 (`500ms << (retry-1)`). Return
`base - time.Duration(rand.Float64()*0.25*float64(base))`. The result is in `(0.75·base, base]`. Use
`math/rand/v2`.

`sleepCtx(ctx, d)`: a `time.NewTimer(d)` and a `select` on `ctx.Done()`. Stop the timer on the ctx branch.
When `d <= 0`, return `ctx.Err()` (nil if ctx is still alive) without a timer.

`drainClose(res)`: `io.Copy(io.Discard, io.LimitReader(res.Body, 4<<10))`, then `res.Body.Close()`. It accepts
nil `res` and nil `res.Body`.

**`RoundTrip` — exact rule order (spell this out in code comments too)**

Let `ctx := req.Context()`. Loop with `retry := 0, 1, 2, …`. `r` is the request for this attempt: `req` itself
on the first attempt. Never modify `req`.

1. `res, err := t.base.RoundTrip(r)`.
2. **Success or not retryable:** if `err == nil && !retryableStatus(res.StatusCode)`, return `res, nil`
   untouched. This covers every 2xx, so a streaming body is never read here.
3. **Context ended:** if `ctx.Err() != nil`, call `drainClose(res)` when `res != nil`, then return
   `nil, ctx.Err()`.
4. **Out of retries:** if `retry == t.maxRetries`, return `res, err` as is.
5. **Body cannot be replayed:** if `req.Body != nil && req.Body != http.NoBody && req.GetBody == nil`, return
   `res, err` as is. (This is only a check. `GetBody` is called in step 8.)
6. **Pick the wait:**
   - If `res != nil`, call `d, ok := serverWait(res.Header, t.now())`.
     - If `ok` and `d > t.maxWait`, return `res, nil` as is. No retry: the router fails over.
     - If `ok`, the wait is exactly `d`.
   - Otherwise (no response, or no usable header), `d = backoff(retry+1)`.
7. **Deadline check:** if `deadline, has := ctx.Deadline(); has && t.now().Add(d).After(deadline)`, return
   `res, err` as is. No retry.
8. **Next body:** when `req.GetBody != nil`, call it now. If it returns an error, return `res, err` as is (the
   `GetBody` error is dropped). From here on, close this new body on every early return.
9. **Log, then release the response:** `t.log.Warn("llm: retrying", ...)` with the fields from "Shared
   names", skipped if `t.log == nil`. Then call `drainClose(res)` if `res != nil`.
10. **Wait:** `if err := t.sleep(ctx, d); err != nil`, close the body from step 8 (if any) and
    `return nil, err`. That `err` is `ctx.Err()`. The response was already closed in step 9.
11. **Next request:** `r = req.Clone(ctx)`, then set `r.Body` to the body from step 8 (leave it as is when
    `req.Body` is nil or `http.NoBody`). Continue the loop.

Every return in steps 2–7 happens before anything is drained, closed or fetched. So a response returned "as
is" always has its body unread and open.

Notes:
- Step 3 comes before step 4 on purpose. A cancelled call returns `ctx.Err()` even on its last attempt, which
  matches what `http.Client` reports today (it wraps the error, and `errors.Is(err, context.Canceled)` still
  holds).
- Any non-nil `err` with a live ctx counts as a connection failure and is retried (refused, reset, EOF before
  headers). Do not try to classify errors further.
- `t.now()` is used for both `serverWait` and the deadline check, so a test with a fake clock stays
  consistent. A test that needs the real deadline check uses the default `time.Now`.

**Test cases** (`llm/retry_test.go`)

Test helpers:
- `fakeSleep` records every `d` in a slice and returns nil. A variant can run a hook first (for case 9).
- `countingServer(t, statuses []int, headers []http.Header)` is an `httptest.Server` that answers request *n*
  with `statuses[n]` (the last entry repeats). It records each request body and counts attempts. The success
  body is `ok`.
- `newTestTransport(lg)` returns `newRetryTransport(http.DefaultTransport, lg)` with `sleep` set to the fake.
  Wrap it in `&http.Client{Transport: tr}`.

Waits are asserted from the fake's record. A "within jitter" range is inclusive: `[0.375s, 0.5s]` for retry 1
and `[0.75s, 1s]` for retry 2.

1. `TestRetry503Then200`: 200 is returned, 2 attempts, one wait in `[0.375s, 0.5s]`. Pass a logger that writes
   into a buffer, and check that exactly one `llm: retrying` line has `status=503 attempt=1`.
2. `TestRetry503ThreeTimesReturnsLast`: the third 503 comes back. Its body is still readable and equals the
   server's third body. 3 attempts. Waits are in `[0.375s, 0.5s]`, then in `[0.75s, 1s]`.
3. `TestRetryAfterTooLongNoRetry`: 429 with `Retry-After: 30` gives 1 attempt, no wait, and the 429 is returned.
4. `TestRetryAfterSeconds`: 429 with `Retry-After: 3`, then 200. The wait is exactly 3s.
5. `TestRetryAfterHTTPDate`: `now` is fixed at T. `Retry-After` is T+5s in `http.TimeFormat`, then 200. The
   wait is exactly 5s.
6. `TestRetryAfterMsWins`: `retry-after-ms: 1500` alone gives a wait of 1.5s. With `retry-after-ms: 1500`
   **and** `Retry-After: 7`, the wait is 1.5s.
7. `TestNoRetryOnClientErrors`: table of 400, 401 and 404. Each gives 1 attempt and no wait.
8. `TestRetryConnectionRefused`: the base is a `roundTripFunc` that returns
   `&net.OpError{Op: "dial", Err: syscall.ECONNREFUSED}` on the first call and calls `http.DefaultTransport`
   after that. The result is 200, 2 attempts, and one backoff wait. The log line has `err=` and no `status=`.
9. `TestRetryCancelledDuringWait`: the fake sleep cancels the ctx, then returns `ctx.Err()`. `Do` gives an
   error with `errors.Is(err, context.Canceled)`. The server saw exactly 1 attempt.
10. `TestRetryDeadlineTooClose`: real `now`, `ctx` with a 1s timeout, 503 with `Retry-After: 3`. The 503 is
    returned after 1 attempt, with no wait. Add a second case with no header and a 100ms deadline: the 503 is
    returned after 1 attempt, because the backoff of ≥375ms passes the deadline.
11. `TestRetryReplaysBody`: POST `{"n":1}` via `http.NewRequestWithContext(…, bytes.NewReader(...))`. With
    503, 503, 200, the server records the same body 3 times.
12. `TestServerWait` and `TestBackoff`, both table tests.
    - `serverWait` cases: no headers gives `(0,false)`. `Retry-After: abc` gives false. `Retry-After: -1`
      gives false. `retry-after-ms: -5` with no `Retry-After` gives false. `retry-after-ms: x` with
      `Retry-After: 2` gives `(2s,true)`. `retry-after-ms: 0` gives `(0,true)`. `retry-after-ms: NaN` gives
      false. A past HTTP date gives `(0,true)`. `Retry-After: " 4 "` gives `(4s,true)`.
    - `backoff` runs 1000 iterations each. `backoff(1)` stays in `(375ms, 500ms]` and `backoff(2)` stays in
      `(750ms, 1s]`.

Extra cases (not numbered in the spec, but required):
- `TestNoRetryWithoutGetBody`: a request with a body and `GetBody = nil` gets 1 attempt, and the 503 is
  returned.
- `TestRetryClosesPreviousBody`: the base is a `roundTripFunc` that returns 503 responses whose bodies record
  `Close`. Before the second attempt starts, the first body is closed.
- `TestRetryNilLogger`: case 1 with a nil logger does not panic.
- `TestSleepCtx`: with an already-cancelled ctx and a 1h wait, it returns `context.Canceled` at once. A 1ms
  wait returns nil.
- `TestRetry2xxBodyUntouched`: the base returns 200 with a body that fails the test if `Read` is called
  before the test reads it. The transport returns it, and the test reads it fine.

**Steps**

- [ ] Write `llm/retry_test.go` with the cases above. Run `go test -race ./llm -run 'Retry|ServerWait|Backoff|SleepCtx'`.
  It is expected to FAIL to compile (`undefined: newRetryTransport`).
- [ ] Write `llm/retry.go`. Keep the helpers small and separate. Give `RoundTrip` comments that number the
  steps the same way as above.
- [ ] Run the same command. Expected: PASS. Then run `go vet ./...`.
- [ ] Commit: `feat(llm): retry transport for 429/5xx and connection errors`.

---

### Task 2: Wire the transport into all three providers

**Files:**
- Modify: `llm/httpclient.go` (`newProviderHTTPClient`), `llm/google.go` (`newGoogle`), `llm/openai.go`
  (`newOpenAI`), `llm/anthropic.go` (`newAnthropic`), `llm/llm.go` (`New`)
- Test: `llm/retry_provider_test.go`

**Interfaces:**
- Consumes: `newRetryTransport(base http.RoundTripper, lg *slog.Logger) *retryTransport` (Task 1).
- Produces: `newProviderHTTPClient(lg *slog.Logger) *http.Client`,
  `newGoogle(apiKey string, timeout time.Duration, lg *slog.Logger) provider`,
  `newAnthropic(apiKey string, lg *slog.Logger, opts ...option.RequestOption) provider`.

**What to build**

- `newProviderHTTPClient(lg)`: build the same cloned `*http.Transport` as today, and return
  `&http.Client{Transport: newRetryTransport(t, lg)}`. Extend the doc comment with one sentence that says
  retries happen here (link the spec), and that a 2xx body is never read, so streaming is safe. Keep the
  "no Timeout" paragraph.
- `newGoogle(apiKey, timeout, lg)`: pass `lg` to `newProviderHTTPClient`. Do not add a logger field to
  `googleProvider`.
- `newOpenAI(...)`: it already takes `log`. Pass it to `newProviderHTTPClient(log)`.
- `newAnthropic(apiKey, lg, opts...)`: build the client from `option.WithAPIKey(apiKey)`,
  `option.WithHTTPClient(newProviderHTTPClient(lg))` and `option.WithMaxRetries(0)`, followed by `opts...`.
  `opts` come last so a test can add `option.WithBaseURL`. Add a comment: the SDK's own retries are off so
  that one policy applies (link the spec). The SDK sets `GetBody` and puts its per-attempt timeout in the
  request context, so the transport's deadline check sees it.
- `New`: call `newAnthropic(cfg.AnthropicAPIKey, lg)` and `newGoogle(cfg.GoogleAPIKey, cfg.RequestTimeout, lg)`.
  `lg` is already non-nil there (the discard logger), so it is fine to pass.

Existing tests build providers by hand with `http.DefaultClient` or their own `anthropic.NewClient(...)`, so
they do not change. Fix any other caller the compiler reports. There should be none outside `llm.go`.

**Test cases** (`llm/retry_provider_test.go`)

Every server below that answers a retryable status also sends `retry-after-ms: 1`, so the real transport
waits 1ms. Providers are built with the **real** constructors' client: `newProviderHTTPClient(nil)`, or
`newAnthropic("k", nil, option.WithBaseURL(srv.URL))`.

13. `TestGeminiRetries503ThenSucceeds`: `&googleProvider{apiKey:"k", baseURL: srv.URL, http: newProviderHTTPClient(nil)}`.
    The server answers 503, then a valid `generateContent` JSON reply. `complete` succeeds with the expected
    text, after 2 attempts.
14. `TestOpenAIStreamRetries503ThenStreams`: `oaiStreamFake` cannot vary its status per attempt, so write a
    small server here. Attempt 1 is 503 with a JSON error body. Attempt 2 is SSE: `oaiTextChunk("Hel")`,
    `oaiTextChunk("lo")`, `oaiFinishChunk("stop")`, `oaiUsageChunk`, `[DONE]`. Pass `streamReq(collectText(&got))`.
    The result: `got == ["Hel","lo"]`, `strings.Join(got,"") == resp.Text`, and 2 attempts.
15. `TestAnthropicSDKRetriesAreOff`: every request gets 529 with an Anthropic error body
    (`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`) and `retry-after-ms: 1`.
    `complete` returns an error after **exactly 3** attempts. With the SDK's retries still on, it would be 9.
    Also check that the error still makes `shouldFailover` return true.
16. `TestRouterFailsOverAfterRetries`: build a `router` by hand. The primary is Google (server: always 503 with
    a Gemini error JSON and `retry-after-ms: 1`). The fallback is an `openaiProvider` (server: a plain
    non-stream 200 reply). Both use `newProviderHTTPClient(nil)`. Set profiles/fallbacks so that `TaskChat` →
    Google `flash` with an OpenAI fallback, and set `log` to a discard logger. The result: `Complete` succeeds,
    `resp.Provider == ProviderOpenAI`, the primary saw 3 attempts and the fallback saw 1.
17. `TestRouterFailsOverAtOnceOnLongRetryAfter`: same setup, but the primary answers 429 with
    `Retry-After: 60`. The primary saw **1** attempt, then the call fails over and succeeds. The test must
    finish in well under a second. Assert `time.Since(start) < 2*time.Second`.

**Steps**

- [ ] Write `llm/retry_provider_test.go`. Run `go test -race ./llm -run 'Retries|SDKRetries|RouterFailsOver'`.
  It is expected to FAIL to compile (the constructor signatures).
- [ ] Make the constructor and `New` changes.
- [ ] Run the targeted tests. Expected: PASS. Then run `go test -race ./llm` once, because constructors are
  shared, and `go vet ./...`.
- [ ] Commit: `feat(llm): all providers retry through one transport; Anthropic SDK retries off`.

---

### Task 3: One request helper for OpenAI (deferred follow-up 1)

`complete` and `completeStream` in `llm/openai.go` repeat the marshal, the request build, the headers, `Do`
and the non-2xx handling. Retries now run under both, so make them share one path.

**Files:**
- Modify: `llm/openai.go` (`complete`, `completeStream`; add `post`)
- Test: `llm/openai_stream_test.go` (one new test for the plain path)

**Interfaces:**
- Consumes: nothing from Tasks 1–2 directly. `o.http` is whatever client the provider has.
- Produces: `func (o *openaiProvider) post(ctx context.Context, oaiReq oaiRequest) (*http.Response, error)`.

**What to build**

`post` marshals `oaiReq`, builds `POST {baseURL}/chat/completions` with `bytes.NewReader` (so `GetBody` is set),
sets `Content-Type: application/json` and `Authorization: Bearer <key>`, and calls `o.http.Do`.
- On a `Do` error, it returns `nil, err` unchanged.
- On a status `>= 300`, it reads the whole body (a read failure gives `readError(ctx, "openai", err)`), closes
  it, and returns `nil, fmt.Errorf("openai: %s (status %d)", msg, code)`. `msg` is the JSON `error.message`
  when the body parses and has one. Otherwise it is the trimmed raw body. This is today's `completeStream`
  logic, moved.
- On a 2xx, it returns the open response. The caller closes it.

`complete`, when there is no `OnEvent`, calls `post`, reads and decodes the 2xx body as today, and still
treats a 2xx body that carries `error` as an error (`openai: <message> (status 200)`). `completeStream` calls
`post` and keeps everything after the status check unchanged.

**Behavior change (intended, small).** Today a non-2xx reply on the plain path whose body is not JSON (an HTML
502 from a proxy, say) returns `openai: decode response: …`. That text has no status code, so `shouldFailover`
may not fail over. After this task it returns `openai: <body> (status 502)`, the same as the stream path. Say
so in the commit message.

**Test cases**
- Every existing OpenAI test in `llm_test.go`, `openai_stream_test.go` and `schema_test.go` passes unchanged.
- New: `TestOpenAIPlainNonJSONErrorHasStatus`: the server answers 502 with the body `<html>bad gateway</html>`
  and no retry header. The provider uses `http.DefaultClient`, so there are no retries. The error contains
  `(status 502)` and `bad gateway`, and `shouldFailover(err)` is true.
- New: `TestOpenAIPlainErrorInto200Body`: 200 with `{"error":{"message":"boom"}}` gives an error containing
  `boom` and `(status 200)`. This is today's behavior, now pinned by a test.

**Steps**
- [ ] Add the two tests. Run `go test -race ./llm -run 'OpenAIPlain'`. The first should FAIL (it says
  `decode response`). The second should PASS already.
- [ ] Add `post`, and switch `complete` and `completeStream` to it.
- [ ] Run `go test -race ./llm -run 'OpenAI|Schema|Refusal|Retries|RouterFailsOver'`. Expected: PASS. Then
  run `go vet ./...`.
- [ ] Commit: `refactor(llm): one OpenAI request path; plain non-JSON errors carry the status`.

---

### Task 4: OpenAI stream test gaps (deferred follow-up 2)

Tests only. No change to `llm/openai.go` unless a new test exposes a real bug. If one does, stop and report
it to the controller, and do not fix it inside this task.

**Files:**
- Modify: `llm/openai_stream_test.go`

**Interfaces:**
- Consumes: `oaiStreamFake` and its `start(t, logs)` helper, `streamReq`, `collectText`, `oaiTextChunk`.
- Produces: an `oaiStreamFake` that can also answer a plain (non-SSE) body.

**What to build**

- Give `oaiStreamFake` a `plain string` field. When it is set and the status is 2xx, the handler writes it as
  one `application/json` body instead of SSE frames.
- Rewrite the plain half of `TestOpenAIStreamRequestShape` to use `(&oaiStreamFake{plain: ...}).start(t, nil)`
  instead of its own `httptest.NewServer`. Keep the same assertions.

**Test cases** (new)
- `TestOpenAIStreamMalformedChunk`: chunks are `oaiTextChunk("a")` and then `{not json`. The error contains
  `openai: decode stream chunk`, and `got == ["a"]` (the text before the bad chunk was delivered).
- `TestOpenAIStreamNonJSONErrorBody`: status 500 (the fake sends no retry header, and the provider uses
  `http.DefaultClient`) with raw body `upstream exploded`. The error contains `upstream exploded` and
  `(status 500)`.
- `TestOpenAIStreamEOFAfterCancel`: this covers the branch in `completeStream` where the body hits `io.EOF`
  while `ctx.Err() != nil`. The existing `TestOpenAIStreamCancelledFromOnEvent` does not reach it, because it
  stops at the check after `OnEvent`. A real HTTP body turns cancellation into a read error, not EOF, so do
  not use a server. Give the provider `http: &http.Client{Transport: roundTripFunc(...)}`. The function
  returns a 200 `text/event-stream` response whose body is a custom `io.ReadCloser`. It first yields
  `data: {"choices":[{"index":0,"delta":{}}]}\n\n` (no text, so `OnEvent` is never called). On the next
  `Read` it calls the test's `cancel()` and returns `0, io.EOF`. Assert
  `errors.Is(err, context.Canceled)` and `!errors.Is(err, io.ErrUnexpectedEOF)`. Reuse the `roundTripFunc`
  type from `llm/retry_test.go` (same package).

**Steps**
- [ ] Add the `plain` field and rewrite the RequestShape plain half. Run
  `go test -race ./llm -run 'OpenAIStreamRequestShape'`. Expected: PASS.
- [ ] Add the three tests. Run `go test -race ./llm -run 'OpenAIStream'`. Expected: PASS.
- [ ] Run `go vet ./...`, then commit: `test(llm): OpenAI stream error branches; RequestShape uses the shared fake`.

---

### Task 5: README, final gates

Docs-only work (Haiku is fine), plus the full gate run.

**Files:**
- Modify: `README.md`

**What to write**

A `### Retries` subsection in the `aikit/llm` part, placed right after the routing paragraph that ends with
`` `Complete` returns `llm.ErrNotConfigured` when no provider key is set. `` and before
`### Refusals are errors, not empty strings`. Keep it to about 8–12 lines, in plain English:
- Before failover, a call is retried on the **same** model at most twice, for 429, 500, 502, 503, 504, 529
  and connection errors.
- The wait follows `retry-after-ms` / `Retry-After` when the server sends one. If the server asks for more
  than 10s, there is no retry and the call fails over at once. Without a header, the waits are 0.5s and then
  1s, a little less with jitter.
- No retry if the wait would pass the context deadline.
- A streamed reply is never retried once it has started, so the streaming fallback rule is unchanged.
- The policy is fixed. There is no `Config` setting for it. Each retry logs one `Warn` line `llm: retrying`
  to `Config.Logger`.
- **Behavior change:** Anthropic used to wait however long `Retry-After` asked. A wait over 10s now fails over
  at once, and Anthropic's backoff now matches the other providers.

Wrap lines at about 100 characters like the rest of the README. Also check the two lines in the Streaming
section that the memory note calls "left unwrapped", and wrap them if they are still long (cosmetic, same
file).

**Steps**
- [ ] Write the subsection.
- [ ] Full gates: `go vet ./...` and `go test -race ./...`. Both must pass. Paste the summary lines into the
  task report.
- [ ] Commit: `docs: retries in the llm README`.

---

## Review guidance (for the controller)

- Task 1: a Sonnet per-task review. Point the reviewer at the rule order (steps 1–11) and the ctx/deadline
  handling. These are the parts most likely to go subtly wrong: a body left open, a request modified in
  place, a retry after ctx ended.
- Task 2: a Sonnet per-task review (it changes constructors used everywhere).
- Tasks 3–4: low risk. The controller reads the diff; the final review covers them.
- Task 5: docs. No per-task review.
- Final whole-branch review: Opus, once.

## Spec coverage check

| Spec item | Task |
| --- | --- |
| Retry table (statuses, connection errors, ctx ended, 2xx untouched) | 1 (steps 2–3; cases 1, 7, 8, 9, extra 2xx case) |
| Wait rule order 1–6 | 1 (steps 6, 7, 10; cases 3–6, 9, 10, 12) |
| At most 2 retries | 1 (step 4; case 2) |
| Drain ≤4 KiB and close, `GetBody` replay, nil `GetBody` = no retry | 1 (steps 5, 8, 9, 11; case 11, extra cases) |
| One `Warn` log line with host/status-or-err/attempt/wait | 1 (step 9; cases 1, 8) |
| Final failure returned as one attempt would | 1 (step 4; case 2), 2 (case 15) |
| Shared transport in `newProviderHTTPClient`; logger plumbing | 2 |
| Anthropic `WithHTTPClient` + `WithMaxRetries(0)` + `opts` | 2 (case 15) |
| Spec tests 13–17 | 2 |
| README "Retries" + release note | 5 |
| Gates `go vet ./...`, `go test -race ./...` | 5 |
| Deferred follow-ups 1 and 2 (owner chose these on 2026-10-13) | 3, 4 |
