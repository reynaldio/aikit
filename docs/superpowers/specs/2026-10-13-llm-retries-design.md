# Same-model retries in `aikit/llm` — design

Date: 2026-10-13. Status: approved in brainstorming, waiting for spec review.

First of four sub-projects, in this order: retries → helpers (`llmtest`, usage wrapper, tool loop) →
`llm.Embed` → Batch. Each one gets its own spec, plan and build.

## Why

When a provider answers 429 or 503, the router fails over straight to another model. A throttle or overload
that lasts a second or two then sends the call to a different model at a different price, for no good
reason. One or two short waits on the same model would usually be enough.

What exists today:

- **Anthropic:** the SDK retries on its own (`MaxRetries: 2`, backoff, and it follows `Retry-After` however
  long that is).
- **Google and OpenAI-compatible:** no retries at all. Errors become strings
  (`"gemini: … (status 503)"`), so the router never sees the status code or `Retry-After`.

## Decisions (owner answers)

| Question | Decision |
| --- | --- |
| Policy | Short: at most 2 retries, and no wait longer than 10s |
| Long `Retry-After` (> 10s) | Don't retry. Return the response at once, so the router fails over as it does today |
| Configurable? | No. Fixed policy, no new `Config` fields |
| Where | One `http.RoundTripper` shared by all three providers (approach A). The router doesn't change |
| Anthropic SDK's own retries | Turned off (`option.WithMaxRetries(0)`), so there is one policy in one place |

Rejected: retrying in the router around `completeOn` (approach B). Every provider would first need typed
errors with the status and `Retry-After`, and Anthropic would retry in two layers.

## Behavior

### What is retried

| Outcome of one attempt | Retried? |
| --- | --- |
| Status 429, 500, 502, 503, 504, 529 | Yes |
| Connection error with no response (refused, reset, unexpected EOF before headers) | Yes |
| Any other status, including every 4xx except 429 | No |
| `ctx` already ended | No; return `ctx.Err()` |
| 2xx | No; returned untouched, including a streaming body |

The decision is made from the status line and headers alone. The transport never reads a 2xx body, so a
streamed reply is never retried after its first byte. That keeps streaming rule 5 (no fallback once text has
been sent) true without the router doing anything new.

### Wait before each retry (rule order)

1. Read the server's request, first match wins:
   - `retry-after-ms` header: milliseconds (a decimal number).
   - `Retry-After` header: whole seconds, or an HTTP date (wait = date − now, at least 0).
2. If the server asked for **more than 10s**: no retry. Return this response as is.
3. If the server asked for 10s or less: wait exactly that long.
4. With no usable header: back off 0.5s before retry 1 and 1s before retry 2, each reduced by a random
   0–25% (jitter).
5. If `ctx` has a deadline and `now + wait` is past it: no retry. Return this response as is (the router may
   still fail over, which is better than spending the rest of the deadline asleep).
6. Wait, and stop at once if `ctx` ends: return `ctx.Err()` and close the held response body.

At most 2 retries, so at most 3 attempts per provider call.

### Each retry

- The previous response body is drained (a capped read of up to 4 KiB) and closed before the next attempt, so
  the connection can be reused.
- The request body is replayed through `req.GetBody`. All our bodies are byte buffers, so `GetBody` is set. If
  it is ever nil while there is a body, don't retry; return the response as is.
- One `Warn` log to `Config.Logger`: message `llm: retrying`, fields `host`, `status` (or `err`), `attempt`
  (1 or 2) and `wait`.

The final failure is returned exactly as one attempt would return it, so provider error text, the router's
`shouldFailover` and every existing error test stay the same.

## Files

| File | Responsibility |
| --- | --- |
| `llm/retry.go` (new) | `retryTransport`: the policy above. Header parsing and the delay calculation are separate small functions |
| `llm/httpclient.go` | `newProviderHTTPClient(lg *slog.Logger)` wraps its transport in `retryTransport` |
| `llm/google.go`, `llm/openai.go` | Pass the logger to `newProviderHTTPClient` |
| `llm/anthropic.go` | `newAnthropic(apiKey string, lg *slog.Logger, opts ...option.RequestOption)` adds `option.WithHTTPClient(newProviderHTTPClient(lg))` and `option.WithMaxRetries(0)`. `opts` lets tests set `option.WithBaseURL` |
| `llm/llm.go` | `New` passes the logger to `newAnthropic` |
| `README.md` | A short "Retries" paragraph next to the failover text |

### Shared names

```go
type retryTransport struct {
    base       http.RoundTripper
    log        *slog.Logger
    maxRetries int                                       // 2
    maxWait    time.Duration                             // 10 * time.Second
    sleep      func(ctx context.Context, d time.Duration) error // tests swap in a fake
    now        func() time.Time                          // for HTTP-date Retry-After
}

func (t *retryTransport) RoundTrip(req *http.Request) (*http.Response, error)
func retryableStatus(code int) bool
func serverWait(h http.Header, now time.Time) (time.Duration, bool) // false = no usable header
func backoff(retry int) time.Duration                                // retry is 1-based
```

The Anthropic client's own request timeout still comes from the SDK, through the context. The shared
`http.Client` keeps having no whole-request `Timeout`, as `httpclient.go` explains.

## Behavior change (release note)

Anthropic used to wait whatever `Retry-After` asked for, up to two times. A wait longer than 10s now fails
over at once instead of waiting, and the Anthropic retry backoff is now the same as Google's and OpenAI's.

## Out of scope

- Reading Gemini's `RetryInfo.retryDelay` from the error body. When the daily quota is used up, about 1.5s
  goes on two pointless retries before failover. That's acceptable.
- Retries for `aikit/decide` (it documents "back off yourself"). It can use this transport later if wanted.
- A `Config` knob for the policy.

## Tests

All with a fake `sleep` (it records the waits and returns at once) and `httptest` servers that count
attempts.

`llm/retry_test.go` (transport level):

1. 503 then 200 → 200 returned, 2 attempts, one wait in [0.375s, 0.5s].
2. 503 ×3 → the third 503 is returned, 3 attempts, waits 0.5s then 1s (within jitter).
3. 429 with `Retry-After: 30` → 1 attempt, no wait, the 429 is returned.
4. 429 with `Retry-After: 3` → wait exactly 3s, then success.
5. `Retry-After` as an HTTP date 5s ahead (fake `now`) → wait 5s.
6. `retry-after-ms: 1500` → wait 1.5s; it wins over `Retry-After` when both are present.
7. 400, 401, 404 → 1 attempt each.
8. Connection refused, then success (a listener closed for the first attempt) → 2 attempts.
9. `ctx` cancelled during the wait → `ctx.Err()`, no further attempt.
10. A `ctx` deadline closer than the wait → no retry, the 503 is returned.
11. The POST body arrives intact on every attempt.
12. `serverWait` and `backoff` table tests (garbage headers, negative values, a date in the past).

Provider and router level:

13. Gemini: 503 then 200 → `Complete` succeeds, no failover.
14. OpenAI streaming: 503 then an SSE reply → the events arrive in full and the join rule holds.
15. Anthropic: 529 ×3 against a test server (`option.WithBaseURL`) → exactly 3 attempts, which proves the
    SDK's own retries are off.
16. Router: the primary returns 503 ×3 → failover to the fallback, which answers.
17. Router: the primary returns 429 with `Retry-After: 60` → 1 attempt, then failover right away.

## Gates

```
go vet ./...
go test -race ./...
```
