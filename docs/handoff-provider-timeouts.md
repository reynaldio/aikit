# Handoff → aikit: Gemini/OpenAI calls give up at a fixed 120 seconds

- **From:** dossio (Review & Analysis risk review), 2026-10-08
- **To:** the aikit maintainer session (`github.com/reynaldio/aikit`)
- **Against:** aikit **v0.6.0**
- **Suggested release:** **v0.7.0** (behavior change for callers that relied on the 120 s cutoff)

dossio does not change aikit itself; this note describes the change we need.

## Problem

The Google and OpenAI providers build their HTTP client with a hard-coded
whole-request timeout:

- `llm/google.go:29`: `&http.Client{Timeout: 120 * time.Second}`
- `llm/openai.go:41`: `&http.Client{Timeout: 120 * time.Second}`

`http.Client.Timeout` covers the whole exchange, including reading the
response body. Any reply that takes longer than 120 s to generate is cut
off, however much time the caller allowed. The caller's context deadline
cannot extend it: the shorter of the two always wins.

The Anthropic provider has no such cap. Since v0.6.0 it streams long requests
and relies on the caller's context, so the providers now behave differently
for the same `Request`.

## Evidence (dossio, 2026-10-08)

- dossio's long Review & Analysis calls (the risk report and the
  cross-document risk list, `MaxTokens` 24000, retried at 48000 on a cut-off)
  run under a per-call context deadline that scales with `MaxTokens`:
  1 h × MaxTokens / 128000 + a third, so 15 min at 24000 and 30 min at 48000.
- In a probe on a real 16-document run, `google:gemini-3.1-pro-preview`
  completed all 12 calls, but **two took 105–106 s**, close to the cap. Typical
  replies were 6,600–15,600 output tokens (thinking included).
- A larger document set, or the 48000 retry, would routinely cross 120 s. The
  call would then fail with a client timeout, even though dossio's own
  deadline was 15–30 minutes away.
- Because of this, dossio can't use Gemini Pro (or an OpenAI model) as the
  model or fallback for those long calls, though it's otherwise a workable
  option. Today they run on Claude.

## Requested change

Let the **caller's context** bound the request, as the Anthropic path already
does, and keep a sane default only for callers that pass no deadline.

1. Build the Google and OpenAI HTTP clients **without** `http.Client.Timeout`.
   Keep a transport-level guard against a dead connection instead, e.g.
   `Transport.ResponseHeaderTimeout` (time to first response byte) plus the
   usual dial/TLS timeouts. Use the same values for both providers.
2. In `send` (google.go:429, openai.go:~320): if `ctx` has **no deadline**,
   wrap it in a default timeout so a deadline-less caller can't hang forever.
   Suggested default: the same scaling as the Anthropic SDK's own estimate,
   `max(120s, 1h × maxTokens / 128000)`, so small requests behave as today
   and long ones get room. If `ctx` **has** a deadline, use it unchanged.
3. Optional: a `Config` field (for example `RequestTimeout time.Duration`) to
   override that default for deadline-less callers. Zero means the scaled
   default.

Behavior that must stay the same:

- A caller-cancelled context still aborts the request at once, and the error
  stays recognisable as a context error (`errors.Is(err, context.Canceled)` /
  `context.DeadlineExceeded`). dossio's retry logic classifies timeouts this way.
- No change to the request/response shapes, token usage reporting,
  `ErrSchemaViolation`, or `ErrRefused`.

## Tests to add (in aikit)

Use an `httptest.Server` that waits before replying:

- **Caller deadline wins over the old cap:** the server replies after 150 ms; the
  provider's default/header timeout is set tiny for the test; `ctx` has a 1 s
  deadline → the call **succeeds**. With v0.6.0's fixed client timeout
  (scaled down for the test) it would fail.
- **Caller deadline still bounds:** the server waits 2 s, `ctx` has a 200 ms
  deadline → the call fails quickly with an error satisfying
  `errors.Is(err, context.DeadlineExceeded)`.
- **No deadline → default applies:** the server never replies; no `ctx` deadline;
  the test-overridden default (e.g. 300 ms) ends the call with a timeout error
  rather than hanging.
- **Cancel:** `ctx` cancelled mid-request → `errors.Is(err, context.Canceled)`.
- The same cases for both the Google and the OpenAI provider (and an
  OpenAI-compatible `OpenAIBaseURL`).

## Release notes to include

- "Google and OpenAI requests are now bounded by the caller's context deadline
  instead of a fixed 120 s client timeout. Requests without a deadline get a
  default that scales with `MaxTokens` (at least 120 s)."
- Call out that a caller who relied on the old 120 s cutoff should now pass a
  context deadline.

## What dossio will do once it's released

- Bump to the new aikit version, then probe Gemini Pro again on the long
  risk-review calls (report + cross-document list, 6+ tries each) to confirm
  replies past 120 s complete.
- Only after that, consider Gemini Pro as a fallback for those calls. That's
  a quality decision for the dossio owner: on 2026-10-08, Opus was clearly
  better on content and Gemini Pro more reliable and about 40% cheaper.
- dossio already wraps each long call in its own deadline
  (`withReportCallDeadline`), so nothing else changes on dossio's side.
