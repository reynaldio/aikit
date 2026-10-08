# Handoff: a `guard` package for aikit (prompt-injection fencing, scanning, limits)

**From:** a dossio session, 2026-10-08.
**Goal:** move the general-purpose half of dossio's AI guardrails (`pkg/aiguard`)
into aikit as a new package, **`aikit/guard`**. Every product built on aikit
(dossio, nathanai, portunar, …) then gets the same protection by wrapping its
`llm.Client` once, without copying code.
**Release target:** the next minor. Ship it as `v0.7.0` together with
`docs/handoff-provider-timeouts.md`, or as `v0.8.0` if that change ships first.
dossio pins `v0.6.0`.

## Why

dossio's guard has been in production code since 2026-10-05 and is fully
tested. About half of it has nothing to do with dossio:

- **fencing:** wrapping outside text (uploaded files, tool results, memory,
  context) so the model treats it as data, never as instructions;
- **injection scanning:** spotting text that tries to instruct the model, in
  English and Indonesian;
- **turn tracking:** one user message that runs several tool rounds counts once;
- **a wrapper around every AI call** that adds the data rule, applies size caps
  and per-user rate limits, audits suspected injections, and strips a refusal
  marker.

Other aikit users want the same protection; today they'd have to copy it.

## What moves and what stays

| dossio file | Moves to `aikit/guard`? | Notes |
|---|---|---|
| `fence.go` | **Yes**, as is | No dossio dependencies |
| `scan.go` | **Yes**, as is | No dossio dependencies; keep the Indonesian patterns |
| `turn.go` | **Yes**, as is | No dossio dependencies |
| `class.go` | **Yes**, as is | The class enum |
| `policy.go` | **Partly** | `DataRule` and the marker mechanics move; dossio's `TopicPolicy` wording stays in dossio and is passed in as an option |
| `client.go` | **Yes, reworked** | Four dossio imports become options (below) |
| `citation.go` | **No** | Checks Indonesian legal citations against dossio's own law library |

The source to port is in the dossio repo at `code/apps/server/pkg/aiguard/`
(branch `rey-dev`); its tests are `client_test.go` and `core_test.go` there.

## Proposed API

```go
package guard

// Classes — how a call is guarded (unchanged from dossio).
type Class int
const (
    ClassOneShot        Class = iota // a user action, no free conversation: data rule; per-call limit
    ClassConversational              // a person typed free text: data rule + topic policy; one count per turn; refusal marker stripped
    ClassBackground                  // system work: data rule only; never rate limited or size-capped
)

// Fencing (unchanged).
const (SourceFile, SourceTool, SourceMemory, SourceContext, SourceTranscript, SourceSummary, SourceDocument = …)
type FenceAttrs struct{ Source, Ref, Name string }
func Fence(a FenceAttrs, text string) string
type Block struct{ Source, Ref, Name, Text string }
func FindBlocks(s string) []Block

// Scanning (unchanged): names of matched patterns, nil when clean. Never blocks.
func ScanInjection(text string) []string

// Turns (unchanged).
func BeginTurn(ctx context.Context) context.Context
func TurnFrom(ctx context.Context) *Turn
func (t *Turn) Suspected() bool

// Policy text.
const DataRule = `--- Untrusted content rule --- …` // the dossio text, unchanged
func StripMarker(s, marker string) (string, bool)  // marker now a parameter

// Size caps and errors — plain sentinels; each app maps them to its own errors.
var (
    ErrRequestTooLarge = errors.New("guard: request too large")
    ErrRateLimited     = errors.New("guard: rate limited")
)
type Limits struct {
    ConversationalPerHour, ConversationalPerMinute, OneShotPerHour int // ≤ 0 disables
    MaxRequestChars, MaxDocumentBytes                              int // ≤ 0 disables
    MaxMessageRunes, MaxAttachments                                int // used by CheckMessage
}
func DefaultLimits() Limits // 60/h, 10/min, 120/h, 600_000 chars, 50 MB, 20_000 runes, 10 attachments
func CheckMessage(text string, attachments int, l Limits) error // ErrRequestTooLarge or nil

// The wrapper.
type Options struct {
    Inner    llm.Client
    Classify func(ctx context.Context) Class                 // dossio: from its usage label on ctx; nil → ClassOneShot
    Identity func(ctx context.Context) (key string, ok bool) // rate-limit key (user); !ok → not rate limited
    Limits   Limits
    Limiter  Limiter                                         // nil → built-in in-process sliding windows
    TopicPolicy   string                                     // appended for ClassConversational; "" → none
    RefusalMarker string                                     // e.g. "⟦dossi:declined⟧"; "" → no marker handling
    OwnSources    []string                                   // fenced but never scanned; nil → {SourceTranscript, SourceMemory}
    Audit    AuditSink                                       // nil → no audit
    OnEvent  func(ctx context.Context, e Event)              // logging hook; nil → nothing logged
    Now      func() time.Time
}
type AuditSink interface{ InjectionSuspected(ctx context.Context, b Block) } // must never store the matched text
type Event struct{ Kind string /* "request_too_large" | "rate_limited" | "refusal" */; Class Class }
type Limiter interface {
    // CheckAndRecord atomically refuses when any window for the class is full,
    // otherwise records one use in each.
    CheckAndRecord(key string, class Class, now time.Time) bool
}
func Wrap(o Options) llm.Client // a nil or disabled Inner is returned as is
```

How dossio's four imports map onto this:

| dossio import | Becomes |
|---|---|
| `pkg/authctx` (user on ctx) | `Options.Identity` |
| `pkg/usagectx` (usage label on ctx) | `Options.Classify` |
| `pkg/ratelimit` (sliding window) | the built-in default `Limiter` (port the small in-process window), or the app's own `Limiter` |
| `pkg/apperr` (localized errors) | plain `ErrRequestTooLarge` / `ErrRateLimited`; the app maps them with `errors.Is` |
| `zerolog` + `authctx` logging | `Options.OnEvent` |

When `TopicPolicy` and `RefusalMarker` are both set, the wrapper appends the
policy and then this sentence: "When you decline, begin your reply with the
exact marker `<marker>` followed by a space." dossio's current policy text ends
with that same sentence, so dossio will drop it from its own text.

## Behavior that must be kept exactly

These are what dossio's tests pin. Port the tests with the code.

1. **Order in `Complete`:** size check (not for background) → rate limit → add
   policies → scan → call the inner client → strip the marker (conversational only).
   A refused call never reaches the inner client, so no tokens are spent.
2. **Size counting:** characters of `SystemCacheable` + every message's content
   + every tool result; bytes of documents and images = decoded base64 length.
3. **Rate limits:** background calls and calls with no identity are never
   limited. Conversational calls use the per-minute and per-hour windows,
   one-shot calls the one-shot hourly window. Check-then-record is **atomic**
   per wrapper, so concurrent calls from one user can't overshoot.
4. **Turns:** inside a `Turn`, only the first call checks and counts; its
   outcome (allowed **or refused**) is reused for every later call in the turn,
   so a refused turn stays refused.
5. **Policies:** `DataRule` is added to every call, the topic policy only to
   conversational ones. They're appended to the first system message, else to
   `SystemCacheable`, else a new system message is prepended. This always
   happens on a copy of the request.
6. **Scanning:** only well-formed fenced blocks are scanned. Sources in
   `OwnSources` are skipped (the user's own words are not a third-party
   injection). Each source is audited once per request and once per turn
   across tool rounds, and any hit marks the turn `Suspected`.
7. **Fence ids are stable:** the id is an HMAC of
   source|ref|name|text under a per-process random key. A random id per call
   would change the system prompt every turn and break prompt caching. A fence
   tag inside the text is neutralised so the text can't close its own block.
8. **Scanner false positives stay fixed:** "DAN" matches only in uppercase
   (the Indonesian "dan" means "and"); "ignore/abaikan … instructions" doesn't
   count after a negation ("shall not ignore the instructions", "jangan abaikan
   aturan"). `TestScanInjectionNegatives` holds the full list of ordinary legal
   wording that must not match.

## Tests to port (from dossio `pkg/aiguard`)

- `core_test.go`: `TestFenceRoundTripsThroughFindBlocks`,
  `TestFenceCannotBeClosedFromInside`, `TestFenceAttributesAreSanitised`,
  `TestFenceID_StableForSameContent`, `TestFenceID_DiffersForDifferentContent`,
  `TestScanInjectionPositives`, `TestScanInjectionNegatives`, `TestStripMarker`,
  `TestTurn`.
- `client_test.go`: `TestPoliciesByClass`, `TestRefusalMarkerStripped`,
  `TestInjectionAuditedOncePerSourceAndTurn`, `TestRateLimitsCountTurnsNotRounds`,
  `TestRateLimitExemptions`, `TestCheckMessage`,
  `TestOversizedRequestRefusedBeforeTheModel`, `TestTurnRefusalSticks`,
  `TestRateLimitConcurrentTurnsDoNotOvershoot`, `TestOwnContentIsNotScanned`.

Adapt them to the options (`Identity` instead of a dossio user on the context,
`Classify` instead of a usage label, `OnEvent` instead of log lines). Add one
test per new option: a custom `Limiter`, a custom `OwnSources`, and an empty
`RefusalMarker` meaning no marker handling.

## README

Add a "Guardrails (`aikit/guard`)" section covering:

- wrap the client once;
- fence every piece of outside text with `guard.Fence`;
- call `guard.BeginTurn` per user message that may run several tool rounds;
- `guard.CheckMessage` at entry points that take typed text;
- the scanner audits and never blocks: the fence and `DataRule` are the
  protection, the audit is for awareness.

## What dossio will do once it's released

- Bump aikit and replace its copies of fence/scan/turn/class and the wrapper
  with `aikit/guard`.
- Keep in dossio:
  - Dossi's `TopicPolicy` text;
  - the legal citation checker (`citation.go`);
  - the mapping from usage label to class;
  - the mapping of `guard.ErrRequestTooLarge` / `guard.ErrRateLimited` onto
    dossio's localized errors (`error.ai_request_too_large`, `RATE_LIMITED`);
  - the audit sink, which writes dossio's audit log.
- Confirm with dossio's existing guard tests and a live chat check that
  behavior is unchanged.
