# Batch API for `aikit/llm` (`llm.Batcher`) — design

Date: 2026-10-14. Status: approved in brainstorming, waiting for spec review.

Fourth of four sub-projects: retries (merged) → helpers (merged, v0.10.0) → Embed (merged, 0346273, not
tagged) → **Batch**. Each one gets its own spec, plan and build.

## What a batch is

You send many requests at once. The provider runs them when it has spare capacity, usually within an hour
and at most within 24 hours, and charges about half price. Good for background work: re-summarizing,
classifying a backlog, embedding a whole library. Not for anything a person is waiting on.

The batch lives at the provider. The app gets an ID back, saves it, and later checks the status and reads
the results, even after a restart.

## Decisions (owner answers)

| Question | Decision |
| --- | --- |
| Tag v0.11.0 before starting | No. Embed and Batch ship together under one tag later |
| Embeddings in batch | Yes, chat and embeddings |
| How the app uses it | Handle-based: `Submit` returns a `BatchID`; `Status`, `Results`, `Cancel` take it; plus a `Wait` helper. aikit keeps no state and runs no goroutines |
| Model choice | One model per batch, from `Task` (via Profiles) or an explicit `Model`. No failover |
| Size | Full provider sizes. Over a limit → `ErrBatchTooLarge` before sending; the app splits. One `BatchID` is always one provider job |
| Results | An iterator (`iter.Seq2`), read as a stream; `CollectResults` for small jobs |
| Metering | One `Usage` per item as results are read, at the batch price; `Usage` gets `Batch`, `BatchID`, `CustomID` so the app can upsert |
| Guard | `guard.WrapBatcher` gives every chat item the background treatment (DataRule + scan) |
| Shape | One `llm.Batcher` interface for chat and embed. `Client` and `Embedder` do not change |
| Providers | Anthropic (chat), OpenAI (chat, embed), Gemini (chat, embed), Moonshot (chat). DeepSeek and a custom `OpenAIBaseURL` → `ErrBatchUnsupported` |
| Parked follow-ups | All four offered are in (see "Small follow-ups") |

## Public API

New file `llm/batch.go`.

```go
type Batcher interface {
    SubmitChat(ctx context.Context, b ChatBatch) (BatchID, error)
    SubmitEmbed(ctx context.Context, b EmbedBatch) (BatchID, error)
    Status(ctx context.Context, id BatchID) (BatchStatus, error)
    Results(ctx context.Context, id BatchID) iter.Seq2[BatchResult, error]
    Cancel(ctx context.Context, id BatchID) error
    // Enabled reports whether at least one provider that can batch has a key.
    Enabled() bool
}

func NewBatcher(cfg Config) Batcher

type ChatBatch struct {
    Task  Task      // resolved through Profiles, like Complete
    Model *ModelRef // optional; overrides Task
    Items []ChatItem
}

type ChatItem struct {
    CustomID string
    Request  Request
}

type EmbedBatch struct {
    Model      *ModelRef // optional; default Config.EmbedModel
    Kind       EmbedKind
    Dimensions int       // 0 = the model's default
    Items      []EmbedItem
}

type EmbedItem struct {
    CustomID string
    Text     string // one text per item, one vector back
}

// BatchID is opaque. Save it as is.
// Format: "{provider}:{kind}:{url.QueryEscape(model)}:{provider's own id}", kind is "chat", "chatjson" or "embed".
// "chatjson" = every item had a JSONSchema, so Results runs the JSON check (aikit keeps no requests).
type BatchID string

func (id BatchID) Provider() Provider // "" if the ID is malformed
func (id BatchID) Model() string      // "" if the ID is malformed

type BatchState string

const (
    BatchRunning BatchState = "running" // queued, validating, running, finalizing or canceling
    BatchDone    BatchState = "done"    // ended; Results can be read (also after cancel or expiry)
    BatchFailed  BatchState = "failed"  // the whole batch was rejected; no results
)

type BatchStatus struct {
    State     BatchState
    Total     int
    Succeeded int
    Failed    int
    Canceled  int
    Expired   int
    Err       error // set when State is BatchFailed: why
}

type BatchResult struct {
    CustomID string
    Response Response  // chat: as from Complete. embed: Provider, Model, InputTokens only
    Vector   []float32 // embed only; unit length
    Err      error     // this item's error; nil on success
}

// Wait polls Status every `every` (<= 0 → 30s) until the state is BatchDone or
// BatchFailed, or ctx ends. It returns the last status.
func Wait(ctx context.Context, b Batcher, id BatchID, every time.Duration) (BatchStatus, error)

// CollectResults reads the whole iterator into a slice. It stops at the first
// iterator error and returns the results read so far with it.
func CollectResults(seq iter.Seq2[BatchResult, error]) ([]BatchResult, error)
```

New errors in `llm/errors.go`:

| Error | When |
| --- | --- |
| `ErrBatchUnsupported` | the provider cannot batch this kind (DeepSeek, custom `OpenAIBaseURL`, Anthropic embed, Moonshot embed) |
| `ErrBatchNotDone` | `Results` called before `BatchDone` |
| `ErrBatchTooLarge` | over the provider's count or byte limit |
| `ErrBadBatchID` | the ID cannot be parsed |
| `ErrBatchItemExpired` | `BatchResult.Err`: the item did not run before the window ended |
| `ErrBatchItemCanceled` | `BatchResult.Err`: the item did not run because the batch was canceled |

No new `Config` fields. `NewBatcher` reads the key fields, `OpenAIBaseURL`, `MoonshotBaseURL`, `Profiles`,
`EmbedModel`, `RequestTimeout` and `Logger`.

### `NewBatcher` and `Enabled`

- Builds a batch backend for each of Anthropic, OpenAI (only when `OpenAIBaseURL` is empty), Google and
  Moonshot that has a key.
- No backend built → a noop batcher: `Enabled()` false, every call returns `ErrNotConfigured`
  (`Results` yields one `ErrNotConfigured`).

### Submit rule order

1. Pick the model.
   - Chat: `b.Model` if set and not empty, else `Task` → its profile model, else the default profile
     (the same resolution `Complete` uses, without failover or `usable()` swaps).
   - Embed: `b.Model` if set and not empty, else `Config.EmbedModel`. Empty → `ErrNotConfigured`.
2. The provider cannot batch this kind → `ErrBatchUnsupported`.
3. The provider has no backend (no key) → `ErrNotConfigured`.
4. Check items. Nothing is sent if any check fails; the error names the item's index.
   - `len(Items) == 0` → error.
   - `CustomID` must match `^[A-Za-z0-9_-]{1,64}$` and be unique in the batch.
   - Chat: item `Request.Model` must be nil or equal to the batch model. Item `Request.Task` must be empty
     or equal to `b.Task`. `Request.OnEvent` set → error. `ToolEvents`, `NoFallback` and `Tier` are
     ignored. Each item uses its own `Request.MaxTokens`, else `Config.MaxTokens`, like `Complete`.
   - Chat: either every item has a `JSONSchema` (kind `chatjson`) or none does (kind `chat`). A mix → error;
     the app sends two batches. Results needs this to know whether to run the JSON check.
   - Embed: `Text` must not be `""`; `Dimensions >= 0`; `Kind` is `""`, `EmbedQuery` or `EmbedDocument`.
5. Build every line with the live request builder for that provider. A build error (for example Gemini
   before v3 with JSONSchema + WebSearch) → error naming the item.
6. Check count and total bytes against the provider's limits → `ErrBatchTooLarge`.
7. Upload the input file when the provider needs one.
8. Create the batch. Return the `BatchID`.

### Money rule: never create a batch twice

The create call (Anthropic `POST /v1/messages/batches`, OpenAI/Moonshot `POST /batches`, Gemini
`batchGenerateContent` / `asyncBatchEmbedContent`) is **never retried by aikit**, on any error. A retry after
a lost reply could start a second batch and bill twice. When that call fails, the error text says the batch
may still have been created.

All other calls (file upload, status, results download, cancel) use the normal transport retry policy.
A duplicate upload creates an unused file, which costs nothing.

### Results rules

- `Results` first reads the status. Not `BatchDone` → the iterator yields one `ErrBatchNotDone` and stops.
  `BatchFailed` → yields one error wrapping `Status.Err`.
- Results come in the provider's order, not input order. Match them on `CustomID`.
- aikit returns only what the provider sends back. It does not know the input list (no state), so a missing
  item is the app's to notice.
- The iterator streams the provider's result file. Stopping early (`break`) closes the download.
- An iterator error (`yield(BatchResult{}, err)`) means reading broke: network, decode of a whole line.
  It is the last thing yielded. A per-item problem is `BatchResult.Err`, and reading goes on.
- A chat item that succeeded goes through the same reply steps as `Complete`: the live response parser
  (tokens, tool calls, stop reason, refusal check), then `checkJSONReply` when the batch kind is `chatjson`. So `Err` can be `ErrRefused` or `ErrSchemaViolation` with `Response` still filled
  (tokens are billed).
- An item the provider failed (bad params, a 4xx line, a Gemini item `error`) → `Err` wraps the
  provider's message; `Response` has `Provider` and `Model` and 0 tokens.
- Expired item → `ErrBatchItemExpired`. Canceled item → `ErrBatchItemCanceled`. Both 0 tokens.
- Embed item → `Vector` normalized to unit length (a zero vector stays zero). An empty vector → `Err`.

### Timeouts

- `Status` and `Cancel`: if ctx has no deadline, `Config.RequestTimeout`, or 60s when that is 0.
- `Submit` and `Results`: the ctx only. A 2 GB upload or a large result file can take a long time.

## Inside

### Files

| File | Responsible for |
| --- | --- |
| `llm/batch.go` | Public types, `NewBatcher`, the batcher (rule order, ID format, limits check), noop batcher, `Wait`, `CollectResults` |
| `llm/batch_anthropic.go` | Anthropic backend, through the SDK's `Messages.Batches` |
| `llm/batch_openai.go` | OpenAI and Moonshot backend: JSONL build, multipart file upload, batches endpoints, output/error files |
| `llm/batch_google.go` | Gemini backend: JSONL build, resumable upload, batch create/status/cancel, responses file |
| `llm/errors.go` | The six new errors |
| `llm/usage.go` | `WithBatchUsage`; new `Usage` fields |
| `llm/pricing.go` | Batch price factor per provider |
| `guard/batch.go` | `WrapBatcher` |
| `llm/llmtest/batcher.go` | Fake `Batcher` |
| `README.md` | "Batch" section |

The chat files (`anthropic.go`, `openai.go`, `google.go`) and embed files get small refactors only where
the batch code needs to call a request builder or reply parser that is now inline. Live behavior must not
change; their existing tests prove it.

### Backend hook (internal)

```go
type batchBackend interface {
    // kinds this backend can batch: "chat", "embed"
    supports(kind string) bool
    limits(kind string) batchLimits // maxItems, maxBytes
    // buildChat / buildEmbed turn one item into one JSONL line (or SDK param) using the live builders.
    submit(ctx context.Context, kind, model string, lines []batchLine) (providerID string, err error)
    status(ctx context.Context, providerID string) (BatchStatus, error)
    results(ctx context.Context, kind, model, providerID string) iter.Seq2[rawBatchResult, error]
    cancel(ctx context.Context, providerID string) error
}
```

The plan fixes the exact names. The rule: the core does the rule order, ID format and reply steps, and the
backends know only their provider's wire format.

### Anthropic

- Create with `Messages.Batches.New`: `requests[] {custom_id, params}`, where params is the same
  `MessageNewParams` the live path builds (system, cache control, tools, output config, effort, thinking).
  No `stream`.
- Limits: 100,000 items, 256 MB.
- Status: `processing_status` `in_progress` / `canceling` → `BatchRunning`, `ended` → `BatchDone`.
  Counts from `request_counts` (Total = the sum of all five).
- Results: the SDK's results stream. `result.type`: `succeeded` → parse `message` with the live
  `anthropicResponse`; `errored` → `Err`; `canceled` → `ErrBatchItemCanceled`; `expired` →
  `ErrBatchItemExpired`.
- Cancel: `Messages.Batches.Cancel`.
- Anthropic has no embed batch → `ErrBatchUnsupported`.
- The plan task checks that `output_config` (JSONSchema) is accepted in batches; if not, a JSONSchema item
  on Anthropic is an error at submit.

### OpenAI and Moonshot

- Line: `{"custom_id": id, "method": "POST", "url": "/v1/chat/completions" | "/v1/embeddings", "body": {...}}`.
  Chat body = the live `oaiRequest` (no `stream`, no `stream_options`). Embed body = `{model, input: text,
  encoding_format: "float", dimensions?}`.
- Upload: `POST {base}/files`, multipart, `purpose=batch`. Then `POST {base}/batches` with
  `{input_file_id, endpoint, completion_window: "24h"}`.
- Limits: OpenAI 50,000 items, 200 MB. Moonshot: the plan task reads its docs and sets them.
- Status: `validating` / `in_progress` / `finalizing` / `cancelling` → `BatchRunning`; `completed` /
  `expired` / `cancelled` → `BatchDone`; `failed` → `BatchFailed` with `errors` as `Err`. Counts from
  `request_counts` (`total`, `completed`, `failed`). Canceled and Expired counts are 0 (OpenAI does not split
  them out).
- Results: stream `GET {base}/files/{output_file_id}/content`, then the `error_file_id` file. Each line:
  `response.status_code` 200 → parse `response.body` with the live parser (chat) or the embed decoder;
  other codes or `error` set → `Err`. An expired batch's unrun items arrive in the error file with code
  `batch_expired` → `ErrBatchItemExpired`; `batch_cancelled` → `ErrBatchItemCanceled`.
- Cancel: `POST {base}/batches/{id}/cancel`.
- Moonshot: same code, Moonshot base URL and key, chat only.

### Gemini

- Always a file, never inline. One code path, and results are always a file to stream.
- Line: `{"key": id, "request": {...}}`. Chat request = the live generateContent body. Embed request = the
  live embed body for one text (taskType or prefix by `Kind`, `outputDimensionality`).
- Upload: resumable Files API (`POST {upload base}/files` with `X-Goog-Upload-Protocol: resumable`, start,
  then upload+finalize). Input files are kept 48h, used at once here.
- Create: `POST /models/{model}:batchGenerateContent` (chat) or `:asyncBatchEmbedContent` (embed) with
  `{batch: {displayName, model: "models/{model}", inputConfig: {fileName}}}`. The returned name
  `batches/{id}` is the provider ID.
- Limits: 2 GB file. No item cap known; the plan task checks.
- Status: `GET /batches/{id}`; match the state on its suffix, so both `JOB_STATE_*` and `BATCH_STATE_*`
  work: `PENDING` / `RUNNING` → `BatchRunning`; `SUCCEEDED` / `CANCELLED` / `EXPIRED` → `BatchDone`;
  `FAILED` → `BatchFailed`. Counts from `batchStats`.
- Results: download the responses file
  (`/download/v1beta/{file}:download?alt=media`). Line `{key, response | error}`. `response` → live parser.
- Cancel: `POST /batches/{id}:cancel`.
- The docs disagree on some field names (`output` vs `dest`, `responsesFile` vs `fileName`). The plan task
  decodes both and records which one a real reply used, if one can be checked.

## Metering: `WithBatchUsage`

```go
func WithBatchUsage(inner Batcher, book *PriceBook, record func(ctx context.Context, u Usage)) Batcher
```

New `Usage` fields:

```go
Batch    bool    // true for batch items
BatchID  BatchID
CustomID string
```

- `record` is called once for each item that `Results` yields with tokens (a success, `ErrRefused`,
  `ErrSchemaViolation`). Failed, expired and canceled items are not billed and not recorded.
- `Task` = the batch's `Task` for chat (empty when chosen by `Model`), `TaskEmbed` for embed. `Duration` = 0.
  `Err` = the item's error. `Estimated` = `Response.UsageEstimated`.
- `CostUSD = book.Cost(Model, counts...) × batchFactor(Provider)`. Factors: Anthropic 0.5, OpenAI 0.5,
  Google 0.5, Moonshot 0.6. The plan task checks each on the pricing pages and notes the date read
  (OpenAI embeddings at 0.5 is not stated directly).
- Reading results twice records twice. The doc says to save with an upsert on (BatchID, CustomID).
- Nil `book` uses `DefaultPrices`. Panics on nil `inner` or nil `record`. `Submit*`, `Status`, `Cancel`,
  `Enabled` pass through.

## Guard: `guard.WrapBatcher`

```go
func WrapBatcher(inner llm.Batcher, o Options) llm.Batcher
```

- `SubmitChat`: every item gets what a `ClassBackground` call gets today: `DataRule` added to its system
  message (same `withPolicies`), and the scan (audit only, hits go to `o.Audit`, as in `Wrap`). `Classify`,
  limits, the limiter and the refusal marker are not used. `o.Inner` is not used.
- `SubmitEmbed`, `Status`, `Results`, `Cancel`, `Enabled`: pass through.
- Panics on nil `inner`.

## Test fake: `llmtest.Batcher`

```go
func NewBatcher(chat *Client, emb *Embedder) *Batcher // either may be nil (that kind → error)
func (b *Batcher) SubmitChat / SubmitEmbed / Status / Results / Cancel / Enabled
func (b *Batcher) Hold()                 // later batches stay BatchRunning
func (b *Batcher) Release(id llm.BatchID) // runs a held batch; it becomes BatchDone
func (b *Batcher) Submits() []any        // ChatBatch or EmbedBatch, a copy, in call order
func (b *Batcher) Fail(err error)        // the next call returns err (once)
```

- Not held: items run through the fake `Client` / `Embedder` at submit time; the batch is `BatchDone`
  at once.
- IDs look like real ones (`llmtest:chat:fake:1`). `Cancel` on a held batch → `BatchDone`, every item
  `ErrBatchItemCanceled`.
- Same item checks as the real core (CustomID rule and uniqueness), so app bugs show up in tests.
- Safe for concurrent use. Imports only `llm`.

## README

A short "Batch" section: when to use it, submit and save the ID, `Wait` or poll `Status`, read `Results`
and match on `CustomID`, the 24h window, `WithBatchUsage` with an upsert, `guard.WrapBatcher`, and
`llmtest.NewBatcher`.

## Version

New API only, nothing breaks. Embed and Batch ship under the next tag, **v0.11.0** (owner's call when).

## Tests

All with `httptest` (Anthropic via the SDK pointed at a test server), no real API calls. Targeted runs per
task; the full `go test ./...`, `go test -race ./...` and `go vet ./...` once before merge.

- **Core:** model pick (Model, Task, default profile, EmbedModel); `ErrBatchUnsupported` for DeepSeek,
  custom OpenAIBaseURL, Anthropic embed, Moonshot embed; `ErrNotConfigured` with no key; noop batcher;
  empty items; bad and duplicate CustomID; item model/task mismatch; OnEvent set; empty embed text;
  schema mix → error; `chatjson` kind; `ErrBatchTooLarge` by count and by bytes; nothing sent on any check failure; BatchID round-trip
  (model with `:` and `/`); `ErrBadBatchID`; `Results` before done → `ErrBatchNotDone`; `BatchFailed` →
  error; break stops the download; `Wait` returns on done, failed and ctx end; `CollectResults`.
- **Create is not retried:** a 503 on create → one request seen, error says the batch may exist; a 503 on
  upload and on status is retried.
- **Anthropic:** params match the live builder (system cache, tools, effort); status and counts mapping;
  results: succeeded, errored, canceled, expired; cancel.
- **OpenAI / Moonshot:** JSONL lines for chat and embed; multipart upload with `purpose=batch`; create body;
  status mapping including `failed`; output file then error file; `batch_expired` / `batch_cancelled`;
  non-200 line → `Err`; Moonshot uses its base URL and key; embed vectors normalized.
- **Gemini:** JSONL lines for chat and embed (taskType vs prefix); resumable upload headers and the two
  steps; create body for both kinds; state suffix mapping for both prefixes; results file download;
  `error` lines; both field-name spellings.
- **Reply steps:** JSONSchema item with bad JSON → `ErrSchemaViolation` with tokens; refusal → `ErrRefused`.
- **`WithBatchUsage`:** one record per billed item; half price; Moonshot 0.6; failed/expired items not
  recorded; fields BatchID and CustomID set; nil args panic.
- **`guard.WrapBatcher`:** DataRule added to every chat item; scan hits reach `Audit`; embed untouched.
- **`llmtest.Batcher`:** done at once; Hold/Release; Cancel on held; Fail once; Submits copy; CustomID
  checks; concurrent use under `-race`.

## Small follow-ups (optional tasks in the plan)

From memory note streaming-deferred-followups; all checked against the code on 2026-10-14 and still apply.

1. **Path-escape Gemini model names** in every URL (chat, stream, embed and the new batch code), with a
   test using a name that needs escaping.
2. **Embed null vectors:** a `null` or missing `embedding` / `values` → a clear error naming the index.
3. **Embed test tidy-ups:** OpenAI error-path tests check `estimated`; a non-JSON error body case; read
   `reqs` under the mutex; Gemini mismatch test adds the no-usage case; G7 asserts retries; deeper
   WithEmbedUsage U1/U2 reply checks; OpenAI price comment names its source page; C8/C10 extra asserts.
4. **OpenAI stream tests:** a test with mixed index and no-index pieces; make T2 fail on the old code.

Still parked (only fix if seen in practice): Gemini leading empty text part on function-call-only streams;
guard tracker counting held-back text as sent; `stream_options` 400 on some compatible backends; Anthropic
`X-Stainless-Retry-Count`.

## Out of scope

- Listing or deleting batches at the provider (the app stores its own IDs).
- Splitting an oversize batch into several provider jobs.
- Mixed models in one batch.
- Background polling or callbacks inside aikit.
- Batch on DeepSeek or custom OpenAI-compatible servers.
- A balance API.
