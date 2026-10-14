# Embeddings for `aikit/llm` (`llm.Embedder`) — design

Date: 2026-10-14. Status: approved in brainstorming, waiting for spec review.

Third of four sub-projects, in this order: retries (merged) → helpers (merged, v0.10.0) → **Embed** →
Batch. Each one gets its own spec, plan and build.

## What an embedding is

You send text and get back a list of numbers (a vector, often 768–3072 long) that captures its meaning.
Texts with similar meaning get vectors that are close. Apps use this for search by meaning (RAG: embed
stored documents once, embed each question, find the closest documents), for "find similar", and for
grouping. aikit only turns text into vectors. Storing and searching them (pgvector, etc.) is the app's job.

Vectors from different models cannot be compared. Switching models means re-embedding everything stored.

## Decisions (owner answers)

| Question | Decision |
| --- | --- |
| First use case | Not known yet: support both search (query/document hint) and plain similarity |
| Where `Embed` lives | Its own interface `llm.Embedder`, built by `llm.NewEmbedder(cfg)`. `llm.Client` does not change, so `guard.Wrap`, `WithUsage`, `llmtest.Client` and app fakes keep compiling |
| How the model is chosen | One configured model, `Config.EmbedModel`, plus an optional per-request `Model` override |
| Failover | None. Mixing two models' vectors would spoil the app's stored data. Transport retries still apply |
| Backends | OpenAI, OpenAI-compatible (`OpenAIBaseURL`), Gemini. Anthropic, DeepSeek and Moonshot have no embedding endpoint: `ErrEmbedUnsupported` |
| Large batches | aikit splits into provider-sized chunks, sent one after another. Any chunk fails → error, no vectors, but tokens already billed are reported |
| Guard | Not applied. Embeds have no reply text to scan |
| Parked follow-ups | All four groups offered are in (see "Small follow-ups") |

## Public API

New file `llm/embed.go`.

```go
type EmbedKind string

const (
    EmbedQuery    EmbedKind = "query"    // a search question
    EmbedDocument EmbedKind = "document" // text being stored for search
)
// "" means no hint (similarity, grouping).

type EmbedRequest struct {
    Texts      []string
    Kind       EmbedKind // optional hint
    Dimensions int       // 0 = the model's default
    Model      *ModelRef // optional; overrides Config.EmbedModel
}

type EmbedResponse struct {
    Vectors        [][]float32 // one per text, same order as Texts, unit length
    Provider       Provider
    Model          string
    InputTokens    int
    UsageEstimated bool // at least one chunk had no token count from the backend, so aikit guessed
}

type Embedder interface {
    Embed(ctx context.Context, req EmbedRequest) (EmbedResponse, error)
    // Enabled reports whether Config.EmbedModel can be served.
    Enabled() bool
}

func NewEmbedder(cfg Config) Embedder

var ErrEmbedUnsupported = errors.New("llm: provider has no embedding endpoint") // in errors.go
```

New `Config` field: `EmbedModel ModelRef`, e.g. `{ProviderOpenAI, "text-embedding-3-small"}` or
`{ProviderGoogle, "gemini-embedding-001"}`. `NewEmbedder` reuses `OpenAIAPIKey`, `OpenAIBaseURL`,
`GoogleAPIKey`, `RequestTimeout` and `Logger`. It ignores the chat fields.

### `NewEmbedder` and `Enabled`

- `NewEmbedder` builds an embed backend for each of OpenAI and Google that has a key.
- If no backend was built, or `EmbedModel` is empty, it returns a noop embedder: `Enabled()` false, every
  `Embed` returns `ErrNotConfigured`.
- Otherwise `Enabled()` is true when `EmbedModel.Provider` has a built backend.

### Rule order inside `Embed`

1. Pick the model: `req.Model` if set and not empty, else `Config.EmbedModel`.
2. Provider is Anthropic, DeepSeek or Moonshot → `ErrEmbedUnsupported`.
3. Provider is OpenAI or Google with no key → `ErrNotConfigured`.
4. `len(req.Texts) == 0` → empty `EmbedResponse` (Provider/Model set, `Vectors` empty, not nil), nil error,
   no call.
5. Any text is `""` → error naming its index, no call. `Dimensions < 0` → error. `Kind` not one of
   `""`, `EmbedQuery`, `EmbedDocument` → error.
6. Split into chunks (see below) and send them in order.
7. Each chunk: the backend must return exactly one vector per text, else an error.
8. A chunk fails → return `EmbedResponse{Provider, Model, InputTokens: tokens of the chunks that worked
   plus any tokens the failed chunk was billed (a wrong-count reply still reports usage), UsageEstimated}`
   with no vectors, plus the error.
9. All chunks worked → normalize every vector to unit length (a zero vector stays zero, never NaN) and
   return them in input order.

Normalizing always is simple and safe: OpenAI's vectors are already unit length, gemini-embedding-2 is
unit length, and gemini-embedding-001 is not when `Dimensions` is set below its default. After
normalizing, cosine similarity and dot product give the same answer.

### The `Kind` hint per backend

- **OpenAI and compatible:** ignored.
- **Gemini, models named `gemini-embedding-001` or starting with `text-embedding-`:** sent as `taskType`:
  `EmbedQuery` → `RETRIEVAL_QUERY`, `EmbedDocument` → `RETRIEVAL_DOCUMENT`, `""` → no taskType.
- **Any other Gemini model (gemini-embedding-2 and later):** these take no taskType and want the task
  written into the text, so aikit adds a prefix:
  `EmbedQuery` → `task: search result | query: {text}`,
  `EmbedDocument` → `title: none | text: {text}`, `""` → text sent as is.

Documents and queries must use the same model and both carry a Kind for search to work well. The README
says so.

## Inside

### Files

| File | Responsible for |
| --- | --- |
| `llm/embed.go` | Public types, `NewEmbedder`, the embedder (rule order, chunking, normalizing), noop embedder |
| `llm/embed_openai.go` | OpenAI/compatible backend: request, response, usage |
| `llm/embed_google.go` | Gemini backend: request, response, usage, Kind mapping |
| `llm/errors.go` | `ErrEmbedUnsupported` |
| `llm/llm.go` | `Config.EmbedModel` field |
| `llm/usage.go` | `WithEmbedUsage`, `TaskEmbed` |
| `llm/pricing.go` | Embedding model prices |
| `llm/llmtest/embedder.go` | Fake `Embedder` |
| `README.md` | "Embeddings" section |

### Backend hook (internal)

```go
type embedBackend interface {
    embed(ctx context.Context, model string, texts []string, kind EmbedKind, dims int) (embedChunk, error)
    maxTexts() int  // texts per chunk
    maxTokens() int // guessed tokens per chunk; 0 = no token cap
}

type embedChunk struct {
    vectors   [][]float32
    tokens    int
    estimated bool
}
```

The embedder never knows provider details. Both backends use `newProviderHTTPClient`, so they get the same
retry policy as chat (2 retries on 429/5xx/network, max 10s wait).

### OpenAI and compatible

- `POST {baseURL}/embeddings`, header `Authorization: Bearer {key}`.
- Body: `{"model": m, "input": [texts...], "encoding_format": "float"}`, plus `"dimensions": n` only when
  `n > 0`. A backend that rejects `dimensions` returns its 400 as the error.
- Response: `data[].embedding` and `data[].index`. Vectors are placed by `index`, not by array order. An
  index out of range or repeated → error.
- Tokens: `usage.prompt_tokens`. If `usage` is missing or 0 (Ollama and others), guess with
  `estimateTokens` on the joined texts and set `estimated`.
- Chunks: at most 2,048 texts and at most 250,000 guessed tokens (the real limit is 300,000; the gap
  covers the guess being off). A single text over the cap gets its own chunk; the provider decides.

### Gemini

- `POST {baseURL}/models/{model}:batchEmbedContents`, API key header as in the chat code.
- Body: `{"requests": [{"model": "models/{model}", "content": {"parts": [{"text": t}]},
  "taskType"?: ..., "outputDimensionality"?: n}, ...]}`. Field names are camelCase. The plan task checks
  them against the live API reference before writing the code.
- Response: `embeddings[].values`, in request order. Tokens: `usageMetadata.promptTokenCount`, else a
  guess on the texts actually sent (with prefixes), and set `estimated`.
- Chunks: at most 100 texts, no token cap.

### Timeout

If `ctx` has no deadline, each chunk gets `Config.RequestTimeout`, or 60s when that is 0. The chat formula
(scaled by MaxTokens) does not fit embeds. A ctx deadline always wins.

## Metering: `WithEmbedUsage`

```go
const TaskEmbed Task = "embed" // a metering label only; not in the chat routing map

func WithEmbedUsage(inner Embedder, book *PriceBook, record func(ctx context.Context, u Usage)) Embedder
```

- Reuses `Usage`, so apps keep one ledger. `Task` = `TaskEmbed`, `InputTokens` from the response, the other
  counts 0, `CostUSD = book.Cost(Model, InputTokens, 0, 0, 0)`, `Estimated` = `UsageEstimated`,
  `Duration` = time in `Embed`, `Err` = the error.
- `record` is called once per `Embed` call with `InputTokens > 0`, including a partial failure. A call with
  0 tokens (not configured, bad input, first chunk failed) is not reported.
- Nil `book` uses `DefaultPrices`. Panics on nil `inner` or nil `record`, like `WithUsage`.
- `Enabled()` passes through.

## Prices

Add to `DefaultPrices`, input rate only (USD per 1M tokens):

| Model | Input |
| --- | --- |
| `text-embedding-3-small` | 0.02 |
| `text-embedding-3-large` | 0.13 |
| `gemini-embedding-001` | 0.15 |
| `gemini-embedding-2` | check the Gemini pricing page; leave out if it cannot be confirmed (meters at 0) |

The plan task checks all four against the pricing pages and notes the date read.

## Test fake: `llmtest.Embedder`

```go
func NewEmbedder(dims int) *Embedder           // dims <= 0 → 8
func (e *Embedder) Embed(ctx, llm.EmbedRequest) (llm.EmbedResponse, error)
func (e *Embedder) Enabled() bool               // true
func (e *Embedder) Requests() []llm.EmbedRequest // a copy, in call order
func (e *Embedder) Fail(err error)              // the next Embed call returns err (once)
```

- Each vector is made from a hash of the text (with Kind), so the same input always gives the same vector.
  Vectors are unit length. `req.Dimensions > 0` overrides `dims`.
- Provider `"llmtest"`, Model `"fake"`, InputTokens about 1 per 4 characters.
- A cancelled ctx returns `ctx.Err()`. Safe for concurrent use. Imports only `llm`.

## README

A short "Embeddings" section: config (`EmbedModel`), one search example (store with `EmbedDocument`, ask
with `EmbedQuery`), the "do not mix models; store Model with each vector" warning, `WithEmbedUsage`, and
`llmtest.NewEmbedder`.

## Version

New API only, nothing breaks → tag **v0.11.0**.

## Tests

All with `httptest`, no real API calls. Targeted runs per task; the full `go test ./...` and `go vet ./...`
once before merge.

- **OpenAI:** body shape (`dimensions` only when > 0, `encoding_format` float); vectors placed by shuffled
  `index`; bad or repeated index → error; missing `usage` → guessed tokens and `UsageEstimated`; 400 passed
  through; wrong vector count → error; custom `OpenAIBaseURL` is used.
- **Gemini:** body shape; 001 gets taskType for Query/Document and none for `""`; embedding-2 gets the
  prefixes and no taskType; `outputDimensionality` only when > 0; `usageMetadata` read; missing → guess.
- **Chunking:** 2,049 texts → 2 OpenAI calls; 250 texts → 3 Gemini calls; the token cap splits large
  texts; order kept across chunks; chunk 2 fails → error, no vectors, chunk 1's tokens reported.
- **Core:** normalizing (unit length; zero vector stays zero); empty `Texts` → no call; `""` text, negative
  `Dimensions`, unknown `Kind` → errors, no call; Anthropic/DeepSeek/Moonshot → `ErrEmbedUnsupported`;
  OpenAI model with no OpenAI key → `ErrNotConfigured`; empty `EmbedModel` or no keys → noop,
  `Enabled()` false; `Model` override used; default 60s timeout when ctx has none; a 503 then 200 is
  retried.
- **`WithEmbedUsage`:** success; partial failure records chunk-1 tokens with `Err`; 0-token error not
  recorded; cost uses the book; nil args panic.
- **`llmtest.Embedder`:** same text → same vector; unit length; `Dimensions` honored; `Fail` works once;
  requests recorded; cancelled ctx.

## Small follow-ups (optional tasks in the plan)

From memory notes helpers-deferred-followups and streaming-deferred-followups; all checked on 2026-10-14
and still apply.

1. **Small doc/estimate fixes.** `WithUsage` "known gap" doc also names the case where the primary breaks
   with tokens and the fallback fails with tokens (the fallback's tokens are dropped).
   `estimateRequestTokens` skips a nil `ToolDef.Schema` instead of counting `null`.
2. **Gemini thinking partial.** On a stopped stream with usage but `CandidatesTokenCount == 0`, output =
   `max(guess, ThoughtsTokenCount)`, plus a test.
3. **OpenAI stream tidy.** One helper for the sorted call-index walk (3 copies in `completeStream`); do not
   overwrite a call's non-empty ID once its start event is sent; a tool piece with no `index` but a new
   `id` starts a new call instead of collapsing into 0 (a piece with neither keeps appending to the last
   call); a tool piece after `finish_reason` still gets its ready. Tests
   for each.
4. **Test gaps and tidy-ups.** OpenAI decode/read-error partial paths; InputTokens checks in
   `TestOpenAIStreamErrorEvent` and Gemini `AfterEarlierEventThenError`; Gemini `ZeroBeforeAnyEvent`
   reaching the `seen` gate; RunTools #22 checks more than `len(Rounds)`; a test that Tools/OnEvent pass
   through each round; a `CompareTargets` pass-through test that can fail; `mustPanic` checks its message;
   router test "primary partial, then fallback fails → primary's stamped reply"; merge `holdServer` into
   `quietServer`; the 200ms Gemini timeout; rename test-local `seen`; retry router tests use a Google model
   name without the thinking-config 400 retry.

Still parked (only fix if seen in practice): guard tracker counting held-back text as sent;
`stream_options` 400 on some compatible backends; Anthropic `X-Stainless-Retry-Count`.

## Out of scope

- Failover between embed models.
- Image, PDF or audio embeddings (gemini-embedding-2 supports them; text only for now).
- Storing or searching vectors.
- The Batch API (next sub-project; about 50% off for big jobs).
- A balance API.
