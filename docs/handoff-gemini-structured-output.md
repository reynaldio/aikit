# Handoff: Gemini structured output (JSONSchema) in aikit

**From:** a dossio session, 2026-10-05.
**Goal:** make `llm.Request.JSONSchema` work on the Google (Gemini) provider, so that
dossio can send schema-based calls to Gemini Flash. The first of these calls is
document-field extraction on upload.
**Release target:** `v0.5.0`. dossio pins `v0.4.0`.

## Why

dossio wants to move its document-reading-on-upload calls from Claude Opus to
**Gemini Flash**. Gemini Flash matches Opus on LegalBench, is roughly 3–10× cheaper,
reads PDFs natively, and handles Indonesian well. dossio would keep a Claude
fallback. Every one of these calls sets `JSONSchema`.

Today only `anthropic.go` honours `JSONSchema` (`anthropic.go:110`,
`output_config.format`). `google.go` never reads it: `geminiRequest.GenerationConfig`
has only `MaxOutputTokens` and `ThinkingConfig`. So Gemini returns free text, often
wrapped in a ```` ```json ```` code fence or surrounded by prose. dossio then calls
`json.Unmarshal(resp.Text)` and silently gets an empty result.

The README's current advice is "pin the model if you need the schema guarantee".
That means Gemini can't be used for any structured call, and most of dossio's AI
calls are structured.

## Scope

### 1. Gemini: send the schema (required)

In `llm/google.go`:
- Add to `GenerationConfig`:
  - `responseMimeType` (`"application/json"` when `req.JSONSchema` is set);
  - the schema field.
- **Check the current Gemini API reference before choosing the schema field.**
  - `responseJsonSchema` takes standard JSON Schema. It's the newer field, and it's
    what dossio's schemas are written in.
  - `responseSchema` takes an older OpenAPI-subset `Schema` object, which has
    different keyword support.
  - Prefer `responseJsonSchema` if the current Flash/Pro models support it.
    Otherwise convert to the older form.
- Schemas dossio sends look like this (`pkg/llmschema` in dossio adds
  `additionalProperties:false` recursively):
  ```json
  {"type":"object","additionalProperties":false,"properties":{
    "docType":{"type":"string","enum":["NATIONAL_ID","...","OTHER"]},
    "docNumber":{"type":"string","description":"..."},
    "docDate":{"type":"string","description":"YYYY-MM-DD"}}}
  ```
  Check that `enum`, `description`, nested objects and arrays, and
  `additionalProperties:false` are all accepted. If one is rejected, strip or
  translate it inside the provider. Don't make callers change their schemas.

### 2. Combinations the API may reject (verify each one, then handle it)

| Combination | What to check | Handling if unsupported |
|---|---|---|
| `JSONSchema` + `Tools` (function calling) | Can JSON mode be combined with function declarations on current models? | Send tools only and drop JSON mode for that round; schema calls in dossio don't use tools. Document the behaviour either way. |
| `JSONSchema` + `WebSearch` (`google_search` grounding) | Is grounding allowed with JSON mode? | Return a clear error rather than silently dropping one of them. |
| `JSONSchema` + the flash thinking-suppression retry loop (`google.go` ~line 323) | The schema must survive every retry attempt. | It's set on `body` before the loop, so it already should. Add a test anyway. |
| Truncated output (`MAX_TOKENS` finish reason) with a schema | The JSON will be invalid. | Return an error. Don't return partial text as success. |

### 3. Failover must not lose the schema (required)

`fallbackRef` and `Fallbacks` can currently move a schema request from Anthropic to
a provider that ignores the schema. After this change, Google can take it. **OpenAI,
DeepSeek and Moonshot still can't.** When `req.JSONSchema` is set, failover must skip
targets whose provider can't enforce a schema. Add a small per-provider capability
check, such as `supportsJSONSchema(provider)`, rather than string checks scattered
through the code.

### 4. A safety net for providers without enforcement (recommended)

When a provider still can't enforce a schema (the OpenAI-compatible family), at least:
- remove a surrounding ```` ```json … ``` ```` code fence from `resp.Text`;
- if the result isn't valid JSON, return an error, e.g. a new `ErrSchemaViolation`
  in `errors.go`.

A caller should get an error, not silently bad text. Whether to also validate against
the schema itself is optional; checking that the JSON parses is the minimum.

### 5. OpenAI `response_format: json_schema` (optional, separate commit)

The same gap exists in `openai.go`. It's out of scope for dossio's immediate need,
because the OpenAI chat API can't take PDFs anyway. Do it only if it's cheap. Keep
DeepSeek and Moonshot, which share that client, on the safety net from item 4 unless
their APIs support it.

### 6. Prices (optional, check your uncommitted work first)

`llm/pricing.go` already has an **uncommitted local change**. Look at that diff
before editing. dossio needs these entries, which `DefaultPrices` currently lacks:
- `claude-sonnet-4-6`: dossio's current default, metered with no rate today;
- `claude-opus-5-5`: $4/$20;
- `gemini-3.8-flash`: $1.5/$7.5 per Vals, October 2026;
- `gemini-4-argon`: $4/$20.

Confirm prices against the providers' price pages, not this note.

## Leave alone

- `decide/` (untracked): separate work in progress, the TypeSafe/Jev decision
  client. Not part of this task.
- Anthropic schema behaviour, which already works.

## Tests (in `llm/llm_test.go`, using the existing httptest style)

- With a schema, the Gemini request body carries `responseMimeType` and the schema
  field. Without one, neither field is sent.
- The schema is still present on each flash retry attempt.
- Failover from Anthropic with a schema: Google is an eligible target; an
  OpenAI-family target is skipped.
- A `MAX_TOKENS` finish with a schema returns an error.
- The safety net: fenced JSON is unwrapped; non-JSON text gives
  `ErrSchemaViolation`.
- One opt-in live smoke test against the real Gemini API (skipped without
  `GOOGLE_API_KEY`), using a dossio-shaped schema with an enum and a PDF part.

## Done when

- Tests pass, the README's structured-output section reflects which providers
  enforce schemas, and the in-code `JSONSchema` doc comment in `llm.go` is updated.
- `v0.5.0` is tagged. Release notes call out that Gemini now honours `JSONSchema`
  and that failover now skips providers that can't enforce a schema.
- Commit and tag only when the user asks.

## dossio follow-up (for reference only; done in the dossio repo, not here)

1. Bump to aikit `v0.5.0`.
2. Move `ExtractDocumentFields` and `ExtractContractFields` from `aillm.TaskReview`
   to `llm.TaskExtract`. Today they're commented "cheap routing" but run on Opus.
3. In `pkg/aillm/config.go`, point `ProfileFast` at Gemini Flash, with
   `Fallbacks[ProfileFast]` set to Claude Sonnet 5.
4. Make `parseExtractedDocFields` and similar parsers return an error on invalid
   JSON instead of `_ = json.Unmarshal`.
5. Remove the stale "Call record_document_fields" wording from `extractUserText`.
6. Before switching, use `CompareTargets` to compare Opus, Gemini Flash and Sonnet 5
   on about 30 real Indonesian documents (KTP, NPWP, akta, contracts).
7. Data-protection paperwork (UU PDP): add Google as a sub-processor and use the
   paid Gemini API tier. aikit calls `generativelanguage.googleapis.com`, not Vertex,
   so there is no region guarantee.
