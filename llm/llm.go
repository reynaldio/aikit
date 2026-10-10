// Package llm is a provider-agnostic LLM seam: a router that resolves each request to a
// (provider, model) and dispatches to Anthropic, Google Gemini, or an OpenAI-compatible
// backend, with per-profile failover and a token-cost pricing catalog.
//
// Shape:
//   - A `provider` (anthropic/google/openai) knows how to run one completion against a
//     named model; it is model-agnostic (the model is chosen per call).
//   - A `router` (the Client) maps a request's Tier — or an explicit per-request
//     ModelRef override — to a (provider, model) and dispatches. It also stamps the
//     Response with the provider+model actually used, so usage metering can attribute
//     tokens/cost per provider.
//
// This is the `llm` package of the shared aikit module
// (github.com/reynaldio/aikit); other AI-provider modalities (e.g. tts) live
// alongside it as sibling packages.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"time"
)

// Provider identifies an LLM vendor/backend.
type Provider string

const (
	ProviderAnthropic Provider = "anthropic"
	ProviderGoogle    Provider = "google"
	ProviderOpenAI    Provider = "openai"
	// DeepSeek and Moonshot (Kimi) are separate, coexisting OpenAI-compatible providers —
	// each reuses the OpenAI client pointed at its own base URL, so profiles can route
	// specific tasks to them (A/B against Claude) without disturbing OpenAI itself.
	ProviderDeepSeek Provider = "deepseek"
	ProviderMoonshot Provider = "moonshot"
)

// ModelRef names a concrete model on a provider (e.g. {google, "gemini-2.5-flash"}).
type ModelRef struct {
	Provider Provider
	Model    string
}

func (m ModelRef) empty() bool { return m.Provider == "" || m.Model == "" }

// Task is what a call site is doing. It's the primary routing key: the router maps a
// Task → a Profile → a concrete (provider, model), so each kind of work can use the
// best/most-efficient model for it (classify on a cheap fast model, reasoning on a
// premium one, vision on a multimodal one) without call sites naming models.
type Task string

const (
	TaskChat     Task = "chat"     // conversational turns (tone matters, high volume)
	TaskNudge    Task = "nudge"    // composing a proactive nudge
	TaskClassify Task = "classify" // short label/routing decisions
	TaskExtract  Task = "extract"  // structured extraction from text (JSON)
	TaskReason   Task = "reason"   // hard reasoning / synthesis
	TaskSearch   Task = "search"   // web-search synthesis (needs a search-capable provider)
	TaskVision   Task = "vision"   // image understanding (receipts, photos)
	TaskDocument Task = "document" // reading/extracting from documents (PDFs)
	// TaskTranscribe is speech-to-text (audio → text) — a DIFFERENT task from image vision.
	// It needs an AUDIO-capable model (Gemini takes audio inline); it is deliberately its own
	// profile so image vision can route to a cheaper vision model without breaking audio.
	TaskTranscribe Task = "transcribe"
)

// Profile is a named model slot configured per deployment (provider+model). Tasks map
// to profiles so you retune the model behind "fast"/"deep" in config without touching
// call sites or the task→profile policy.
type Profile string

const (
	ProfileFast   Profile = "fast"   // cheapest/fastest — classify/extract
	ProfileChat   Profile = "chat"   // good tone, still cheap — chat/nudge
	ProfileDeep   Profile = "deep"   // premium — hard reasoning / long documents
	ProfileVision Profile = "vision" // image understanding (receipts, photos)
	// ProfileDocument reads/extracts from DOCUMENTS (PDFs). Split from vision because a PDF must
	// be sent natively — Gemini/Claude take PDFs, but the OpenAI chat API (our provider) does not,
	// so image-vision can move to a cheaper OpenAI model while documents stay on Gemini/Claude.
	ProfileDocument Profile = "document"
	// ProfileTranscribe backs speech-to-text; MUST be an AUDIO-capable model (Gemini today).
	// Split out from vision so image vision can be re-pointed at a cheaper model independently.
	ProfileTranscribe Profile = "transcribe"
	ProfileSearch     Profile = "search" // web-search capability — MUST be a search-capable model
)

// defaultTaskProfile is the built-in routing policy (task → profile). This lives in
// code (it changes with task logic/prompts, not per deployment); config tunes the
// models each profile points to. A task with no mapping falls back to ProfileChat.
var defaultTaskProfile = map[Task]Profile{
	TaskChat:       ProfileChat,
	TaskNudge:      ProfileChat,
	TaskClassify:   ProfileFast,
	TaskExtract:    ProfileFast,
	TaskReason:     ProfileDeep,
	TaskSearch:     ProfileSearch, // capability slot — must point at a web-search-capable model
	TaskVision:     ProfileVision,
	TaskDocument:   ProfileDocument,
	TaskTranscribe: ProfileTranscribe,
}

// Tier is the legacy coarse knob, kept as a fallback for call sites not yet tagged
// with a Task: cheap → the fast profile, premium → the deep profile.
type Tier int

const (
	TierCheap   Tier = iota // default — high-volume turns (→ fast profile)
	TierPremium             // escalation (→ deep profile)
)

// Effort is how hard the model should work on a request — it trades intelligence
// against latency and token spend, and on models that support it (Claude Opus 5 /
// Sonnet 5 / Opus 4.7+) it is the primary cost lever, replacing the removed
// per-request thinking budget. Empty means the provider's own default; providers
// that have no equivalent knob ignore it.
//
// Rough guidance on the Claude models: EffortXHigh for coding and agentic work,
// EffortHigh for other intelligence-sensitive work, EffortMedium/EffortLow for
// routine or latency-sensitive calls. Sweep it against your own evals rather than
// inheriting a number — the low end is stronger than it looks.
type Effort string

const (
	EffortLow    Effort = "low"
	EffortMedium Effort = "medium"
	EffortHigh   Effort = "high"
	EffortXHigh  Effort = "xhigh"
	EffortMax    Effort = "max"
)

// ToolDef declares a tool the model may call. Schema is the JSON Schema of the
// input object — the whole schema, which each adapter reshapes as its provider
// requires.
type ToolDef struct {
	Name        string
	Description string
	Schema      map[string]any
}

// ToolCall is the model asking for one invocation. ID is provider-assigned and
// OPAQUE: echo it back verbatim on the matching ToolResult and never parse it.
//
// Input is GUARANTEED to be a non-empty, syntactically valid JSON object on
// every ToolCall aikit returns: a call with no arguments arrives as `{}`, never
// as nil, "" or `null`. Providers disagree here — Anthropic emits `{}`, Gemini
// has no arguments field at all, and an OpenAI-compatible backend can return
// `"arguments": ""` — and an empty json.RawMessage is worse than merely
// inconsistent: it makes json.Marshal of the whole next request fail with
// "unexpected end of JSON input", so one such call poisons every later turn of
// the conversation. Handlers may therefore unmarshal Input unconditionally.
//
// The same normalization is applied on the way out, so an assistant turn echoed
// back with a nil Input still serialises as `{}`.
type ToolCall struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// ToolResult answers exactly one ToolCall. IsError reports that the tool failed,
// so the model can adapt rather than being told nothing.
//
// How the flag reaches the model differs per provider. Anthropic has a native
// is_error on the tool_result block and Gemini carries an "isError" key in the
// functionResponse object; the OpenAI chat API's "tool" message has NO field for
// it, so on OpenAI-compatible backends the flag is instead rendered into the
// content, which is prefixed with "Error: ". The model still sees the failure on
// all three, but on OpenAI it sees it as text — do not rely on the distinction
// being structured there.
type ToolResult struct {
	ToolCallID string
	Content    string
	IsError    bool
}

// normalizeToolInput coerces a tool call's arguments to a usable JSON object.
// An absent, empty, null or unparseable input becomes `{}` — see the guarantee
// documented on ToolCall.Input. It is applied on BOTH directions (parsing a
// provider's reply and echoing an assistant turn back), because the value that
// poisons a request can enter the history from either side.
func normalizeToolInput(in json.RawMessage) json.RawMessage {
	trimmed := bytes.TrimSpace(in)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) || !json.Valid(trimmed) {
		return json.RawMessage("{}")
	}
	return trimmed
}

// StopReason is why generation ended. Truncation matters: a turn cut off at the
// token ceiling otherwise reads exactly like a finished one.
//
// It is only meaningful when Complete returned a nil error. A transport error
// and the noop client return a zero Response, whose StopReason is the unnamed
// zero value "". A refusal is the exception, not another instance of that rule:
// it returns a populated Response — partial text and usage alongside its error,
// so a caller metering cost can still see what was billed — and that Response's
// StopReason reads StopEndTurn, not "", which would otherwise look like a
// completed turn. Branch on the error first either way; a Response's fields,
// StopReason included, are not safe to read until err == nil.
type StopReason string

const (
	StopEndTurn   StopReason = "end_turn"
	StopToolUse   StopReason = "tool_use"
	StopTruncated StopReason = "truncated"
)

// Image is an image attached to a user turn for vision (ADR-0008). Base64 is the raw
// image bytes base64-encoded (no "data:" prefix); MediaType is e.g. "image/jpeg".
type Image struct {
	Base64    string
	MediaType string
}

// Document is a document attached to a user turn (ADR-0008). Currently PDFs are sent
// natively (MediaType "application/pdf"); other formats should be pre-extracted to
// text by the caller. Base64 is the raw bytes base64-encoded.
type Document struct {
	Base64    string
	MediaType string
}

// Message is one turn in a chat exchange. Images/Documents (if any) attach to a user
// turn for multimodal requests; text-only providers ignore them.
type Message struct {
	Role      string // "system" | "user" | "assistant"
	Content   string
	Images    []Image
	Documents []Document
	// ToolCalls belong to an assistant turn being echoed back.
	ToolCalls []ToolCall
	// ToolResults belong to the user turn answering a round. Every result from
	// one round goes in ONE message: providers that want them together reject a
	// split, and Anthropic silently stops making parallel calls instead.
	ToolResults []ToolResult
}

// UserLocation focuses web-search results near the user. All fields optional; Country
// is an ISO 3166-1 alpha-2 code, Timezone an IANA name (e.g. "Asia/Jakarta").
type UserLocation struct {
	City     string
	Region   string
	Country  string
	Timezone string
}

// Request is a completion request. SystemCacheable marks the (large, reusable)
// system/memory context that a provider should prompt-cache. Model, when set,
// overrides Tier-based routing to force a specific (provider, model).
type Request struct {
	// Task is the primary routing key (preferred). Tier is the legacy fallback used
	// when Task is empty. Model, when set, overrides both to force a specific model.
	Task            Task
	Tier            Tier
	Model           *ModelRef
	Messages        []Message
	SystemCacheable string
	// CacheHistory adds a rolling cache breakpoint on the final message block, so a
	// multi-round tool loop reads its accumulated history at the cached rate instead
	// of resending it at full input rate every round. Off by default: a cache WRITE
	// costs 1.25x base input, so a single-shot call pays the premium and never reads
	// it back. Turn it on only for loops that actually reuse their history.
	//
	// LIMIT: Anthropic's cache lookup walks back at most 20 content blocks from a
	// breakpoint to find a prior entry. A round emitting more than 20 blocks (many
	// parallel tool calls, each contributing a tool_use plus a tool_result) silently
	// misses — no error, just a full-price round. If Response.CachedTokens comes back
	// zero on a loop that should be hitting, check this first.
	//
	// Anthropic-only. Providers without an explicit breakpoint model ignore it.
	CacheHistory bool
	MaxTokens    int
	// WebSearch enables the provider's server-side web-search tool for this request;
	// providers without web search ignore it (and UserLocation).
	WebSearch    bool
	UserLocation *UserLocation
	// NoFallback keeps the request on its primary model — no cross-provider failover.
	// For tasks the fallback CAN'T do (e.g. audio transcription, where a text/vision
	// fallback would "reply" instead of transcribe), a clean error beats a wrong answer.
	NoFallback bool
	// Effort tunes deliberation vs cost for this call. Empty = the provider's default
	// (Claude's is "high"). Providers without an effort knob ignore it.
	Effort Effort
	// JSONSchema constrains the reply to a JSON schema (structured outputs), so a
	// malformed shape is impossible rather than merely unlikely — worth it wherever a
	// downstream invariant depends on the fields being present. Nil = free-form text.
	//
	// Anthropic, Google and OpenAI enforce it (see supportsJSONSchema); failover with
	// a schema set only moves to providers that do. OpenAI enforces only on its own
	// endpoint and only schemas its strict mode can express — every object closed, no
	// allOf/not/if, root an object; an optional property is sent as required-but-
	// nullable, so it comes back null rather than absent. DeepSeek, Moonshot, a custom
	// OpenAIBaseURL, or a schema strict mode rejects get no enforcement, so when one is
	// the PRIMARY the reply is only checked: a ```json fence is unwrapped and anything
	// that still isn't valid JSON is ErrSchemaViolation — it parses, but its shape is
	// not guaranteed. Any provider's
	// truncated reply (StopTruncated) is ErrSchemaViolation too. A tool-calling round
	// (ToolCalls non-empty) is exempt: its answer is the calls, not the JSON.
	//
	// Google strips schema keywords Gemini does not support (pattern, minLength, …)
	// rather than sending them, and before Gemini 3 cannot combine a schema with tools:
	// with WebSearch that is an error, with function Tools the schema is dropped for
	// that round and only the JSON check above applies.
	JSONSchema map[string]any
	// Tools the model may call this turn. Complete does ONE round: a reply with
	// ToolCalls means the caller should run them, append an assistant turn
	// echoing the calls plus a user turn carrying every result, and call again.
	Tools []ToolDef
	// OnEvent, when set, receives the reply's text as it is generated. nil means no
	// streaming, and Complete behaves exactly as without this field. Rules:
	//
	//   - It is called on the goroutine that called Complete, in order, never
	//     concurrently, and never after Complete returns.
	//   - It must be quick and must not block: the stream is read on the same
	//     goroutine, so a slow callback slows the reply.
	//   - v1 sends only StreamText events. Ignore kinds you do not know; later
	//     versions may add more.
	//   - Join rule: when Complete returns a nil error, the Text of all StreamText
	//     events, joined in order, equals Response.Text exactly.
	//   - On error, events already sent stay sent, and the Response is whatever
	//     Complete returns without streaming (a refusal carries its partial text).
	//     Once text has been sent the router does not fail over: a second model's
	//     text would be glued onto the first's. An error before any text still
	//     fails over as usual.
	//
	// With JSONSchema set, the text arrives as ONE event after the reply has been
	// checked, since half a JSON document is not usable and the check may rewrite it.
	// A provider that cannot stream also sends its whole text as one event.
	OnEvent func(StreamEvent)
}

// StreamKind names what a StreamEvent carries. v1 has only StreamText.
type StreamKind string

// StreamText is a new piece of the reply's text.
const StreamText StreamKind = "text"

// StreamEvent is one thing sent to Request.OnEvent.
type StreamEvent struct {
	Kind StreamKind
	Text string // the new piece of text (for StreamText)
}

// Response is a completion result. Provider/Model report which model served the call;
// InputTokens/OutputTokens/CachedTokens/CacheWriteTokens feed per-person cost
// instrumentation — every real provider must populate them.
type Response struct {
	Text         string
	Provider     Provider
	Model        string
	InputTokens  int
	OutputTokens int
	CachedTokens int
	// CacheWriteTokens are tokens WRITTEN into the prompt cache this call, billed
	// at the model's premium CacheWrite rate. Reported separately by Anthropic and
	// excluded from InputTokens. Providers without a write premium report 0.
	CacheWriteTokens int
	// ToolCalls is non-empty when the model wants tools run.
	ToolCalls []ToolCall
	// StopReason is why generation ended. Treating StopTruncated as StopEndTurn
	// records incomplete work as finished.
	StopReason StopReason
}

// Client is the provider-agnostic LLM interface used by every call site.
type Client interface {
	// Complete returns a completion, or ErrNotConfigured when no provider is wired.
	Complete(ctx context.Context, req Request) (Response, error)
	// Enabled reports whether at least one real provider is configured.
	Enabled() bool
	// CompareTargets lists the (provider, model) pairs to run side-by-side in the admin
	// "compare providers" tool — the configured llm_compare list, or the distinct models
	// behind the profiles if that's empty. Only configured providers are returned.
	CompareTargets() []ModelRef
}

// provider runs one completion against a named model. Implemented by each vendor
// client; model-agnostic so the router picks the model per call.
type provider interface {
	complete(ctx context.Context, model string, maxTokens int, req Request) (Response, error)
}

// streamer is implemented by a provider that can send text to Request.OnEvent as
// it arrives. A provider without it is never given OnEvent (see completeOn).
type streamer interface{ streams() bool }

// canStream reports whether p implements streamer and streams() returns true.
func canStream(p provider) bool {
	s, ok := p.(streamer)
	return ok && s.streams()
}

// streamTracker remembers whether any text has reached the caller, which is what
// decides if a failed call may still fail over.
type streamTracker struct {
	next func(StreamEvent) // the caller's OnEvent
	sent bool              // a StreamText event with non-empty Text has been passed on
}

func (t *streamTracker) emit(ev StreamEvent) {
	if ev.Kind == StreamText && ev.Text != "" {
		t.sent = true
	}
	t.next(ev)
}

// Config wires the providers + the named Profiles (provider+model per slot). Only
// providers with a key are constructed; if a resolved profile points at an
// unconfigured provider, the router falls back to any configured profile so the app
// degrades instead of erroring.
type Config struct {
	AnthropicAPIKey string
	GoogleAPIKey    string
	OpenAIAPIKey    string
	OpenAIBaseURL   string // optional; set for OpenAI-compatible backends
	DeepSeekAPIKey  string
	DeepSeekBaseURL string // optional; default https://api.deepseek.com
	MoonshotAPIKey  string
	MoonshotBaseURL string // optional; default https://api.moonshot.ai/v1

	MaxTokens int
	// RequestTimeout bounds a Google or OpenAI-compatible request whose context has
	// no deadline. Zero = max(120s, 1h × MaxTokens / 128000), so long replies get
	// room. A context deadline, when set, always wins — there is no fixed cap
	// underneath it. (Anthropic requests use the Anthropic SDK's own default.)
	RequestTimeout time.Duration
	Profiles       map[Profile]ModelRef // fast/chat/deep/vision → (provider, model)
	// Fallbacks names each profile's configured failover model — used when the primary's
	// provider is down (throttled / overloaded / auth revoked / unreachable). Configure it
	// on a DIFFERENT provider so one vendor's outage never silences the app.
	Fallbacks map[Profile]ModelRef
	// CompareModels is the explicit list for the admin "compare providers" tool. Empty →
	// fall back to the distinct models behind the profiles.
	CompareModels []ModelRef

	// Logger receives the library's occasional operational logs (e.g. a failover
	// notice). Nil means no logging — the library never writes to stdout/stderr on its
	// own. Consumers pass their own *slog.Logger to route these into their log pipeline.
	Logger *slog.Logger
}

// New builds the routing Client from config. Returns the noop client if no provider
// key is set (the app runs without AI).
func New(cfg Config) Client {
	providers := map[Provider]provider{}
	if cfg.AnthropicAPIKey != "" {
		providers[ProviderAnthropic] = newAnthropic(cfg.AnthropicAPIKey)
	}
	if cfg.GoogleAPIKey != "" {
		providers[ProviderGoogle] = newGoogle(cfg.GoogleAPIKey, cfg.RequestTimeout)
	}
	if cfg.OpenAIAPIKey != "" {
		providers[ProviderOpenAI] = newOpenAI(cfg.OpenAIAPIKey, cfg.OpenAIBaseURL, cfg.RequestTimeout)
	}
	if cfg.DeepSeekAPIKey != "" {
		base := cfg.DeepSeekBaseURL
		if base == "" {
			base = "https://api.deepseek.com"
		}
		providers[ProviderDeepSeek] = newOpenAI(cfg.DeepSeekAPIKey, base, cfg.RequestTimeout)
	}
	if cfg.MoonshotAPIKey != "" {
		base := cfg.MoonshotBaseURL
		if base == "" {
			base = "https://api.moonshot.ai/v1"
		}
		providers[ProviderMoonshot] = newOpenAI(cfg.MoonshotAPIKey, base, cfg.RequestTimeout)
	}
	if len(providers) == 0 {
		return NewNoop()
	}
	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1024
	}
	lg := cfg.Logger
	if lg == nil {
		lg = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	return &router{
		providers:     providers,
		profiles:      cfg.Profiles,
		fallbacks:     cfg.Fallbacks,
		compareModels: cfg.CompareModels,
		maxTokens:     maxTokens,
		log:           lg,
	}
}

// router is the Client: resolves a request to a (provider, model) and dispatches.
type router struct {
	providers     map[Provider]provider
	profiles      map[Profile]ModelRef
	fallbacks     map[Profile]ModelRef
	compareModels []ModelRef
	maxTokens     int
	log           *slog.Logger
}

func (r *router) Enabled() bool { return len(r.providers) > 0 }

// CompareTargets returns the models to run side-by-side: the explicit compareModels, or the
// distinct models behind the profiles as a fallback — filtered to configured providers only.
func (r *router) CompareTargets() []ModelRef {
	seen := map[ModelRef]bool{}
	var out []ModelRef
	add := func(ref ModelRef) {
		if ref.empty() || seen[ref] {
			return
		}
		if _, ok := r.providers[ref.Provider]; !ok {
			return
		}
		seen[ref] = true
		out = append(out, ref)
	}
	for _, ref := range r.compareModels {
		add(ref)
	}
	if len(out) == 0 {
		for _, prof := range []Profile{ProfileFast, ProfileChat, ProfileDeep, ProfileVision, ProfileDocument, ProfileTranscribe, ProfileSearch} {
			add(r.profiles[prof])
		}
	}
	return out
}

func (r *router) Complete(ctx context.Context, req Request) (Response, error) {
	if req.OnEvent == nil {
		return r.complete(ctx, req, nil)
	}
	if len(req.JSONSchema) > 0 {
		// A schema reply is only usable whole, and checkJSONReply may rewrite it, so
		// no provider streams it: run without OnEvent and send the checked text once.
		onEvent := req.OnEvent
		req.OnEvent = nil
		resp, err := r.complete(ctx, req, nil)
		if err == nil && resp.Text != "" {
			onEvent(StreamEvent{Kind: StreamText, Text: resp.Text})
		}
		return resp, err
	}
	t := &streamTracker{next: req.OnEvent}
	req.OnEvent = t.emit
	return r.complete(ctx, req, t)
}

// complete is the routing body of Complete. t is nil when not streaming.
func (r *router) complete(ctx context.Context, req Request, t *streamTracker) (Response, error) {
	// An explicit per-request model is a deliberate choice (e.g. the admin Compare tool
	// measuring THAT model) — never silently answer with a different one.
	if req.Model != nil && !req.Model.empty() {
		return r.completeOn(ctx, *req.Model, req)
	}
	ref, prof := r.resolve(req)
	if ref.empty() {
		return Response{}, ErrNotConfigured
	}
	resp, err := r.completeOn(ctx, ref, req)
	if err == nil {
		return resp, nil
	}
	// Text already reached the caller: a fallback's reply would be appended to it.
	if t != nil && t.sent {
		if r.log != nil {
			r.log.Warn("llm: stream failed after text was sent; not falling back",
				"err", err,
				"profile", string(prof),
				"model", string(ref.Provider)+"/"+ref.Model)
		}
		return resp, err
	}
	if !shouldFailover(err) {
		return resp, err
	}
	// Some tasks must NOT fail over to a different provider (NoFallback) — e.g. audio
	// transcription, where the fallback model can't take audio and would answer
	// conversationally instead of transcribing. Surface the error instead.
	if req.NoFallback {
		return resp, err
	}
	// The primary's provider is down — throttled (429/quota), overloaded (503), auth
	// revoked (401/403, a real incident: a Google project denial killed every Google
	// profile at once), or unreachable. Fail over to the profile's CONFIGURED fallback,
	// else any model on a different provider, so one vendor's outage never silences the
	// app (a dropped nudge or briefing fails invisibly). The response keeps the model
	// that ACTUALLY answered, so the usage ledger attributes failover days honestly.
	fb := r.usableRef(r.fallbacks[prof])
	if fb.empty() || fb == ref || !r.canServe(fb.Provider, req) {
		fb = r.fallbackRef(ref, req)
	}
	if !fb.empty() {
		if resp2, err2 := r.completeOn(ctx, fb, req); err2 == nil {
			// r.log is nil for a router built as a struct literal (e.g. in tests) rather
			// than via New, which always resolves a non-nil default. Guard defensively so
			// a bypassed New() never panics on the log call.
			if r.log != nil {
				r.log.Warn("llm: primary failed; answered via fallback model",
					"err", err,
					"profile", string(prof),
					"from", string(ref.Provider)+"/"+ref.Model,
					"to", string(fb.Provider)+"/"+fb.Model)
			}
			return resp2, nil
		}
	}
	return resp, err
}

// usableRef returns ref only when its provider is actually configured.
func (r *router) usableRef(ref ModelRef) ModelRef {
	if ref.empty() {
		return ModelRef{}
	}
	if _, ok := r.providers[ref.Provider]; !ok {
		return ModelRef{}
	}
	return ref
}

func (r *router) completeOn(ctx context.Context, ref ModelRef, req Request) (Response, error) {
	p, ok := r.providers[ref.Provider]
	if !ok {
		return Response{}, ErrNotConfigured
	}
	// A provider that cannot stream gets a plain request; its whole text goes out as
	// one event once the reply has passed checkJSONReply.
	onEvent := req.OnEvent
	synth := onEvent != nil && !canStream(p)
	if synth {
		req.OnEvent = nil
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = r.maxTokens
	}
	resp, err := p.complete(ctx, ref.Model, maxTokens, req)
	if err != nil {
		// A refusal returns a POPULATED Response (partial text + billed tokens)
		// alongside its error; a transport error returns a zero Response. Stamp the
		// refusal so cost metering attributes the spend to a model even when no
		// fallback answers (NoFallback, or the fallback also declined).
		if errors.Is(err, ErrRefused) {
			resp.Provider = ref.Provider
			resp.Model = ref.Model
		}
		return resp, err
	}
	resp.Provider = ref.Provider
	resp.Model = ref.Model
	resp, err = checkJSONReply(resp, req)
	if err == nil && synth && resp.Text != "" {
		onEvent(StreamEvent{Kind: StreamText, Text: resp.Text})
	}
	return resp, err
}

// schemaEnforcer is implemented by a provider whose enforcement depends on its
// configuration or on the schema itself (OpenAI: only its own endpoint, and only
// for schemas strict mode accepts).
type schemaEnforcer interface {
	enforcesJSONSchema(schema map[string]any) bool
}

// supportsJSONSchema reports whether a configured provider ENFORCES schema, as
// opposed to merely being asked nicely. The one place that knowledge lives.
func (r *router) supportsJSONSchema(p Provider, schema map[string]any) bool {
	switch p {
	case ProviderAnthropic, ProviderGoogle:
		return true
	}
	e, ok := r.providers[p].(schemaEnforcer)
	return ok && e.enforcesJSONSchema(schema)
}

// canServe reports whether a provider may take req on the failover path. A schema
// request must not fail over onto a provider that would drop the schema: the
// primary's outage would quietly turn into an unenforced answer.
func (r *router) canServe(p Provider, req Request) bool {
	return len(req.JSONSchema) == 0 || r.supportsJSONSchema(p, req.JSONSchema)
}

// checkJSONReply is the post-condition for a schema request (see Request.JSONSchema):
// a truncated reply, or text that isn't JSON, is ErrSchemaViolation with the
// Response kept populated for metering. A ```json fence is unwrapped first, which
// is what a non-enforcing provider most often wraps an otherwise-valid reply in.
func checkJSONReply(resp Response, req Request) (Response, error) {
	if len(req.JSONSchema) == 0 || len(resp.ToolCalls) > 0 {
		return resp, nil
	}
	if resp.StopReason == StopTruncated {
		return resp, fmt.Errorf("%w: %s/%s reply was cut off at the token ceiling; raise MaxTokens",
			ErrSchemaViolation, resp.Provider, resp.Model)
	}
	text := unfenceJSON(resp.Text)
	if !json.Valid([]byte(text)) {
		return resp, fmt.Errorf("%w: %s/%s reply is not valid JSON", ErrSchemaViolation, resp.Provider, resp.Model)
	}
	resp.Text = text
	return resp, nil
}

// unfenceJSON trims whitespace and removes one surrounding Markdown code fence
// (```json … ``` or a bare ``` … ```). Anything else is returned trimmed.
func unfenceJSON(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") || !strings.HasSuffix(s, "```") || len(s) < 6 {
		return s
	}
	inner := s[3 : len(s)-3]
	if nl := strings.IndexByte(inner, '\n'); nl >= 0 && !strings.ContainsAny(inner[:nl], "{[\"") {
		inner = inner[nl+1:] // drop the info string ("json", "JSON", "")
	}
	return strings.TrimSpace(inner)
}

// fallbackRef picks a usable model on a DIFFERENT provider than the one that just failed —
// preferring the cheap-but-capable chat model (Claude Haiku), then deep, so a throttled
// Gemini transparently degrades to Claude. Candidates that cannot serve req (a schema
// request on a non-enforcing provider) are skipped.
func (r *router) fallbackRef(failed ModelRef, req Request) ModelRef {
	for _, prof := range []Profile{ProfileChat, ProfileDeep, ProfileVision, ProfileFast, ProfileTranscribe, ProfileSearch} {
		if cand := r.usable(prof); !cand.empty() && cand.Provider != failed.Provider && r.canServe(cand.Provider, req) {
			return cand
		}
	}
	return ModelRef{}
}

// shouldFailover reports whether an error means the PROVIDER is unusable — worth
// answering via a different provider. Matches throttles/overloads (429/quota/503),
// auth failures (401/403 — key revoked, project denied), server errors (5xx), and
// network-level failures. Content-shaped 400s are excluded: they'd likely fail
// anywhere, and provider-knob 400s are handled inside each provider client.
func shouldFailover(err error) bool {
	if err == nil {
		return false
	}
	// A safety refusal is not a provider outage, but it is worth the same retry: policy
	// classifiers differ per model, so the profile's configured fallback is frequently
	// the model that will answer. Checked with errors.Is rather than by string match —
	// a refusal explanation is provider prose and must never be pattern-matched.
	if errors.Is(err, ErrRefused) {
		return true
	}
	// A schema violation is the reply's fault, not the provider's (see its doc), and
	// its message names the model — which must never be keyword-matched below.
	if errors.Is(err, ErrSchemaViolation) {
		return false
	}
	s := strings.ToLower(err.Error())
	for _, k := range []string{
		// throttle / overload
		"429", "rate limit", "quota", "exceeded", "resource_exhausted",
		"503", "high demand", "overloaded", "unavailable", "please retry",
		"please try again", "temporarily",
		// auth / access revoked
		"401", "403", "unauthorized", "forbidden", "permission", "denied",
		// model gone (retired/renamed — e.g. gemini-2.5-flash "no longer available")
		"404", "not found", "no longer available",
		// server errors
		"500", "502", "504", "internal server error", "bad gateway",
		// network
		"connection refused", "connection reset", "no such host",
		"timeout", "deadline exceeded", "unexpected eof",
	} {
		if strings.Contains(s, k) {
			return true
		}
	}
	return false
}

// resolve picks the (provider, model) — and the profile it came from, so failover can
// look up that profile's configured fallback. Order: the Task's profile; else the
// legacy Tier (cheap→fast, premium→deep); and if the chosen profile's provider isn't
// configured, fall back to any configured profile (degrade rather than fail).
// (An explicit req.Model override is handled in Complete, before resolve.)
func (r *router) resolve(req Request) (ModelRef, Profile) {
	if req.Task != "" {
		prof, ok := defaultTaskProfile[req.Task]
		if !ok {
			prof = ProfileChat
		}
		if ref := r.usable(prof); !ref.empty() {
			return ref, prof
		}
	}
	tierProfile := ProfileFast
	if req.Tier == TierPremium {
		tierProfile = ProfileDeep
	}
	if ref := r.usable(tierProfile); !ref.empty() {
		return ref, tierProfile
	}
	// Last resort: any configured profile so the app degrades instead of erroring.
	for _, prof := range []Profile{ProfileFast, ProfileChat, ProfileDeep, ProfileVision, ProfileDocument, ProfileTranscribe, ProfileSearch} {
		if ref := r.usable(prof); !ref.empty() {
			return ref, prof
		}
	}
	return ModelRef{}, ""
}

// usable returns a profile's ModelRef only if its provider is actually configured.
func (r *router) usable(prof Profile) ModelRef {
	ref := r.profiles[prof]
	if ref.empty() {
		return ModelRef{}
	}
	if _, ok := r.providers[ref.Provider]; !ok {
		return ModelRef{}
	}
	return ref
}
