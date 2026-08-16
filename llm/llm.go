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
// This is the seam intended to be extracted into a shared `llmkit` module once it
// stabilizes (used by Nathan + sibling projects).
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"strings"
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
	TaskChat:     ProfileChat,
	TaskNudge:    ProfileChat,
	TaskClassify: ProfileFast,
	TaskExtract:  ProfileFast,
	TaskReason:   ProfileDeep,
	TaskSearch:   ProfileSearch, // capability slot — must point at a web-search-capable model
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
type ToolCall struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// ToolResult answers exactly one ToolCall. IsError reports that the tool failed,
// so the model can adapt rather than being told nothing.
type ToolResult struct {
	ToolCallID string
	Content    string
	IsError    bool
}

// StopReason is why generation ended. Truncation matters: a turn cut off at the
// token ceiling otherwise reads exactly like a finished one.
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
	MaxTokens       int
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
	// Providers without schema enforcement ignore it, so a caller that REQUIRES the
	// guarantee must pin the model rather than rely on profile routing.
	JSONSchema map[string]any
	// Tools the model may call this turn. Complete does ONE round: a reply with
	// ToolCalls means the caller should run them, append an assistant turn
	// echoing the calls plus a user turn carrying every result, and call again.
	Tools []ToolDef
}

// Response is a completion result. Provider/Model report which model served the call;
// InputTokens/OutputTokens/CachedTokens feed per-person cost instrumentation — every
// real provider must populate them.
type Response struct {
	Text         string
	Provider     Provider
	Model        string
	InputTokens  int
	OutputTokens int
	CachedTokens int
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
	Profiles  map[Profile]ModelRef // fast/chat/deep/vision → (provider, model)
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
		providers[ProviderGoogle] = newGoogle(cfg.GoogleAPIKey)
	}
	if cfg.OpenAIAPIKey != "" {
		providers[ProviderOpenAI] = newOpenAI(cfg.OpenAIAPIKey, cfg.OpenAIBaseURL)
	}
	if cfg.DeepSeekAPIKey != "" {
		base := cfg.DeepSeekBaseURL
		if base == "" {
			base = "https://api.deepseek.com"
		}
		providers[ProviderDeepSeek] = newOpenAI(cfg.DeepSeekAPIKey, base)
	}
	if cfg.MoonshotAPIKey != "" {
		base := cfg.MoonshotBaseURL
		if base == "" {
			base = "https://api.moonshot.ai/v1"
		}
		providers[ProviderMoonshot] = newOpenAI(cfg.MoonshotAPIKey, base)
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
	if err == nil || !shouldFailover(err) {
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
	if fb.empty() || fb == ref {
		fb = r.fallbackRef(ref)
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
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = r.maxTokens
	}
	resp, err := p.complete(ctx, ref.Model, maxTokens, req)
	if err != nil {
		return resp, err
	}
	resp.Provider = ref.Provider
	resp.Model = ref.Model
	return resp, nil
}

// fallbackRef picks a usable model on a DIFFERENT provider than the one that just failed —
// preferring the cheap-but-capable chat model (Claude Haiku), then deep, so a throttled
// Gemini transparently degrades to Claude.
func (r *router) fallbackRef(failed ModelRef) ModelRef {
	for _, prof := range []Profile{ProfileChat, ProfileDeep, ProfileVision, ProfileFast, ProfileTranscribe, ProfileSearch} {
		if cand := r.usable(prof); !cand.empty() && cand.Provider != failed.Provider {
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
