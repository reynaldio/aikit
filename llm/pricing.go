package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Rate is a model's price, in USD per 1,000,000 tokens. CachedRead is the discounted
// rate for prompt-cache reads (billed apart from Input, which excludes cache).
type Rate struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CachedRead float64 `json:"cachedRead"`
}

// Cost returns the USD cost of a call at this rate. cachedTokens are billed at the
// cache-read rate and are NOT part of inputTokens (providers report them apart —
// see the provider clients, which normalize input to exclude cache).
func (r Rate) Cost(inputTokens, outputTokens, cachedTokens int) float64 {
	return float64(inputTokens)/1e6*r.Input +
		float64(outputTokens)/1e6*r.Output +
		float64(cachedTokens)/1e6*r.CachedRead
}

// DefaultPrices is the built-in fallback catalog for the models the router routes
// to. ⚠️ VERIFY against each provider's pricing — these are placeholders, and the
// canonical source is your invoice. Apps override these via a PriceBook (admin-set
// rates) and/or refresh them from the LiteLLM feed.
var DefaultPrices = map[string]Rate{
	// Anthropic — verified against platform.claude.com/docs/en/about-claude/pricing (cache-read
	// hit = 0.1x base input). Our two ACTIVE models (haiku-4-5, opus-4-8) were already correct.
	"claude-haiku-4-5":  {Input: 1.00, Output: 5.00, CachedRead: 0.10},
	"claude-opus-4-8":   {Input: 5.00, Output: 25.00, CachedRead: 0.50},
	"claude-opus-5":     {Input: 5.00, Output: 25.00, CachedRead: 0.50},
	"claude-sonnet-5":   {Input: 2.00, Output: 10.00, CachedRead: 0.20}, // intro thru 2026-08-31; then 3.00/15.00
	"claude-sonnet-4-5": {Input: 3.00, Output: 15.00, CachedRead: 0.30},
	"claude-fable-5":    {Input: 10.00, Output: 50.00, CachedRead: 1.00},
	// Google Gemini — verified against ai.google.dev/gemini-api/docs/pricing (paid tier, text/
	// image/video input; audio input costs more). Output INCLUDES "thinking" tokens — and 2.5/3.x
	// Flash burn a lot of them (a simple turn can spend 400–500 thought tokens), so effective cost
	// is well above the input rate implies unless thinking is suppressed. Pro's input/output rise
	// above a 200k-token prompt (we bill the ≤200k rate — our prompts are far under).
	// ⚠️ The `*-latest` entries are Google's ROLLING aliases: they currently resolve to the 2.5
	// series (below), but Google may repoint them to a 3.x Flash ($1.50/$7.50–9.00 — up to 5×
	// pricier). Pin a concrete gemini-2.5-* / gemini-3.x-* id in the profiles if you need price
	// stability. (Backlog #34: migrate prod to gemini-3.x + its own paid key.)
	"gemini-flash-latest":      {Input: 0.30, Output: 2.50, CachedRead: 0.03},   // ≈ gemini-2.5-flash today
	"gemini-flash-lite-latest": {Input: 0.10, Output: 0.40, CachedRead: 0.01},   // ≈ gemini-2.5-flash-lite today
	"gemini-pro-latest":        {Input: 1.25, Output: 10.00, CachedRead: 0.125}, // ≈ gemini-2.5-pro today (≤200k)
	// Pinned 2.5 versions
	"gemini-2.5-flash":      {Input: 0.30, Output: 2.50, CachedRead: 0.03},
	"gemini-2.5-flash-lite": {Input: 0.10, Output: 0.40, CachedRead: 0.01},
	"gemini-2.5-pro":        {Input: 1.25, Output: 10.00, CachedRead: 0.125}, // ≤200k prompt; >200k = 2.50/15.00
	// Gemini 3.x (newer, pricier Flash tier)
	"gemini-3.6-flash":       {Input: 1.50, Output: 7.50, CachedRead: 0.15},
	"gemini-3.5-flash":       {Input: 1.50, Output: 9.00, CachedRead: 0.15},
	"gemini-3.5-flash-lite":  {Input: 0.30, Output: 2.50, CachedRead: 0.03},
	"gemini-3.1-flash-lite":  {Input: 0.25, Output: 1.50, CachedRead: 0.025},
	"gemini-3.1-pro-preview": {Input: 2.00, Output: 12.00, CachedRead: 0.20}, // ≤200k prompt
	// OpenAI. Rates below are the STANDARD tier, SHORT-context column (per 1M tokens) — the
	// common case and the level of detail the rest of this catalog uses (flat Input/Output/
	// CachedRead). The 2026 pricing page also lists a LONG-context column (~2× these) and a
	// "cache writes" price (Sol 6.25 / Terra 2.50 / Luna 0.25 short) that our flat Rate struct
	// does NOT model — so metering an OpenAI model on very long prompts under-counts a bit.
	// Batch/Flex/Fast-mode tiers differ again. Model ids match the API (verified off the page).
	"gpt-4o":      {Input: 2.50, Output: 10.00, CachedRead: 1.25},
	"gpt-4o-mini": {Input: 0.15, Output: 0.60, CachedRead: 0.075},
	// GPT-5.6 — Luna = cheap everyday tier, Terra = balanced, Sol = flagship. Cost note: Luna
	// ($0.20/$1.20) undercuts gemini-flash-latest but is STILL pricier than the gemini-flash-lite
	// ($0.10/$0.40) we use for chat/search — a quality option, not a cheaper one.
	"gpt-5.6-luna":  {Input: 0.20, Output: 1.20, CachedRead: 0.02},
	"gpt-5.6-terra": {Input: 2.00, Output: 12.00, CachedRead: 0.20},
	"gpt-5.6-sol":   {Input: 5.00, Output: 30.00, CachedRead: 0.50},
	// GPT-5.5 / 5.4 lineup (still current on the page). *-pro have no cached-input discount.
	"gpt-5.5":      {Input: 5.00, Output: 30.00, CachedRead: 0.50},
	"gpt-5.5-pro":  {Input: 30.00, Output: 180.00, CachedRead: 0},
	"gpt-5.4":      {Input: 2.50, Output: 15.00, CachedRead: 0.25},
	"gpt-5.4-mini": {Input: 0.75, Output: 4.50, CachedRead: 0.075},
	"gpt-5.4-nano": {Input: 0.20, Output: 1.25, CachedRead: 0.02},
	"gpt-5.4-pro":  {Input: 30.00, Output: 180.00, CachedRead: 0},
	// DeepSeek (OpenAI-compatible) — off-peak (standard) rates from
	// api-docs.deepseek.com/quick_start/pricing (Input = cache-miss, CachedRead = cache-hit).
	// ⚠️ Peak hours (09:00–12:00 & 14:00–18:00 Beijing/UTC+8) are announced to be 2× these
	// (effective date pending). Catalog holds the base rate; treat peak as a transient 2× surge.
	// The API aliases map to the V4 tiers: deepseek-chat → v4-flash, deepseek-reasoner → v4-pro.
	"deepseek-chat":     {Input: 0.14, Output: 0.28, CachedRead: 0.0028},
	"deepseek-v4-flash": {Input: 0.14, Output: 0.28, CachedRead: 0.0028},
	"deepseek-reasoner": {Input: 0.435, Output: 0.87, CachedRead: 0.003625},
	"deepseek-v4-pro":   {Input: 0.435, Output: 0.87, CachedRead: 0.003625},
	// Moonshot (Kimi, OpenAI-compatible). ⚠️ PLACEHOLDERS — verify at platform.moonshot.ai and
	// add the exact model id you configure (e.g. a kimi-k2-* / newer id) if it's missing here;
	// an unpriced model just meters at $0 until its rate is set (admin Prices page or here).
	"kimi-k2-0711-preview": {Input: 0.60, Output: 2.50, CachedRead: 0.15},
	"moonshot-v1-8k":       {Input: 0.20, Output: 2.00, CachedRead: 0.05},
	// Non-LLM APIs metered on this ledger (unit = 1 call, recorded as 1 input "token"):
	// Google Places (New) Enterprise text search — $35 / 1000 calls (key mirrors
	// usage.PlacesUsageModel). So Input per-1M-units = 0.035 × 1e6 = 35000.
	"places-text-search-enterprise": {Input: 35000},
}

// PriceBook resolves a model's Rate from layered sources, most-authoritative first:
// admin overrides → the fetched feed (LiteLLM) → the built-in defaults. Apps load the
// overrides from their own store and pass them in; the merge policy lives here so it's
// shared. A model with no rate anywhere resolves to the zero Rate (cost 0).
type PriceBook struct {
	overrides map[string]Rate
	fetched   map[string]Rate
}

// NewPriceBook builds a book from the fetched feed + admin overrides (either may be
// nil). Defaults are always consulted last.
func NewPriceBook(fetched, overrides map[string]Rate) *PriceBook {
	return &PriceBook{overrides: overrides, fetched: fetched}
}

// Rate resolves a model's rate: override > fetched > default > zero.
func (b *PriceBook) Rate(model string) Rate {
	if b != nil {
		if r, ok := b.overrides[model]; ok {
			return r
		}
		if r, ok := b.fetched[model]; ok {
			return r
		}
	}
	return DefaultPrices[model]
}

// Cost is the USD cost of a call for a model, using the resolved rate.
func (b *PriceBook) Cost(model string, inputTokens, outputTokens, cachedTokens int) float64 {
	return b.Rate(model).Cost(inputTokens, outputTokens, cachedTokens)
}

// LiteLLMFeedURL is the community-maintained price catalog aikit refreshes from. It's
// a plain JSON data file (MIT-licensed) — not the LiteLLM runtime/SDK.
const LiteLLMFeedURL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

// litellmEntry is the subset of a LiteLLM catalog row we read. Prices are per-token.
type litellmEntry struct {
	InputCostPerToken       float64 `json:"input_cost_per_token"`
	OutputCostPerToken      float64 `json:"output_cost_per_token"`
	CacheReadInputTokenCost float64 `json:"cache_read_input_token_cost"`
}

// FetchLiteLLM downloads + parses the LiteLLM price catalog into per-Mtok Rates keyed
// by model name. It returns only entries for `models` (the models you actually use) so
// the result is bounded and name-matched; a model absent from the feed is simply
// omitted (the caller keeps its existing/default rate). Per-token feed values are
// scaled ×1e6 to our per-Mtok unit.
func FetchLiteLLM(ctx context.Context, hc *http.Client, models []string) (map[string]Rate, error) {
	if hc == nil {
		hc = &http.Client{Timeout: 30 * time.Second}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, LiteLLMFeedURL, nil)
	if err != nil {
		return nil, err
	}
	res, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode >= 300 {
		return nil, fmt.Errorf("litellm feed: status %d", res.StatusCode)
	}
	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}
	var catalog map[string]litellmEntry
	if err := json.Unmarshal(raw, &catalog); err != nil {
		return nil, fmt.Errorf("litellm feed: decode: %w", err)
	}
	out := map[string]Rate{}
	for _, m := range models {
		e, ok := catalog[m]
		if !ok || (e.InputCostPerToken == 0 && e.OutputCostPerToken == 0) {
			continue // absent or priceless entry → keep the caller's existing rate
		}
		out[m] = Rate{
			Input:      e.InputCostPerToken * 1e6,
			Output:     e.OutputCostPerToken * 1e6,
			CachedRead: e.CacheReadInputTokenCost * 1e6,
		}
	}
	return out, nil
}
