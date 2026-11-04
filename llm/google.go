package llm

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// googleProvider is the Google Gemini provider (generateContent REST API, raw HTTP —
// no SDK, matching the repo's hand-rolled-HTTP idiom for non-Anthropic backends).
// Gemini uses roles "user"/"model" and a separate system_instruction; usage comes back
// in usageMetadata (prompt / candidates / cached token counts).
type googleProvider struct {
	apiKey  string
	baseURL string // without the /models/... suffix; tests point it at httptest
	http    *http.Client
	timeout time.Duration // Config.RequestTimeout: deadline for a ctx without one; 0 = scaled
}

const geminiBaseURL = "https://generativelanguage.googleapis.com/v1beta"

func newGoogle(apiKey string, timeout time.Duration, lg *slog.Logger) provider {
	return &googleProvider{apiKey: apiKey, baseURL: geminiBaseURL, http: newProviderHTTPClient(lg), timeout: timeout}
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	Thought          bool                    `json:"thought,omitempty"`
	InlineData       *geminiInline           `json:"inline_data,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
	// ThoughtSignature is Gemini's opaque proof of the model's reasoning. Gemini 3
	// rejects a history whose function-call turn comes back without it (400).
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
	// ToolCall and ToolResponse are a server-side tool's call and result (Google
	// Search run by Gemini itself). They only arrive with includeServerSideToolInvocations,
	// are never a ToolCall of ours, and only matter when echoed back.
	ToolCall     json.RawMessage `json:"toolCall,omitempty"`
	ToolResponse json.RawMessage `json:"toolResponse,omitempty"`

	// raw is the part exactly as Gemini sent it. An echoed part is sent back as
	// raw, so fields aikit does not know survive the round trip.
	raw json.RawMessage
}

// UnmarshalJSON decodes a part and keeps its raw bytes.
func (p *geminiPart) UnmarshalJSON(b []byte) error {
	type plain geminiPart
	if err := json.Unmarshal(b, (*plain)(p)); err != nil {
		return err
	}
	p.raw = append(json.RawMessage(nil), b...)
	return nil
}

// MarshalJSON sends a decoded part back as it came, and a built one field by field.
func (p geminiPart) MarshalJSON() ([]byte, error) {
	if p.raw != nil {
		return p.raw, nil
	}
	type plain geminiPart
	return json.Marshal(plain(p))
}

// echoed reports whether the part must go back in history for the next round to
// work: it carries a signature or a server-side tool step.
func (p geminiPart) echoed() bool {
	return p.ThoughtSignature != "" || len(p.ToolCall) > 0 || len(p.ToolResponse) > 0
}

// plainText reports a text part with nothing else on it, so streamed pieces can
// be joined into one part.
func (p geminiPart) plainText() bool {
	return p.Text != "" && p.FunctionCall == nil && p.InlineData == nil && !p.echoed()
}

// geminiProviderData is the Response.ProviderData of a Gemini model turn.
type geminiProviderData struct {
	Provider Provider     `json:"provider"`
	Parts    []geminiPart `json:"parts"`
}

// geminiTurnData returns the model turn to echo, or nil when no part needs it.
func geminiTurnData(parts []geminiPart) []byte {
	need := false
	for _, p := range parts {
		need = need || p.echoed()
	}
	if !need {
		return nil
	}
	// Without HTML escaping the stored parts stay byte for byte as Gemini sent them.
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if enc.Encode(geminiProviderData{Provider: ProviderGoogle, Parts: parts}) != nil {
		return nil
	}
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n"))
}

// geminiEchoParts returns the parts of m.ProviderData when it is a Gemini turn whose
// function calls match m.ToolCalls in order (by name). Otherwise ok is false and
// the turn is rebuilt from Content and ToolCalls.
func geminiEchoParts(m Message) (parts []geminiPart, ok bool) {
	if len(m.ProviderData) == 0 {
		return nil, false
	}
	var d geminiProviderData
	if json.Unmarshal(m.ProviderData, &d) != nil || d.Provider != ProviderGoogle || len(d.Parts) == 0 {
		return nil, false
	}
	n := 0
	for _, p := range d.Parts {
		if p.FunctionCall == nil {
			continue
		}
		if n >= len(m.ToolCalls) || m.ToolCalls[n].Name != p.FunctionCall.Name {
			return nil, false
		}
		n++
	}
	if n != len(m.ToolCalls) {
		return nil, false
	}
	return d.Parts, true
}

// geminiSkipSignature is Google's documented stand-in for a lost thought signature.
// It stops the 400 on a rebuilt function-call turn (a history stored without
// ProviderData, or written by another provider), at some cost to answer quality.
const geminiSkipSignature = "skip_thought_signature_validator"

// geminiFunctionCall is the model's request to invoke a tool. Args is a bare
// map because Gemini's function arguments are an arbitrary JSON object, not a
// typed schema aikit knows about.
type geminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

// geminiFunctionResponse answers a geminiFunctionCall. Response is a bare map
// (verified against the API reference: "response: record<string, unknown>"),
// so it can carry arbitrary keys — including "isError", which has no field of
// its own in the wire format.
type geminiFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type geminiInline struct {
	MimeType string `json:"mime_type"`
	Data     string `json:"data"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiRequest struct {
	SystemInstruction *geminiContent    `json:"system_instruction,omitempty"`
	Contents          []geminiContent   `json:"contents"`
	Tools             []geminiTool      `json:"tools,omitempty"`
	ToolConfig        *geminiToolConfig `json:"toolConfig,omitempty"`
	GenerationConfig  struct {
		MaxOutputTokens int                   `json:"maxOutputTokens,omitempty"`
		ThinkingConfig  *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
		// ResponseFormat carries Request.JSONSchema. It replaces the deprecated
		// responseMimeType + responseSchema/responseJsonSchema fields, and its schema
		// takes standard JSON Schema.
		ResponseFormat *geminiResponseFormat `json:"responseFormat,omitempty"`
	} `json:"generationConfig"`
}

// geminiResponseFormat is generationConfig.responseFormat: per-modality output
// configuration, of which aikit only sets text.
type geminiResponseFormat struct {
	Text *geminiTextFormat `json:"text,omitempty"`
}

// geminiTextFormat constrains text output. MimeType is an enum on the wire
// ("APPLICATION_JSON", "TEXT_PLAIN"), not a MIME string; Schema is only read with
// APPLICATION_JSON.
type geminiTextFormat struct {
	MimeType string         `json:"mimeType"`
	Schema   map[string]any `json:"schema,omitempty"`
}

// geminiTool carries the built-in tools. GoogleSearch enables Gemini's native Google
// Search grounding — the cross-provider equivalent of Anthropic's web-search tool
// (req.WebSearch), and much cheaper for the recipe web search. Empty object payload.
// FunctionDeclarations carries user-declared tools (ToolDef); the JSON key is
// verified against the current API reference/examples as "functionDeclarations"
// (camelCase) — NOT the snake_case "function_declarations" this repo uses for
// some other fields.
type geminiTool struct {
	GoogleSearch         *geminiGoogleSearch         `json:"google_search,omitempty"`
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations,omitempty"`
}

type geminiGoogleSearch struct{}

// geminiToolConfig is toolConfig. IncludeServerSideToolInvocations must be on to
// combine a built-in tool (Google Search) with function declarations: without it
// Gemini rejects the request with a 400 (checked live, 2026-11-03). With it, the
// search steps come back as toolCall/toolResponse parts that must be echoed.
type geminiToolConfig struct {
	IncludeServerSideToolInvocations bool `json:"includeServerSideToolInvocations,omitempty"`
}

// geminiFunctionDeclaration declares one tool to the model.
type geminiFunctionDeclaration struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// geminiFunctionDecls maps ToolDefs to declarations. Returns nil for none so
// the caller can leave the tools array off entirely.
func geminiFunctionDecls(defs []ToolDef) []geminiFunctionDeclaration {
	if len(defs) == 0 {
		return nil
	}
	out := make([]geminiFunctionDeclaration, 0, len(defs))
	for _, d := range defs {
		out = append(out, geminiFunctionDeclaration{
			Name: d.Name, Description: d.Description, Parameters: d.Schema,
		})
	}
	return out
}

// geminiCallID synthesizes an ID for one Gemini function call, since Gemini
// issues none. It must be unique across a whole CONVERSATION, not merely within
// one reply: an ID derived from the call's index and name (the obvious choice)
// repeats the moment a second round calls the same tool, and while Gemini itself
// survives that — it resolves against the preceding assistant turn — a history
// handed to Anthropic or OpenAI on the cross-provider failover path would then
// carry two distinct calls sharing one ID. Randomness is what makes the string
// unique; the entropy is far past any conversation length.
//
// The ID is deliberately OPAQUE: it encodes neither the position nor the
// function name, so there is nothing in it to tempt a parser. Nothing may parse
// it — resolution is by equality against the call list, which is also what lets
// a foreign provider's ID survive a failover unchanged.
func geminiCallID() string {
	var b [12]byte
	// crypto/rand.Read never returns an error; it crashes the program instead.
	_, _ = rand.Read(b[:])
	return "gemini" + hex.EncodeToString(b[:])
}

// geminiToolCalls pulls function calls out of a reply, synthesizing an ID for
// each (see geminiCallID).
func geminiToolCalls(parts []geminiPart) []ToolCall {
	var out []ToolCall
	for _, p := range parts {
		if p.FunctionCall == nil {
			continue
		}
		// A call with no arguments marshals to `null` (Args is a nil map), which is
		// not the `{}` ToolCall.Input guarantees — normalizeToolInput fixes both
		// that and a marshal failure.
		args, err := json.Marshal(p.FunctionCall.Args)
		if err != nil {
			args = nil
		}
		out = append(out, ToolCall{
			ID:    geminiCallID(),
			Name:  p.FunctionCall.Name,
			Input: normalizeToolInput(args),
		})
	}
	return out
}

// geminiToolResponses renders results as functionResponse parts, IN CALL ORDER.
// calls is the preceding assistant turn's ToolCalls: it is the only thing that
// knows a result's function name and position, since the ID carries neither.
// Gemini disambiguates same-name calls by POSITION, so silently skipping an
// unmatched call would shift every later result onto the wrong index — that is
// precisely the misattribution this design exists to prevent. A result whose
// ToolCallID names no call in calls, or a call with no matching result, is
// therefore a loud error naming the offending ID rather than a silent drop.
func geminiToolResponses(results []ToolResult, calls []ToolCall) ([]geminiPart, error) {
	byID := make(map[string]ToolResult, len(results))
	callIDs := make(map[string]bool, len(calls))
	for _, c := range calls {
		callIDs[c.ID] = true
	}
	for _, r := range results {
		if !callIDs[r.ToolCallID] {
			return nil, fmt.Errorf("%w: gemini: tool result references unknown call %q", ErrToolResultMismatch, r.ToolCallID)
		}
		byID[r.ToolCallID] = r
	}
	out := make([]geminiPart, 0, len(calls))
	for _, c := range calls {
		r, ok := byID[c.ID]
		if !ok {
			return nil, fmt.Errorf("%w: gemini: missing tool result for call %q", ErrToolResultMismatch, c.ID)
		}
		out = append(out, geminiPart{FunctionResponse: &geminiFunctionResponse{
			Name:     c.Name,
			Response: map[string]any{"content": r.Content, "isError": r.IsError},
		}})
	}
	return out, nil
}

// geminiStopReason maps finishReason, given whether the turn also parsed any
// tool calls. Unknown or absent reads as a completed turn rather than inventing
// a failure.
//
// The tool-call arm exists because Gemini's finishReason on a functionCall turn
// is just "STOP" — it carries no tool-use signal of its own, unlike Anthropic's
// tool_use stop reason or OpenAI's tool_calls finish_reason, so the presence of
// parsed calls IS that signal.
//
// Truncation still wins over it. A turn that hit the token ceiling WHILE
// emitting calls is truncated first and foremost — the calls are reported either
// way, but reporting StopToolUse would hide the cut-off, and Anthropic reports
// StopTruncated for the identical situation. Callers therefore branch on
// len(ToolCalls) to run tools and on StopTruncated to notice the ceiling, and
// get the same answer on all three providers.
func geminiStopReason(fr string, hasToolCalls bool) StopReason {
	if fr == "MAX_TOKENS" {
		return StopTruncated
	}
	if hasToolCalls {
		return StopToolUse
	}
	return StopEndTurn
}

// geminiBlockReasons are the finishReason (candidate-level) and blockReason
// (prompt-level) values that mean a safety classifier declined the request — a
// 200 with no usable content. RECITATION (copyright) and MAX_TOKENS are
// deliberately excluded: neither is a safety refusal, and neither has a
// classifiers-differ-per-model fallback story.
var geminiBlockReasons = map[string]bool{
	"SAFETY":             true,
	"PROHIBITED_CONTENT": true,
	"BLOCKLIST":          true,
	"SPII":               true,
}

// geminiRefusalCategory returns the safety category when a response is a decline,
// or "" for a normal turn. Gemini signals a block two ways: a candidate whose
// finishReason is a block reason, or — for a prompt rejected before generation —
// no candidates at all plus a promptFeedback.blockReason.
func geminiRefusalCategory(out geminiResponse) string {
	if len(out.Candidates) == 0 {
		if out.PromptFeedback != nil && geminiBlockReasons[out.PromptFeedback.BlockReason] {
			return out.PromptFeedback.BlockReason
		}
		return ""
	}
	if geminiBlockReasons[out.Candidates[0].FinishReason] {
		return out.Candidates[0].FinishReason
	}
	return ""
}

// geminiThinkingConfig caps Gemini's internal reasoning. Thinking tokens COUNT
// AGAINST maxOutputTokens, so with the default budget a small llm_max_tokens gets
// eaten by reasoning and the visible reply truncates mid-sentence. The knob changed
// across generations: 2.5 models take thinkingBudget (0 = off); the newer models
// behind the -latest aliases reject it and take thinkingLevel ("minimal") instead.
type geminiThinkingConfig struct {
	ThinkingBudget *int   `json:"thinkingBudget,omitempty"`
	ThinkingLevel  string `json:"thinkingLevel,omitempty"`
}

type geminiResponse struct {
	Candidates []struct {
		Content           geminiContent    `json:"content"`
		FinishReason      string           `json:"finishReason"`
		GroundingMetadata *geminiGrounding `json:"groundingMetadata"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount"`
		ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	} `json:"usageMetadata"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	// PromptFeedback carries a prompt-level safety block: the whole request was
	// declined before generation, so Candidates is empty and BlockReason names why.
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
}

// geminiGrounding is what Google Search did for a candidate: the queries it ran, the
// pages it read, and which pages back which parts of the answer.
type geminiGrounding struct {
	WebSearchQueries []string `json:"webSearchQueries"`
	GroundingChunks  []struct {
		Web *struct {
			URI   string `json:"uri"`
			Title string `json:"title"`
		} `json:"web"`
	} `json:"groundingChunks"`
	GroundingSupports []struct {
		GroundingChunkIndices []int `json:"groundingChunkIndices"`
	} `json:"groundingSupports"`
}

// geminiWebSearch reads the search count and citations of a turn.
//
// With includeServerSideToolInvocations on (invocations), only the toolCall parts
// count. They are the searches of this turn. groundingMetadata can't be used: it is
// missing on a turn that searches and then calls a function, and on the next round
// it repeats the earlier queries though no new search ran (both seen live).
//
// With it off there are no earlier search steps to repeat, so groundingMetadata is
// the count. The larger of the two only matters for a batch sent before v0.15.0,
// whose key does not say that the setting was on.
//
// Empty queries are not billed and are not counted.
//
// Only pages a support refers to are citations: the others were read but not used.
// CitedText stays empty, because a support's text is the model's own sentence, not
// text from the page.
func geminiWebSearch(parts []geminiPart, grounding []*geminiGrounding, invocations bool) (int, []Citation) {
	fromParts := 0
	for _, p := range parts {
		if len(p.ToolCall) == 0 {
			continue
		}
		var call struct {
			ToolType string `json:"toolType"`
			Args     struct {
				Queries []string `json:"queries"`
			} `json:"args"`
		}
		if json.Unmarshal(p.ToolCall, &call) == nil && call.ToolType == "GOOGLE_SEARCH_WEB" {
			fromParts += countQueries(call.Args.Queries)
		}
	}
	fromMeta := 0
	var citations []Citation
	for _, g := range grounding {
		if g == nil {
			continue
		}
		fromMeta = max(fromMeta, countQueries(g.WebSearchQueries))
		for _, s := range g.GroundingSupports {
			for _, i := range s.GroundingChunkIndices {
				if i < 0 || i >= len(g.GroundingChunks) || g.GroundingChunks[i].Web == nil {
					continue
				}
				web := g.GroundingChunks[i].Web
				citations = append(citations, Citation{URL: web.URI, Title: web.Title})
			}
		}
	}
	if invocations {
		return fromParts, dedupeCitations(citations)
	}
	return max(fromParts, fromMeta), dedupeCitations(citations)
}

func countQueries(queries []string) int {
	n := 0
	for _, q := range queries {
		if strings.TrimSpace(q) != "" {
			n++
		}
	}
	return n
}

// flashThinkingConfigs are the "as little thinking as possible" configs for
// flash-class models, tried in order: the current API's thinkingLevel first, the
// 2.5-era thinkingBudget for pinned older models, then none (let it think) as the
// last resort so a future knob change degrades to a working (if truncable) call.
func flashThinkingConfigs() []*geminiThinkingConfig {
	zero := 0
	return []*geminiThinkingConfig{
		{ThinkingLevel: "minimal"},
		{ThinkingBudget: &zero},
		nil,
	}
}

// geminiBody builds the request body up to (not including) the thinking config:
// max output tokens, web search, system, contents, function declarations and the
// JSON schema. The live path and the batch path both use it.
func geminiBody(model string, maxTokens int, req Request) (geminiRequest, error) {
	body := geminiRequest{}
	body.GenerationConfig.MaxOutputTokens = maxTokens

	// Web search → Gemini's native Google Search grounding (it runs the search itself and
	// grounds the answer). No per-request UserLocation knob like Anthropic's; the prompt
	// carries the Indonesia context. Cheaper than Anthropic web search — the recipe cost lever.
	if req.WebSearch {
		body.Tools = []geminiTool{{GoogleSearch: &geminiGoogleSearch{}}}
	}

	// System: the cacheable prefix + any "system" turns.
	system := req.SystemCacheable
	for _, m := range req.Messages {
		if m.Role == "system" {
			if system != "" {
				system += "\n\n"
			}
			system += m.Content
		}
	}
	if system != "" {
		body.SystemInstruction = &geminiContent{Parts: []geminiPart{{Text: system}}}
	}

	contents, err := geminiContents(req)
	if err != nil {
		return geminiRequest{}, err
	}
	body.Contents = contents
	if decls := geminiFunctionDecls(req.Tools); len(decls) > 0 {
		body.Tools = append(body.Tools, geminiTool{FunctionDeclarations: decls})
		if req.WebSearch {
			body.ToolConfig = &geminiToolConfig{IncludeServerSideToolInvocations: true}
		}
	}
	// A last round can drop the function tools but still echo search steps. Its
	// groundingMetadata repeats those queries, so keep the setting on: then only a
	// real new search shows up as a toolCall part (see geminiWebSearch).
	if req.WebSearch && body.ToolConfig == nil && geminiEchoesSearch(contents) {
		body.ToolConfig = &geminiToolConfig{IncludeServerSideToolInvocations: true}
	}
	if err := geminiApplySchema(&body, model, req); err != nil {
		return geminiRequest{}, err
	}
	return body, nil
}

func (g *googleProvider) complete(ctx context.Context, model string, maxTokens int, req Request) (Response, error) {
	body, err := geminiBody(model, maxTokens, req)
	if err != nil {
		return Response{}, err
	}

	// Flash-class models fill the cheap/fast profile slots — suppress their default
	// thinking (it spends maxOutputTokens and truncated replies to ~20 tokens). The
	// suppression knob differs per generation, so walk the candidates; a 400 means
	// "this knob doesn't exist on this model", anything else is a real failure.
	attempts := []*geminiThinkingConfig{nil}
	if strings.Contains(model, "flash") {
		attempts = flashThinkingConfigs()
	}
	var lastErr error
	for _, tc := range attempts {
		body.GenerationConfig.ThinkingConfig = tc
		var resp Response
		var status int
		var err error
		if req.OnEvent != nil {
			resp, status, err = g.sendStream(ctx, model, body, req.OnEvent, func() int { return estimateRequestTokens(req) })
		} else {
			resp, status, err = g.send(ctx, model, body)
		}
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if status != http.StatusBadRequest {
			// Not the "unsupported knob" 400 we retry. Return resp as-is: a refusal
			// (200) arrives here with a POPULATED resp (partial text + usage), while a
			// transport error's resp is already the zero value — either is correct.
			return resp, err
		}
	}
	return Response{}, lastErr
}

// geminiEchoesSearch reports whether a history holds a server-side tool step.
func geminiEchoesSearch(contents []geminiContent) bool {
	for _, c := range contents {
		for _, p := range c.Parts {
			if len(p.ToolCall) > 0 {
				return true
			}
		}
	}
	return false
}

// geminiContents converts the request's turns into Gemini contents, resolving
// each round's tool results against the calls that preceded them. Pure and
// separate from complete so the resolution rules — the part that fails silently
// and misattributes results — are testable without an HTTP round trip.
func geminiContents(req Request) ([]geminiContent, error) {
	var out []geminiContent
	// prevCalls tracks the most recent preceding assistant turn's ToolCalls, the
	// only source of a result's function name and position — Gemini's
	// synthesized IDs carry neither. See geminiToolResponses. It is cleared as
	// soon as a user turn consumes it.
	var prevCalls []ToolCall
	for _, m := range req.Messages {
		if m.Role == "system" {
			continue
		}
		role := "user"
		if m.Role == "assistant" {
			role = "model"
		}
		if m.Role == "assistant" {
			if echo, ok := geminiEchoParts(m); ok {
				out = append(out, geminiContent{Role: role, Parts: echo})
				prevCalls = m.ToolCalls
				continue
			}
		}
		parts := make([]geminiPart, 0, 1+len(m.Images)+len(m.Documents))
		for _, img := range m.Images {
			parts = append(parts, geminiPart{InlineData: &geminiInline{MimeType: img.MediaType, Data: img.Base64}})
		}
		for _, doc := range m.Documents {
			parts = append(parts, geminiPart{InlineData: &geminiInline{MimeType: doc.MediaType, Data: doc.Base64}})
		}
		if m.Content != "" {
			parts = append(parts, geminiPart{Text: m.Content})
		}
		if m.Role == "assistant" {
			for i, tc := range m.ToolCalls {
				var args map[string]any
				_ = json.Unmarshal(normalizeToolInput(tc.Input), &args)
				part := geminiPart{FunctionCall: &geminiFunctionCall{Name: tc.Name, Args: args}}
				if i == 0 {
					// The real signature is gone; Gemini 3 checks the first call of a turn.
					part.ThoughtSignature = geminiSkipSignature
				}
				parts = append(parts, part)
			}
			prevCalls = m.ToolCalls
		} else {
			respParts, err := geminiToolResponses(m.ToolResults, prevCalls)
			if err != nil {
				return nil, err
			}
			parts = append(parts, respParts...)
			// A round is answered exactly ONCE, so the calls are consumed here.
			// Leaving them set breaks two ordinary histories: a plain-text user turn
			// following an answered round would be checked against calls that already
			// have their results and rejected with a spurious "missing tool result",
			// and a round accidentally appended twice would validate against the
			// stale list and emit a SECOND turn answering the same call — the
			// misattribution this guard exists to prevent. Cleared, the duplicate
			// instead fails loudly as a result naming an unknown call.
			prevCalls = nil
		}
		out = append(out, geminiContent{Role: role, Parts: parts})
	}
	return out, nil
}

// send posts one generateContent request and parses the reply. The HTTP status is
// returned alongside the error so callers can distinguish an unsupported knob (400)
// from real failures.
func (g *googleProvider) send(ctx context.Context, model string, body geminiRequest) (Response, int, error) {
	ctx, cancel := withDefaultDeadline(ctx, g.timeout, body.GenerationConfig.MaxOutputTokens)
	defer cancel()
	buf, err := json.Marshal(body)
	if err != nil {
		return Response{}, 0, err
	}
	url := fmt.Sprintf("%s/models/%s:generateContent", g.baseURL, geminiModelPath(model))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		return Response{}, 0, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", g.apiKey)

	res, err := g.http.Do(httpReq)
	if err != nil {
		return Response{}, 0, transportError(ctx, ProviderGoogle, err)
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return Response{}, res.StatusCode, transportError(ctx, ProviderGoogle, readError(ctx, "gemini", err))
	}

	var out geminiResponse
	jerr := json.Unmarshal(raw, &out)
	if res.StatusCode >= 300 || (jerr == nil && out.Error != nil) {
		return Response{}, res.StatusCode, geminiStatusError(res.StatusCode, raw, out)
	}
	if jerr != nil {
		return Response{}, res.StatusCode, fmt.Errorf("gemini: decode response: %w", jerr)
	}

	resp, err := geminiBuildResponse(model, out, body.ToolConfig != nil)
	return resp, res.StatusCode, err
}

// geminiStatusError is the error for a non-2xx reply, or a 2xx one with an error
// body, whose status is then the body's own code. out is the decoded body, zero
// when it was not JSON; the raw body is the message then.
func geminiStatusError(status int, raw []byte, out geminiResponse) error {
	msg := strings.TrimSpace(string(raw))
	if out.Error != nil {
		msg = out.Error.Message
		if status < 300 {
			status = out.Error.Code
		}
	}
	return statusError(ProviderGoogle, status, fmt.Errorf("gemini: %s (status %d)", msg, status))
}

// geminiBuildResponse is the part of a reply shared by the plain and streamed paths:
// text (thought parts left out), tool calls, stop reason, usage and the refusal check.
func geminiBuildResponse(model string, out geminiResponse, invocations bool) (Response, error) {
	var text strings.Builder
	var toolCalls []ToolCall
	var stopReason StopReason
	if len(out.Candidates) > 0 {
		for _, p := range out.Candidates[0].Content.Parts {
			if !p.Thought {
				text.WriteString(p.Text)
			}
		}
		toolCalls = geminiToolCalls(out.Candidates[0].Content.Parts)
		stopReason = geminiStopReason(out.Candidates[0].FinishReason, len(toolCalls) > 0)
	}
	cached := out.UsageMetadata.CachedContentTokenCount
	input := out.UsageMetadata.PromptTokenCount - cached // exclude cache from input (Anthropic-style)
	if input < 0 {
		input = 0
	}
	resp := Response{
		Text:        text.String(),
		InputTokens: input,
		// Thought tokens are billed as output — count them so the usage ledger's
		// cost math stays honest when a Gemini model does think.
		OutputTokens: out.UsageMetadata.CandidatesTokenCount + out.UsageMetadata.ThoughtsTokenCount,
		CachedTokens: cached,
		ToolCalls:    toolCalls,
		StopReason:   stopReason,
	}
	if len(out.Candidates) > 0 {
		c := out.Candidates[0]
		resp.ProviderData = geminiTurnData(c.Content.Parts)
		resp.WebSearchRequests, resp.Citations = geminiWebSearch(c.Content.Parts, []*geminiGrounding{c.GroundingMetadata}, invocations)
	}
	// A safety classifier decline is a 200 with no usable content — turn it into an
	// error so the caller doesn't ship an empty answer, and let the router fail it
	// over like Anthropic's refusal. Any partial text + usage ride along.
	if cat := geminiRefusalCategory(out); cat != "" {
		return resp, &RefusalError{Provider: ProviderGoogle, Model: model, Category: cat}
	}
	return resp, nil
}

// streams marks this provider as able to send text to Request.OnEvent as it arrives.
func (g *googleProvider) streams() bool { return true }

var _ streamer = (*googleProvider)(nil)

// sendStream is send for a request with OnEvent: the same request on
// streamGenerateContent, read as server-sent events. Text goes to onEvent as it
// arrives; function calls come whole and are collected. The events are folded into one
// geminiResponse so geminiBuildResponse shapes the result exactly as on the plain path.
// The status is returned as in send, so the caller's 400 retry still works: a 400 comes
// before any body, so nothing has been sent when the next attempt starts.
// If the stream ends early, the Response carries the text and tokens used so far
// (see partial below). inputGuess is called only then, when Gemini sent no usage.
func (g *googleProvider) sendStream(ctx context.Context, model string, body geminiRequest, onEvent func(StreamEvent), inputGuess func() int) (Response, int, error) {
	ctx, cancel := withDefaultDeadline(ctx, g.timeout, body.GenerationConfig.MaxOutputTokens)
	defer cancel()
	buf, err := json.Marshal(body)
	if err != nil {
		return Response{}, 0, err
	}
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse", g.baseURL, geminiModelPath(model))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		return Response{}, 0, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", g.apiKey)

	res, err := g.http.Do(httpReq)
	if err != nil {
		return Response{}, 0, transportError(ctx, ProviderGoogle, err)
	}
	defer res.Body.Close()
	if res.StatusCode >= 300 {
		raw, err := io.ReadAll(res.Body)
		if err != nil {
			return Response{}, res.StatusCode, transportError(ctx, ProviderGoogle, readError(ctx, "gemini", err))
		}
		var out geminiResponse
		_ = json.Unmarshal(raw, &out)
		return Response{}, res.StatusCode, geminiStatusError(res.StatusCode, raw, out)
	}

	var (
		text         strings.Builder
		calls        []geminiPart
		ids          []string // ids[i] is the ID announced for calls[i]
		finishReason string
		sawCandidate bool
		usage        = geminiResponse{}.UsageMetadata
		feedback     = geminiResponse{}.PromptFeedback
		thought      strings.Builder
		seen         bool         // a payload decoded without an error field
		turn         []geminiPart // every part in arrival order, text pieces joined: the turn to echo
		grounding    []*geminiGrounding
	)
	// partial is the reply for an early end: text and tokens only, no tool calls.
	// It is zero until one good payload has arrived. Real usage is used when Gemini
	// sent it; otherwise the numbers are guessed and UsageEstimated is set.
	partial := func() Response {
		if !seen {
			return Response{}
		}
		r := Response{Text: text.String()}
		r.WebSearchRequests, r.Citations = geminiWebSearch(turn, grounding, body.ToolConfig != nil)
		if usage != (geminiResponse{}.UsageMetadata) {
			r.CachedTokens = usage.CachedContentTokenCount
			r.InputTokens = usage.PromptTokenCount - usage.CachedContentTokenCount
		} else {
			r.InputTokens = inputGuess()
			r.UsageEstimated = true
		}
		if usage.CandidatesTokenCount > 0 {
			r.OutputTokens = usage.CandidatesTokenCount + usage.ThoughtsTokenCount
		} else {
			guess := text.String() + thought.String()
			for _, c := range calls {
				if b, err := json.Marshal(c.FunctionCall.Args); err == nil {
					guess += string(b)
				}
			}
			r.OutputTokens = max(estimateTokens(guess), usage.ThoughtsTokenCount)
			r.UsageEstimated = true
		}
		return r
	}
	sse := newSSEReader(res.Body)
	for {
		payload, err := sse.next()
		if err == io.EOF {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return partial(), res.StatusCode, readError(ctx, "gemini", ctxErr)
			}
			// Without a finishReason the reply was cut off; a prompt-level block has none
			// by design and is reported as the refusal it is.
			blocked := feedback != nil && geminiBlockReasons[feedback.BlockReason]
			if finishReason == "" && !blocked {
				return partial(), res.StatusCode, transportError(ctx, ProviderGoogle,
					fmt.Errorf("gemini: stream ended without finishReason: %w", io.ErrUnexpectedEOF))
			}
			break
		}
		if err != nil {
			return partial(), res.StatusCode, transportError(ctx, ProviderGoogle, readError(ctx, "gemini", err))
		}
		if len(bytes.TrimSpace(payload)) == 0 {
			continue
		}
		var ev geminiResponse
		if err := json.Unmarshal(payload, &ev); err != nil {
			return partial(), res.StatusCode, fmt.Errorf("gemini: decode stream event: %w", err)
		}
		if ev.Error != nil {
			// The stream was accepted, so the request was fine: without a code of its own
			// the error is the provider's (Status 0, Retryable).
			return partial(), res.StatusCode, statusError(ProviderGoogle, ev.Error.Code,
				fmt.Errorf("gemini: %s (stream)", ev.Error.Message))
		}
		seen = true
		if ev.UsageMetadata != (geminiResponse{}.UsageMetadata) {
			usage = ev.UsageMetadata
		}
		if ev.PromptFeedback != nil {
			feedback = ev.PromptFeedback
		}
		if len(ev.Candidates) == 0 {
			continue
		}
		sawCandidate = true
		c := ev.Candidates[0]
		if c.GroundingMetadata != nil {
			grounding = append(grounding, c.GroundingMetadata)
		}
		for _, p := range c.Content.Parts {
			// Join only unsigned text: a signature belongs to its part as sent.
			if n := len(turn); n > 0 && p.plainText() && turn[n-1].plainText() && turn[n-1].Thought == p.Thought {
				turn[n-1].Text += p.Text
				turn[n-1].raw = nil // rebuilt from the fields, now with the joined text
			} else {
				turn = append(turn, p)
			}
			switch {
			case p.FunctionCall != nil:
				id := geminiCallID()
				calls = append(calls, p)
				ids = append(ids, id)
				// Function calls arrive whole, so start and ready go out together.
				for _, kind := range []StreamKind{StreamToolStart, StreamToolReady} {
					onEvent(StreamEvent{Kind: kind, ToolCallID: id, ToolName: p.FunctionCall.Name})
					if ctxErr := ctx.Err(); ctxErr != nil {
						return partial(), res.StatusCode, readError(ctx, "gemini", ctxErr)
					}
				}
			case p.Text != "" && p.Thought:
				thought.WriteString(p.Text)
			case p.Text != "":
				text.WriteString(p.Text)
				onEvent(StreamEvent{Kind: StreamText, Text: p.Text})
				// A callback that cancels ctx must stop the stream even when the next
				// events are already buffered and would be read without touching the network.
				if ctxErr := ctx.Err(); ctxErr != nil {
					return partial(), res.StatusCode, readError(ctx, "gemini", ctxErr)
				}
			}
		}
		if c.FinishReason != "" {
			finishReason = c.FinishReason
		}
	}

	synthetic := geminiResponse{PromptFeedback: feedback}
	synthetic.UsageMetadata = usage
	if sawCandidate {
		synthetic.Candidates = append(synthetic.Candidates, struct {
			Content           geminiContent    `json:"content"`
			FinishReason      string           `json:"finishReason"`
			GroundingMetadata *geminiGrounding `json:"groundingMetadata"`
		}{
			Content:      geminiContent{Parts: append([]geminiPart{{Text: text.String()}}, calls...)},
			FinishReason: finishReason,
		})
	}
	resp, err := geminiBuildResponse(model, synthetic, body.ToolConfig != nil)
	// Give the calls the IDs already sent in events. The synthetic candidate holds
	// the call parts in arrival order and geminiToolCalls keeps that order; the
	// length check only guards that assumption.
	if len(resp.ToolCalls) == len(ids) {
		for i := range ids {
			resp.ToolCalls[i].ID = ids[i]
		}
	}
	resp.ProviderData = geminiTurnData(turn)
	resp.WebSearchRequests, resp.Citations = geminiWebSearch(turn, grounding, body.ToolConfig != nil)
	return resp, res.StatusCode, err
}

// geminiPreJSONToolsModel reports a model from before Gemini 3, which cannot combine
// JSON mode with tools (function calling or Google Search grounding); Gemini 3 can.
// Only explicitly versioned 1.x/2.x ids count: a -latest alias is assumed current,
// and if Google has it pointing at an older model the API's 400 says so.
func geminiPreJSONToolsModel(model string) bool {
	return strings.HasPrefix(model, "gemini-1.") || strings.HasPrefix(model, "gemini-2.")
}

// geminiApplySchema puts Request.JSONSchema on the request body. It is set once,
// before the thinking-config retry loop, so every attempt carries it.
//
// On a pre-3 model the API rejects JSON mode alongside tools, so:
//   - with WebSearch it is an error: dropping either the schema or the grounding
//     would silently answer a different question than the caller asked;
//   - with function Tools the schema is left off for that round. A tool round's
//     answer is calls, not the final JSON; the router still checks any text reply
//     is valid JSON (see completeOn), so a non-JSON final answer is an error, not
//     a silent pass.
func geminiApplySchema(body *geminiRequest, model string, req Request) error {
	if len(req.JSONSchema) == 0 {
		return nil
	}
	if geminiPreJSONToolsModel(model) {
		if req.WebSearch {
			return fmt.Errorf("gemini: %s cannot combine JSONSchema with WebSearch; use a Gemini 3 model", model)
		}
		if len(req.Tools) > 0 {
			return nil
		}
	}
	body.GenerationConfig.ResponseFormat = &geminiResponseFormat{Text: &geminiTextFormat{
		MimeType: "APPLICATION_JSON",
		Schema:   geminiJSONSchema(req.JSONSchema),
	}}
	return nil
}

// geminiSchemaKeywords is the JSON Schema subset Gemini documents as supported
// (the list sits under the older responseJsonSchema field), plus Google's
// non-standard propertyOrdering. Anything else is stripped rather than sent, so a
// caller's schema written for Claude never has to change to reach Gemini.
var geminiSchemaKeywords = map[string]bool{
	"$id": true, "$defs": true, "$ref": true, "$anchor": true,
	"type": true, "format": true, "title": true, "description": true, "enum": true,
	"items": true, "prefixItems": true, "minItems": true, "maxItems": true,
	"minimum": true, "maximum": true, "anyOf": true, "oneOf": true,
	"properties": true, "additionalProperties": true, "required": true,
	"propertyOrdering": true,
}

// geminiJSONSchema returns a copy of a schema reduced to geminiSchemaKeywords,
// recursing into every subschema position. "const" becomes a one-value "enum",
// which says the same thing in a keyword Gemini accepts. The caller's map is
// never mutated.
func geminiJSONSchema(schema map[string]any) map[string]any {
	out := make(map[string]any, len(schema))
	for k, v := range schema {
		switch {
		case k == "const":
			if _, has := schema["enum"]; !has {
				out["enum"] = []any{v}
			}
		case !geminiSchemaKeywords[k]:
			// dropped
		case k == "properties" || k == "$defs":
			out[k] = geminiSchemaByName(v)
		case k == "items" || k == "additionalProperties":
			out[k] = geminiSubschema(v) // a schema, or a bool left as-is
		case k == "prefixItems" || k == "anyOf" || k == "oneOf":
			out[k] = geminiSchemaList(v)
		default:
			out[k] = v
		}
	}
	return out
}

// geminiSubschema reduces v when it is a schema object and returns it unchanged
// otherwise (a bool additionalProperties, say).
func geminiSubschema(v any) any {
	if m, ok := v.(map[string]any); ok {
		return geminiJSONSchema(m)
	}
	return v
}

// geminiSchemaByName reduces a NAME -> subschema map (properties, $defs). The
// names are not keywords, so every one is kept.
func geminiSchemaByName(v any) any {
	m, ok := v.(map[string]any)
	if !ok {
		return v
	}
	out := make(map[string]any, len(m))
	for name, s := range m {
		out[name] = geminiSubschema(s)
	}
	return out
}

// geminiSchemaList reduces a list of subschemas (anyOf, oneOf, prefixItems),
// whether it arrived as decoded JSON or as a Go literal.
func geminiSchemaList(v any) any {
	switch list := v.(type) {
	case []any:
		out := make([]any, len(list))
		for i, s := range list {
			out[i] = geminiSubschema(s)
		}
		return out
	case []map[string]any:
		out := make([]any, len(list))
		for i, s := range list {
			out[i] = geminiJSONSchema(s)
		}
		return out
	default:
		return v
	}
}

// geminiModelPath makes a model name safe as one URL path segment: it drops a
// "models/" prefix and escapes the rest (so "tuned/x y" cannot add a segment).
func geminiModelPath(model string) string {
	return url.PathEscape(strings.TrimPrefix(model, "models/"))
}
