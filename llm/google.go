package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// googleProvider is the Google Gemini provider (generateContent REST API, raw HTTP —
// no SDK, matching the repo's hand-rolled-HTTP idiom for non-Anthropic backends).
// Gemini uses roles "user"/"model" and a separate system_instruction; usage comes back
// in usageMetadata (prompt / candidates / cached token counts).
type googleProvider struct {
	apiKey string
	http   *http.Client
}

func newGoogle(apiKey string) provider {
	return &googleProvider{apiKey: apiKey, http: &http.Client{Timeout: 120 * time.Second}}
}

type geminiPart struct {
	Text       string        `json:"text,omitempty"`
	InlineData *geminiInline `json:"inline_data,omitempty"`
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
	SystemInstruction *geminiContent  `json:"system_instruction,omitempty"`
	Contents          []geminiContent `json:"contents"`
	Tools             []geminiTool    `json:"tools,omitempty"`
	GenerationConfig  struct {
		MaxOutputTokens int                   `json:"maxOutputTokens,omitempty"`
		ThinkingConfig  *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
	} `json:"generationConfig"`
}

// geminiTool carries the built-in tools. GoogleSearch enables Gemini's native Google
// Search grounding — the cross-provider equivalent of Anthropic's web-search tool
// (req.WebSearch), and much cheaper for the recipe web search. Empty object payload.
type geminiTool struct {
	GoogleSearch *geminiGoogleSearch `json:"google_search,omitempty"`
}

type geminiGoogleSearch struct{}

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
		Content geminiContent `json:"content"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount"`
		ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	} `json:"usageMetadata"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
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

func (g *googleProvider) complete(ctx context.Context, model string, maxTokens int, req Request) (Response, error) {
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

	for _, m := range req.Messages {
		if m.Role == "system" {
			continue
		}
		role := "user"
		if m.Role == "assistant" {
			role = "model"
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
		body.Contents = append(body.Contents, geminiContent{Role: role, Parts: parts})
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
		resp, status, err := g.send(ctx, model, body)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if status != http.StatusBadRequest {
			return Response{}, err
		}
	}
	return Response{}, lastErr
}

// send posts one generateContent request and parses the reply. The HTTP status is
// returned alongside the error so callers can distinguish an unsupported knob (400)
// from real failures.
func (g *googleProvider) send(ctx context.Context, model string, body geminiRequest) (Response, int, error) {
	buf, err := json.Marshal(body)
	if err != nil {
		return Response{}, 0, err
	}
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent", model)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		return Response{}, 0, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", g.apiKey)

	res, err := g.http.Do(httpReq)
	if err != nil {
		return Response{}, 0, err
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(res.Body)

	var out geminiResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return Response{}, res.StatusCode, fmt.Errorf("gemini: decode response: %w", err)
	}
	if res.StatusCode >= 300 || out.Error != nil {
		msg := strings.TrimSpace(string(raw))
		if out.Error != nil {
			msg = out.Error.Message
		}
		return Response{}, res.StatusCode, fmt.Errorf("gemini: %s (status %d)", msg, res.StatusCode)
	}

	var text strings.Builder
	if len(out.Candidates) > 0 {
		for _, p := range out.Candidates[0].Content.Parts {
			text.WriteString(p.Text)
		}
	}
	cached := out.UsageMetadata.CachedContentTokenCount
	input := out.UsageMetadata.PromptTokenCount - cached // exclude cache from input (Anthropic-style)
	if input < 0 {
		input = 0
	}
	return Response{
		Text:        text.String(),
		InputTokens: input,
		// Thought tokens are billed as output — count them so the usage ledger's
		// cost math stays honest when a Gemini model does think.
		OutputTokens: out.UsageMetadata.CandidatesTokenCount + out.UsageMetadata.ThoughtsTokenCount,
		CachedTokens: cached,
	}, res.StatusCode, nil
}
