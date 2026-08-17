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

// openaiProvider is an OpenAI-compatible chat-completions provider (raw HTTP). The
// same wire format serves OpenAI and OpenAI-compatible backends (DeepSeek, Together,
// Groq, OpenRouter, local) — point baseURL at the vendor. Text + images; PDFs aren't
// sent (pre-extract to text). Usage: prompt/completion tokens, with cached prompt
// tokens split out so cost matches the Anthropic accounting (input excludes cache).
type openaiProvider struct {
	apiKey  string
	baseURL string
	http    *http.Client
}

func newOpenAI(apiKey, baseURL string) provider {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	return &openaiProvider{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 120 * time.Second},
	}
}

type oaiMessage struct {
	Role       string        `json:"role"`
	Content    any           `json:"content"` // string, or []oaiPart for multimodal
	ToolCalls  []oaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

type oaiToolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON encoded as a string
}

type oaiToolCall struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Function oaiToolCallFunc `json:"function"`
}

type oaiToolFunc struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type oaiTool struct {
	Type     string      `json:"type"` // always "function"
	Function oaiToolFunc `json:"function"`
}

// oaiTools maps declarations onto the wire shape.
func oaiTools(defs []ToolDef) []oaiTool {
	if len(defs) == 0 {
		return nil
	}
	out := make([]oaiTool, 0, len(defs))
	for _, d := range defs {
		out = append(out, oaiTool{
			Type: "function",
			Function: oaiToolFunc{
				Name: d.Name, Description: d.Description, Parameters: d.Schema,
			},
		})
	}
	return out
}

// oaiToolCalls maps a reply's tool calls. Arguments arrive as a JSON *string*,
// so the bytes go straight into Input.
func oaiToolCalls(in []oaiToolCall) []ToolCall {
	var out []ToolCall
	for _, c := range in {
		out = append(out, ToolCall{
			ID: c.ID, Name: c.Function.Name,
			// A backend that returns `"arguments": ""` for a no-arg call would
			// otherwise put an EMPTY json.RawMessage into the history, which makes
			// json.Marshal of every later request fail outright.
			Input: normalizeToolInput(json.RawMessage(c.Function.Arguments)),
		})
	}
	return out
}

// oaiStopReason maps finish_reason. An unknown or absent value reads as a
// completed turn rather than inventing a failure.
func oaiStopReason(fr string) StopReason {
	switch fr {
	case "tool_calls":
		return StopToolUse
	case "length":
		return StopTruncated
	default:
		return StopEndTurn
	}
}

// oaiRefusal reports a safety decline from a choice: finish_reason "content_filter",
// or a populated structured `refusal` field (GPT-4o+). It returns the category and a
// human-readable explanation, or ("", "") for a normal completion. DeepSeek/Moonshot
// don't emit either field, so they never trigger.
func oaiRefusal(finishReason, refusal string) (category, explanation string) {
	if finishReason == "content_filter" {
		return "content_filter", refusal
	}
	if refusal != "" {
		return "refusal", refusal
	}
	return "", ""
}

type oaiPart struct {
	Type     string       `json:"type"`
	Text     string       `json:"text,omitempty"`
	ImageURL *oaiImageURL `json:"image_url,omitempty"`
}

type oaiImageURL struct {
	URL string `json:"url"`
}

type oaiRequest struct {
	Model string `json:"model"`
	// Newer OpenAI models (GPT-5+ and the o-series) REPLACED max_tokens with
	// max_completion_tokens and reject the old name; older OpenAI (gpt-4o) and the
	// OpenAI-COMPATIBLE providers (DeepSeek/Moonshot) still use max_tokens. Exactly one is
	// set per request by usesMaxCompletionTokens(model).
	MaxTokens           int          `json:"max_tokens,omitempty"`
	MaxCompletionTokens int          `json:"max_completion_tokens,omitempty"`
	Messages            []oaiMessage `json:"messages"`
	Tools               []oaiTool    `json:"tools,omitempty"`
}

// usesMaxCompletionTokens reports whether a model wants max_completion_tokens (GPT-5+/o-series)
// instead of max_tokens. DeepSeek/Moonshot/gpt-4o are unaffected (they keep max_tokens).
func usesMaxCompletionTokens(model string) bool {
	for _, p := range []string{"gpt-5", "gpt-6", "o1", "o3", "o4"} {
		if strings.HasPrefix(model, p) {
			return true
		}
	}
	return false
}

type oaiResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
			// Refusal is OpenAI's structured safety decline (GPT-4o+): when the model
			// refuses, this carries the message and Content is empty.
			Refusal   string        `json:"refusal"`
			ToolCalls []oaiToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// oaiToolResultContent renders one result's content for a "tool" message.
// OpenAI's tool message carries no is_error field, so a failed result is marked
// in the text — otherwise the model is told nothing and cannot adapt, which is
// the behaviour ToolResult.IsError promises on the other two providers.
func oaiToolResultContent(tr ToolResult) string {
	if !tr.IsError {
		return tr.Content
	}
	return "Error: " + tr.Content
}

// oaiMessages converts the request's turns into the wire message list. Pure, so
// the message SHAPE (which turns exist, and in what order) is testable without
// an HTTP round trip — the ordering rule around tool results is the thing that
// breaks silently.
func oaiMessages(req Request) []oaiMessage {
	msgs := make([]oaiMessage, 0, len(req.Messages)+1)
	if req.SystemCacheable != "" {
		msgs = append(msgs, oaiMessage{Role: "system", Content: req.SystemCacheable})
	}
	for _, m := range req.Messages {
		var toolCalls []oaiToolCall
		if len(m.ToolCalls) > 0 {
			toolCalls = make([]oaiToolCall, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				toolCalls = append(toolCalls, oaiToolCall{
					ID: tc.ID, Type: "function",
					// normalizeToolInput: an empty Arguments string is not valid JSON,
					// and a nil Input would send "" — an echoed no-arg call stays `{}`.
					Function: oaiToolCallFunc{Name: tc.Name, Arguments: string(normalizeToolInput(tc.Input))},
				})
			}
		}
		// The base message is emitted only when it carries something of its own.
		// The prescribed shape for answering a round — Message{Role:"user",
		// ToolResults: results} with no Content — would otherwise wedge an empty
		// user message between the assistant turn's tool_calls and the "tool"
		// messages below. OpenAI requires each "tool" message to IMMEDIATELY follow
		// the assistant turn whose tool_calls it answers, so that empty turn fails
		// every second-round call. (Anthropic's adapter guards the same way, via
		// its `len(blocks) > 0` checks.)
		//
		// A message with nothing at all AND no results is still emitted verbatim,
		// so a request that declares no tools serialises exactly as it did before.
		if m.Content != "" || len(m.Images) > 0 || len(toolCalls) > 0 || len(m.ToolResults) == 0 {
			if len(m.Images) == 0 {
				msgs = append(msgs, oaiMessage{Role: m.Role, Content: m.Content, ToolCalls: toolCalls})
			} else {
				// Multimodal: content becomes an array of text + image parts.
				parts := make([]oaiPart, 0, 1+len(m.Images))
				if m.Content != "" {
					parts = append(parts, oaiPart{Type: "text", Text: m.Content})
				}
				for _, img := range m.Images {
					parts = append(parts, oaiPart{
						Type:     "image_url",
						ImageURL: &oaiImageURL{URL: fmt.Sprintf("data:%s;base64,%s", img.MediaType, img.Base64)},
					})
				}
				msgs = append(msgs, oaiMessage{Role: m.Role, Content: parts, ToolCalls: toolCalls})
			}
		}
		// Every ToolResult from one round fans out into its own "tool" message —
		// the wire format wants one message per result, unlike Anthropic's
		// single-block-per-result-in-one-turn shape.
		//
		// The "tool" message has NO is_error field, so ToolResult.IsError cannot be
		// carried structurally here. Dropping it silently would break the contract
		// that the model sees a failure and can adapt, so the flag is rendered into
		// the content instead — the conventional workaround. Documented on
		// ToolResult.IsError and in the README.
		for _, tr := range m.ToolResults {
			msgs = append(msgs, oaiMessage{
				Role: "tool", ToolCallID: tr.ToolCallID, Content: oaiToolResultContent(tr),
			})
		}
	}
	return msgs
}

func (o *openaiProvider) complete(ctx context.Context, model string, maxTokens int, req Request) (Response, error) {
	oaiReq := oaiRequest{Model: model, Messages: oaiMessages(req), Tools: oaiTools(req.Tools)}
	if usesMaxCompletionTokens(model) {
		oaiReq.MaxCompletionTokens = maxTokens
	} else {
		oaiReq.MaxTokens = maxTokens
	}
	buf, err := json.Marshal(oaiReq)
	if err != nil {
		return Response{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return Response{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+o.apiKey)

	res, err := o.http.Do(httpReq)
	if err != nil {
		return Response{}, err
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(res.Body)

	var out oaiResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return Response{}, fmt.Errorf("openai: decode response: %w", err)
	}
	if res.StatusCode >= 300 || out.Error != nil {
		msg := strings.TrimSpace(string(raw))
		if out.Error != nil {
			msg = out.Error.Message
		}
		return Response{}, fmt.Errorf("openai: %s (status %d)", msg, res.StatusCode)
	}

	var text string
	var toolCalls []ToolCall
	var stopReason StopReason
	if len(out.Choices) > 0 {
		text = out.Choices[0].Message.Content
		toolCalls = oaiToolCalls(out.Choices[0].Message.ToolCalls)
		stopReason = oaiStopReason(out.Choices[0].FinishReason)
	}
	cached := out.Usage.PromptTokensDetails.CachedTokens
	input := out.Usage.PromptTokens - cached // exclude cache from input (Anthropic-style)
	if input < 0 {
		input = 0
	}
	resp := Response{
		Text:         text,
		InputTokens:  input,
		OutputTokens: out.Usage.CompletionTokens,
		CachedTokens: cached,
		ToolCalls:    toolCalls,
		StopReason:   stopReason,
	}
	// A safety decline is a 200 with no usable answer — surface it as an error so the
	// caller doesn't ship an empty string, and let it fail over like the other
	// providers. Provider is stamped OpenAI here; the router sets the authoritative
	// model attribution on the Response.
	if len(out.Choices) > 0 {
		if cat, expl := oaiRefusal(out.Choices[0].FinishReason, out.Choices[0].Message.Refusal); cat != "" {
			return resp, &RefusalError{Provider: ProviderOpenAI, Model: model, Category: cat, Explanation: expl}
		}
	}
	return resp, nil
}
