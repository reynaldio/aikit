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
	Role    string `json:"role"`
	Content any    `json:"content"` // string, or []oaiPart for multimodal
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
		} `json:"message"`
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

func (o *openaiProvider) complete(ctx context.Context, model string, maxTokens int, req Request) (Response, error) {
	msgs := make([]oaiMessage, 0, len(req.Messages)+1)
	if req.SystemCacheable != "" {
		msgs = append(msgs, oaiMessage{Role: "system", Content: req.SystemCacheable})
	}
	for _, m := range req.Messages {
		if len(m.Images) == 0 {
			msgs = append(msgs, oaiMessage{Role: m.Role, Content: m.Content})
			continue
		}
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
		msgs = append(msgs, oaiMessage{Role: m.Role, Content: parts})
	}

	oaiReq := oaiRequest{Model: model, Messages: msgs}
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
	if len(out.Choices) > 0 {
		text = out.Choices[0].Message.Content
	}
	cached := out.Usage.PromptTokensDetails.CachedTokens
	input := out.Usage.PromptTokens - cached // exclude cache from input (Anthropic-style)
	if input < 0 {
		input = 0
	}
	return Response{
		Text:         text,
		InputTokens:  input,
		OutputTokens: out.Usage.CompletionTokens,
		CachedTokens: cached,
	}, nil
}
