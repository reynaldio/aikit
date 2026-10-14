package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
)

// openaiEmbed is the OpenAI (and compatible) embeddings endpoint.
type openaiEmbed struct {
	apiKey  string
	baseURL string
	http    *http.Client
}

func newOpenAIEmbed(apiKey, baseURL string, lg *slog.Logger) embedBackend {
	if baseURL == "" {
		baseURL = openaiBaseURL
	}
	return &openaiEmbed{
		apiKey:  apiKey,
		baseURL: strings.TrimSuffix(baseURL, "/"),
		http:    newProviderHTTPClient(lg),
	}
}

func (*openaiEmbed) maxTexts() int  { return 2048 }
func (*openaiEmbed) maxTokens() int { return 250000 }

type oaiEmbedRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format"`
	Dimensions     int      `json:"dimensions,omitempty"`
}

type oaiEmbedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
	} `json:"usage"`
}

// embed sends one chunk. kind is ignored: OpenAI has no task types.
func (o *openaiEmbed) embed(ctx context.Context, model string, texts []string, _ EmbedKind, dims int) (embedChunk, error) {
	payload, err := json.Marshal(oaiEmbedRequest{Model: model, Input: texts, EncodingFormat: "float", Dimensions: dims})
	if err != nil {
		return embedChunk{}, fmt.Errorf("openai: encode embeddings: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/embeddings", bytes.NewReader(payload))
	if err != nil {
		return embedChunk{}, fmt.Errorf("openai: build embeddings request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)
	res, err := o.http.Do(req)
	if err != nil {
		return embedChunk{}, err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(res.Body)
	if err != nil {
		if res.StatusCode < 300 {
			// A 2xx reply was likely billed.
			return embedChunk{tokens: estimateTokens(strings.Join(texts, "")), estimated: true}, readError(ctx, "openai", err)
		}
		return embedChunk{}, readError(ctx, "openai", err)
	}
	if res.StatusCode >= 300 {
		msg := strings.TrimSpace(string(raw))
		var e oaiResponse
		if json.Unmarshal(raw, &e) == nil && e.Error != nil {
			msg = e.Error.Message
		}
		return embedChunk{}, fmt.Errorf("openai: %s (status %d)", msg, res.StatusCode)
	}
	var out oaiEmbedResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		// The reply was 2xx, so it was likely billed. Count a guess.
		return embedChunk{tokens: estimateTokens(strings.Join(texts, "")), estimated: true},
			fmt.Errorf("openai: decode embeddings: %w", err)
	}

	// Tokens first, so every error below can carry them (the chunk was billed).
	chunk := embedChunk{tokens: out.Usage.PromptTokens}
	if chunk.tokens <= 0 {
		chunk.tokens = estimateTokens(strings.Join(texts, ""))
		chunk.estimated = true
	}
	vectors := make([][]float32, len(texts))
	for _, item := range out.Data {
		if item.Index < 0 || item.Index >= len(texts) || vectors[item.Index] != nil {
			return chunk, fmt.Errorf("openai: embeddings index %d out of range or repeated", item.Index)
		}
		vectors[item.Index] = item.Embedding
	}
	for _, v := range vectors {
		if v == nil {
			return chunk, fmt.Errorf("openai: got %d embeddings for %d texts", len(out.Data), len(texts))
		}
	}
	chunk.vectors = vectors
	return chunk, nil
}
