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

// googleEmbed is the Gemini batchEmbedContents endpoint.
type googleEmbed struct {
	apiKey  string
	baseURL string
	http    *http.Client
}

func newGoogleEmbed(apiKey string, lg *slog.Logger) embedBackend {
	return &googleEmbed{apiKey: apiKey, baseURL: geminiBaseURL, http: newProviderHTTPClient(lg)}
}

func (*googleEmbed) maxTexts() int  { return 100 }
func (*googleEmbed) maxTokens() int { return 0 }

// geminiEmbedUsesTaskType reports whether the model takes a taskType field.
// Newer models (gemini-embedding-2 and later) take a text prefix instead.
func geminiEmbedUsesTaskType(model string) bool {
	model = strings.TrimPrefix(model, "models/")
	return model == "gemini-embedding-001" || model == "embedding-001" || strings.HasPrefix(model, "text-embedding-")
}

type geminiEmbedConfig struct {
	TaskType             string `json:"taskType,omitempty"`
	OutputDimensionality int    `json:"outputDimensionality,omitempty"`
}

type geminiEmbedItem struct {
	Model   string `json:"model"`
	Content struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"content"`
	Config *geminiEmbedConfig `json:"embedContentConfig,omitempty"`
}

type geminiEmbedRequest struct {
	Requests []geminiEmbedItem `json:"requests"`
}

type geminiEmbedResponse struct {
	Embeddings []struct {
		Values []float32 `json:"values"`
	} `json:"embeddings"`
	UsageMetadata struct {
		PromptTokenCount int `json:"promptTokenCount"`
	} `json:"usageMetadata"`
}

func (g *googleEmbed) embed(ctx context.Context, model string, texts []string, kind EmbedKind, dims int) (embedChunk, error) {
	m := strings.TrimPrefix(model, "models/")
	taskModel := geminiEmbedUsesTaskType(m)
	cfg := &geminiEmbedConfig{OutputDimensionality: dims}
	if taskModel {
		switch kind {
		case EmbedQuery:
			cfg.TaskType = "RETRIEVAL_QUERY"
		case EmbedDocument:
			cfg.TaskType = "RETRIEVAL_DOCUMENT"
		}
	}
	if cfg.TaskType == "" && cfg.OutputDimensionality <= 0 {
		cfg = nil
	}

	sent := make([]string, len(texts))
	body := geminiEmbedRequest{Requests: make([]geminiEmbedItem, len(texts))}
	for i, text := range texts {
		if !taskModel {
			switch kind {
			case EmbedQuery:
				text = "task: search result | query: " + text
			case EmbedDocument:
				text = "title: none | text: " + text
			}
		}
		sent[i] = text
		item := &body.Requests[i]
		item.Model = "models/" + m
		item.Content.Parts = []struct {
			Text string `json:"text"`
		}{{Text: text}}
		item.Config = cfg
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return embedChunk{}, fmt.Errorf("gemini: encode embeddings: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.baseURL+"/models/"+m+":batchEmbedContents", bytes.NewReader(payload))
	if err != nil {
		return embedChunk{}, fmt.Errorf("gemini: build embeddings request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", g.apiKey)
	res, err := g.http.Do(req)
	if err != nil {
		return embedChunk{}, err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return embedChunk{}, readError(ctx, "gemini", err)
	}
	if res.StatusCode >= 300 {
		msg := strings.TrimSpace(string(raw))
		var e geminiResponse
		if json.Unmarshal(raw, &e) == nil && e.Error != nil {
			msg = e.Error.Message
		}
		return embedChunk{}, fmt.Errorf("gemini: %s (status %d)", msg, res.StatusCode)
	}
	var out geminiEmbedResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return embedChunk{}, fmt.Errorf("gemini: decode embeddings: %w", err)
	}

	// Tokens first, so the count error below can carry them (the chunk was billed).
	chunk := embedChunk{tokens: out.UsageMetadata.PromptTokenCount}
	if chunk.tokens <= 0 {
		chunk.tokens = estimateTokens(strings.Join(sent, ""))
		chunk.estimated = true
	}
	if len(out.Embeddings) != len(texts) {
		return chunk, fmt.Errorf("gemini: got %d embeddings for %d texts", len(out.Embeddings), len(texts))
	}
	chunk.vectors = make([][]float32, len(texts))
	for i, e := range out.Embeddings {
		chunk.vectors[i] = e.Values
	}
	return chunk, nil
}
