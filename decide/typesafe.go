package decide

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

// DefaultBaseURL is TypeSafe's hosted API.
const DefaultBaseURL = "https://api.typesafe.ai/v1"

// DefaultModel is TypeSafe's rolling alias. Pin a concrete version (e.g. "jev-1.13")
// where a silent model upgrade would invalidate an eval baseline.
const DefaultModel = "jev-latest"

// Config wires the TypeSafe Jev backend.
type Config struct {
	APIKey  string // empty → New returns the noop client
	BaseURL string // optional; default DefaultBaseURL
	Model   string // optional; default DefaultModel
	// HTTPClient is optional; the default has a 30s timeout. Jev answers in a single
	// pass with no generation, so a call that takes longer than that is stuck.
	HTTPClient *http.Client
}

// New builds the Jev client. Returns the noop client if no API key is set.
func New(cfg Config) Client {
	if cfg.APIKey == "" {
		return NewNoop()
	}
	base := cfg.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	model := cfg.Model
	if model == "" {
		model = DefaultModel
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 30 * time.Second}
	}
	return &jevClient{apiKey: cfg.APIKey, baseURL: strings.TrimRight(base, "/"), model: model, http: hc}
}

// jevClient calls TypeSafe's System One endpoint (raw HTTP; there is no Go SDK).
type jevClient struct {
	apiKey  string
	baseURL string
	model   string
	http    *http.Client
}

func (c *jevClient) Enabled() bool { return true }

type jevRequest struct {
	Model     string              `json:"model"`
	State     any                 `json:"state"`
	Questions map[string]Question `json:"questions"`
}

type jevResponse struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func (c *jevClient) Evaluate(ctx context.Context, req Request) (Response, error) {
	if err := req.validate(); err != nil {
		return Response{}, err
	}
	model := req.Model
	if model == "" {
		model = c.model
	}
	buf, err := json.Marshal(jevRequest{Model: model, State: req.State, Questions: req.Questions})
	if err != nil {
		return Response{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/systemone", bytes.NewReader(buf))
	if err != nil {
		return Response{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	res, err := c.http.Do(httpReq)
	if err != nil {
		return Response{}, err
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(res.Body)

	if res.StatusCode >= 300 {
		return Response{}, &APIError{StatusCode: res.StatusCode, Message: errorMessage(raw, res.StatusCode)}
	}
	var out jevResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return Response{}, fmt.Errorf("decide: decode response: %w", err)
	}
	resp := Response{
		Model:        out.Model,
		Answers:      out.Answers,
		InputTokens:  out.Usage.InputTokens,
		OutputTokens: out.Usage.OutputTokens,
	}
	if resp.Model == "" {
		resp.Model = model
	}
	// A missing answer must not read as a zero-valued one: Noul 0 is a confident "no".
	// The tokens were still billed, so the Response comes back populated for metering.
	for id := range req.Questions {
		if _, ok := out.Answers[id]; !ok {
			return resp, fmt.Errorf("decide: no answer for question %q", id)
		}
	}
	return resp, nil
}

// errorMessage pulls a message out of an error body. TypeSafe does not document the
// error shape, so accept the common ones and fall back to the raw body.
func errorMessage(raw []byte, status int) string {
	var body struct {
		Message string `json:"message"`
		Detail  any    `json:"detail"`
		Error   any    `json:"error"`
	}
	if json.Unmarshal(raw, &body) == nil {
		if body.Message != "" {
			return body.Message
		}
		for _, v := range []any{body.Error, body.Detail} {
			switch v := v.(type) {
			case string:
				if v != "" {
					return v
				}
			case map[string]any:
				if m, ok := v["message"].(string); ok && m != "" {
					return m
				}
			}
		}
	}
	if s := strings.TrimSpace(string(raw)); s != "" {
		return s
	}
	return http.StatusText(status)
}
