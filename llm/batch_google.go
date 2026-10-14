package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// googleBatch is the Gemini Batch API (file input, raw HTTP).
//
// Docs checked on 2026-10-14 through a summary (not the raw pages), so the field
// names below still need a check with a real key:
//   - https://ai.google.dev/gemini-api/docs/batch-api : a file line is
//     {"key": ..., "request": {...}}; a result line is {"key": ..., "response": {...}}
//     or {"key": ..., "error": {...}}; states are JOB_STATE_*; the result file is
//     dest.file_name.
//   - https://ai.google.dev/api/embeddings#method:-models.asyncbatchembedcontent :
//     body {"batch": {"displayName", "model": "models/{m}", "inputConfig":
//     {"fileName": "files/..."}}}. An embed request has model, content and the
//     (deprecated) taskType and outputDimensionality, or an embedContentConfig.
//
// The docs disagree in places (state prefix, result file field, embed reply shape),
// so the readers below accept every spelling seen. The embed line body follows the
// live embed body (geminiEmbedItem: model, content, embedContentConfig).
type googleBatch struct {
	apiKey      string
	baseURL     string // API base, overridable in tests
	uploadURL   string
	downloadURL string
	http        *http.Client
	log         *slog.Logger
}

const (
	geminiUploadURL   = "https://generativelanguage.googleapis.com/upload/v1beta"
	geminiDownloadURL = "https://generativelanguage.googleapis.com/download/v1beta"
)

func newGoogleBatch(apiKey string, lg *slog.Logger) batchBackend {
	return &googleBatch{
		apiKey:      apiKey,
		baseURL:     geminiBaseURL,
		uploadURL:   geminiUploadURL,
		downloadURL: geminiDownloadURL,
		http:        newProviderHTTPClient(lg),
		log:         lg,
	}
}

func (*googleBatch) supports(kind batchKind) bool {
	return kind == batchChat || kind == batchChatJSON || kind == batchEmbed
}

// limits: Gemini allows an input file up to 2 GB and sets no item cap.
func (*googleBatch) limits(batchKind) batchLimits { return batchLimits{maxBytes: 2 << 30} }

// geminiBatchThinking picks the thinking config for a flash model in a batch. The
// live path tries knobs and retries on a 400; a batch cannot, so it picks by name:
// gemini-3 and later take a level, gemini-2 takes a budget of 0, others none.
func geminiBatchThinking(model string) *geminiThinkingConfig {
	model = strings.TrimPrefix(model, "models/")
	if !strings.Contains(model, "flash") || !strings.HasPrefix(model, "gemini-") {
		return nil
	}
	rest := strings.TrimPrefix(model, "gemini-")
	end := 0
	for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
		end++
	}
	major, err := strconv.Atoi(rest[:end])
	if err != nil {
		return nil
	}
	switch {
	case major >= 3:
		return &geminiThinkingConfig{ThinkingLevel: "minimal"}
	case major == 2:
		zero := 0
		return &geminiThinkingConfig{ThinkingBudget: &zero}
	}
	return nil
}

func (*googleBatch) buildChat(model string, maxTokens int, req Request) ([]byte, error) {
	body, err := geminiBody(model, maxTokens, req)
	if err != nil {
		return nil, err
	}
	body.GenerationConfig.ThinkingConfig = geminiBatchThinking(model)
	return json.Marshal(body)
}

func (*googleBatch) buildEmbed(model, text string, kind EmbedKind, dims int) ([]byte, error) {
	// The live embed body: model, content and embedContentConfig. The batch carries
	// the model too; the line keeps it, as the live body does.
	return json.Marshal(geminiEmbedItemFor(model, text, kind, dims))
}

// do sends one request with the API key. The caller closes the body. A non-2xx reply
// becomes an error (its body is closed).
func (g *googleBatch) do(ctx context.Context, method, rawURL string, hdr map[string]string, body []byte) (*http.Response, error) {
	var rd io.Reader
	if body != nil {
		rd = bytes.NewReader(body) // sets GetBody, so the retry transport can replay it
	}
	req, err := http.NewRequestWithContext(ctx, method, rawURL, rd)
	if err != nil {
		return nil, fmt.Errorf("gemini: build request: %w", err)
	}
	req.Header.Set("x-goog-api-key", g.apiKey)
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	res, err := g.http.Do(req)
	if err != nil {
		return nil, err
	}
	if res.StatusCode >= 300 {
		defer res.Body.Close()
		raw, rerr := io.ReadAll(io.LimitReader(res.Body, 1<<20))
		if rerr != nil {
			return nil, readError(ctx, "gemini", rerr)
		}
		return nil, fmt.Errorf("gemini: %s (status %d)", errorText(raw), res.StatusCode)
	}
	return res, nil
}

// doJSON sends a request (JSON body when in is set) and decodes the reply into out.
func (g *googleBatch) doJSON(ctx context.Context, method, rawURL string, in, out any) error {
	var body []byte
	var hdr map[string]string
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return fmt.Errorf("gemini: encode request: %w", err)
		}
		body, hdr = b, map[string]string{"Content-Type": "application/json"}
	}
	res, err := g.do(ctx, method, rawURL, hdr, body)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return readError(ctx, "gemini", err)
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("gemini: decode reply: %w", err)
	}
	return nil
}

// upload sends the JSONL with the resumable protocol and returns the file name.
// Both steps use the normal retry policy: a second upload only leaves an unused file.
func (g *googleBatch) upload(ctx context.Context, data []byte) (string, error) {
	res, err := g.do(ctx, http.MethodPost, g.uploadURL+"/files", map[string]string{
		"X-Goog-Upload-Protocol":              "resumable",
		"X-Goog-Upload-Command":               "start",
		"X-Goog-Upload-Header-Content-Length": strconv.Itoa(len(data)),
		"X-Goog-Upload-Header-Content-Type":   "application/jsonl",
		"Content-Type":                        "application/json",
	}, []byte(`{"file":{"display_name":"aikit-batch"}}`))
	if err != nil {
		return "", fmt.Errorf("gemini: start upload: %w", err)
	}
	loc := res.Header.Get("X-Goog-Upload-URL")
	_, _ = io.Copy(io.Discard, res.Body)
	res.Body.Close()
	if loc == "" {
		return "", fmt.Errorf("gemini: start upload: the reply has no X-Goog-Upload-URL header")
	}
	res, err = g.do(ctx, http.MethodPost, loc, map[string]string{
		"X-Goog-Upload-Command": "upload, finalize",
		"X-Goog-Upload-Offset":  "0",
		"Content-Type":          "application/jsonl",
	}, data)
	if err != nil {
		return "", fmt.Errorf("gemini: upload batch file: %w", err)
	}
	raw, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		return "", readError(ctx, "gemini", err)
	}
	var up struct {
		File struct {
			Name string `json:"name"`
		} `json:"file"`
	}
	if err := json.Unmarshal(raw, &up); err != nil || up.File.Name == "" {
		return "", fmt.Errorf("gemini: upload batch file: the reply has no file name")
	}
	return up.File.Name, nil
}

func (g *googleBatch) submit(ctx context.Context, kind batchKind, model string, lines []batchLine) (string, error) {
	type jl struct {
		Key     string          `json:"key"`
		Request json.RawMessage `json:"request"`
	}
	var jsonl bytes.Buffer
	for _, l := range lines {
		b, err := json.Marshal(jl{l.customID, l.body})
		if err != nil {
			return "", fmt.Errorf("gemini: encode batch line: %w", err)
		}
		jsonl.Write(b)
		jsonl.WriteByte('\n')
	}
	file, err := g.upload(ctx, jsonl.Bytes())
	if err != nil {
		return "", err
	}

	// Create. Never retried: any failure may hide a created batch.
	method := ":batchGenerateContent"
	if kind == batchEmbed {
		method = ":asyncBatchEmbedContent"
	}
	model = strings.TrimPrefix(model, "models/")
	body := map[string]any{"batch": map[string]any{
		"displayName": "aikit-batch",
		"model":       "models/" + model,
		"inputConfig": map[string]string{"fileName": file},
	}}
	var op struct {
		Name     string `json:"name"`
		Metadata struct {
			Name string `json:"name"`
		} `json:"metadata"`
	}
	err = g.doJSON(withNoRetry(ctx), http.MethodPost, g.baseURL+"/models/"+geminiModelPath(model)+method, body, &op)
	if err == nil {
		switch {
		case strings.HasPrefix(op.Metadata.Name, "batches/"):
			return op.Metadata.Name, nil
		case strings.HasPrefix(op.Name, "batches/"):
			return op.Name, nil
		}
		err = fmt.Errorf("gemini: the reply has no batch name")
	}
	return "", errCreateMayExist("gemini", err)
}

// gbCount reads a count that Google sends as a string (int64) or a number.
type gbCount int

func (c *gbCount) UnmarshalJSON(b []byte) error {
	s := strings.Trim(strings.TrimSpace(string(b)), `"`)
	if s == "" || s == "null" {
		return nil
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return err
	}
	*c = gbCount(n)
	return nil
}

// gbBatchFields are the batch resource fields. They come bare, or under "metadata"
// when the reply is an Operation.
type gbBatchFields struct {
	State      string `json:"state"`
	BatchStats struct {
		RequestCount           gbCount `json:"requestCount"`
		SuccessfulRequestCount gbCount `json:"successfulRequestCount"`
		FailedRequestCount     gbCount `json:"failedRequestCount"`
	} `json:"batchStats"`
	Output struct {
		ResponsesFile string `json:"responsesFile"`
		FileName      string `json:"fileName"`
	} `json:"output"`
	Dest struct {
		FileName string `json:"fileName"`
	} `json:"dest"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (f gbBatchFields) file() string {
	for _, s := range []string{f.Output.ResponsesFile, f.Dest.FileName, f.Output.FileName} {
		if s != "" {
			return s
		}
	}
	return ""
}

type gbBatch struct {
	gbBatchFields
	Metadata *gbBatchFields `json:"metadata"`
}

// fields returns the bare fields, or the metadata ones when the bare ones are empty.
func (b gbBatch) fields() gbBatchFields {
	f := b.gbBatchFields
	if m := b.Metadata; m != nil {
		if f.State == "" {
			f.State = m.State
		}
		if f.BatchStats == (gbBatchFields{}).BatchStats {
			f.BatchStats = m.BatchStats
		}
		if f.file() == "" {
			f.Output, f.Dest = m.Output, m.Dest
		}
		if f.Error == nil {
			f.Error = m.Error
		}
	}
	return f
}

func (g *googleBatch) get(ctx context.Context, id string) (gbBatchFields, error) {
	var b gbBatch
	if err := g.doJSON(ctx, http.MethodGet, g.baseURL+"/"+escapePath(id), nil, &b); err != nil {
		return gbBatchFields{}, err
	}
	return b.fields(), nil
}

// escapePath escapes each segment of a path like "batches/abc".
func escapePath(p string) string {
	parts := strings.Split(p, "/")
	for i, s := range parts {
		parts[i] = url.PathEscape(s)
	}
	return strings.Join(parts, "/")
}

func (g *googleBatch) status(ctx context.Context, id string) (BatchStatus, error) {
	f, err := g.get(ctx, id)
	if err != nil {
		return BatchStatus{}, err
	}
	st := BatchStatus{
		State:     BatchRunning,
		Total:     int(f.BatchStats.RequestCount),
		Succeeded: int(f.BatchStats.SuccessfulRequestCount),
		Failed:    int(f.BatchStats.FailedRequestCount),
	}
	state := f.State
	if i := strings.LastIndex(state, "_"); i >= 0 {
		state = state[i+1:]
	}
	switch state {
	case "PENDING", "RUNNING", "UNSPECIFIED":
	case "SUCCEEDED", "CANCELLED", "EXPIRED":
		st.State = BatchDone
	case "FAILED":
		st.State = BatchFailed
		msg := "batch failed"
		if f.Error != nil && f.Error.Message != "" {
			msg = f.Error.Message
		}
		st.Err = fmt.Errorf("gemini: %s", msg)
	default:
		if g.log != nil {
			g.log.Warn("llm: unknown batch state, treating it as running", "provider", "gemini", "state", f.State)
		}
	}
	return st, nil
}

type gbEmbedding struct{ values []float32 }

// UnmarshalJSON takes {"values": [...]} or a bare array.
func (e *gbEmbedding) UnmarshalJSON(b []byte) error {
	var obj struct {
		Values []float32 `json:"values"`
	}
	if json.Unmarshal(b, &obj) == nil {
		e.values = obj.Values
		return nil
	}
	return json.Unmarshal(b, &e.values)
}

type gbLine struct {
	Key      string `json:"key"`
	Response *struct {
		geminiResponse
		Embedding *gbEmbedding `json:"embedding"`
	} `json:"response"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// results streams the responses file. Gemini lists no per-item cancel or expiry, so
// a batch that was cancelled or expired yields the lines that exist, as they are:
// ErrBatchItemCanceled and ErrBatchItemExpired are never used here.
func (g *googleBatch) results(ctx context.Context, kind batchKind, model, id string) iter.Seq2[rawBatchResult, error] {
	return func(yield func(rawBatchResult, error) bool) {
		f, err := g.get(ctx, id)
		if err != nil {
			yield(rawBatchResult{}, err)
			return
		}
		file := f.file()
		if file == "" {
			yield(rawBatchResult{}, fmt.Errorf("gemini: batch has no results file"))
			return
		}
		res, err := g.do(ctx, http.MethodGet, g.downloadURL+"/"+escapePath(file)+":download?alt=media", nil, nil)
		if err != nil {
			yield(rawBatchResult{}, err)
			return
		}
		defer res.Body.Close()
		for line, err := range jsonlLines(res.Body) {
			if err != nil {
				yield(rawBatchResult{}, err)
				return
			}
			var item gbLine
			if err := json.Unmarshal(line, &item); err != nil {
				yield(rawBatchResult{}, fmt.Errorf("gemini: bad batch result line: %w", err))
				return
			}
			if !yield(g.parseLine(kind, model, item), nil) {
				return
			}
		}
	}
}

func (g *googleBatch) parseLine(kind batchKind, model string, item gbLine) rawBatchResult {
	raw := rawBatchResult{customID: item.Key}
	if e := item.Error; e != nil {
		raw.err = fmt.Errorf("gemini: %s (code %d)", e.Message, e.Code)
		return raw
	}
	r := item.Response
	if r == nil {
		raw.err = fmt.Errorf("gemini: batch result line has no response")
		return raw
	}
	if kind == batchEmbed {
		if r.Embedding == nil || len(r.Embedding.values) == 0 {
			raw.err = fmt.Errorf("gemini: batch item has no embedding")
			return raw
		}
		raw.vector = r.Embedding.values
		raw.resp = Response{InputTokens: r.UsageMetadata.PromptTokenCount}
		return raw
	}
	raw.resp, raw.err = geminiBuildResponse(model, r.geminiResponse)
	return raw
}

func (g *googleBatch) cancel(ctx context.Context, id string) error {
	return g.doJSON(ctx, http.MethodPost, g.baseURL+"/"+escapePath(id)+":cancel", nil, nil)
}
