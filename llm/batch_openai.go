package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"strings"
)

// openaiBatch is the Batch API of OpenAI and of Moonshot (same wire format, raw HTTP).
type openaiBatch struct {
	name    Provider // ProviderOpenAI or ProviderMoonshot; used in error text
	apiKey  string
	baseURL string
	http    *http.Client
	p       *openaiProvider // only for the live request builder (oaiChatRequest)
	log     *slog.Logger
}

// newOpenAIBatch builds the backend. baseURL "" = OpenAI's own. Strict JSON schema
// is sent for OpenAI only, as on the live path.
func newOpenAIBatch(name Provider, apiKey, baseURL string, lg *slog.Logger) batchBackend {
	if baseURL == "" {
		baseURL = openaiBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")
	p := newOpenAI(apiKey, baseURL, 0, lg).(*openaiProvider)
	p.strictSchema = name == ProviderOpenAI
	return &openaiBatch{name: name, apiKey: apiKey, baseURL: baseURL, http: p.http, p: p, log: lg}
}

func (o *openaiBatch) supports(kind batchKind) bool {
	if kind == batchEmbed {
		return o.name == ProviderOpenAI
	}
	return kind == batchChat || kind == batchChatJSON
}

// limits: OpenAI documents 50,000 requests and 200 MB per input file.
// Moonshot: see moonshotBatchLimits.
func (o *openaiBatch) limits(batchKind) batchLimits {
	if o.name == ProviderMoonshot {
		return moonshotBatchLimits
	}
	return batchLimits{maxItems: 50_000, maxBytes: 200 << 20}
}

func (o *openaiBatch) buildChat(model string, maxTokens int, req Request) ([]byte, error) {
	return json.Marshal(o.p.oaiChatRequest(model, maxTokens, req))
}

func (o *openaiBatch) buildEmbed(model, text string, _ EmbedKind, dims int) ([]byte, error) {
	if o.name != ProviderOpenAI {
		return nil, fmt.Errorf("%w: %s has no batch embeddings", ErrBatchUnsupported, o.name)
	}
	// One text per line, so input is a single string.
	return json.Marshal(struct {
		Model          string `json:"model"`
		Input          string `json:"input"`
		EncodingFormat string `json:"encoding_format"`
		Dimensions     int    `json:"dimensions,omitempty"`
	}{model, text, "float", dims})
}

func endpointFor(kind batchKind) string {
	if kind == batchEmbed {
		return "/v1/embeddings"
	}
	return "/v1/chat/completions"
}

// do sends one request. The caller closes the body. A non-2xx reply is turned into
// an error (and its body closed).
func (o *openaiBatch) do(ctx context.Context, method, path, ctype string, body []byte) (*http.Response, error) {
	var rd io.Reader
	if body != nil {
		rd = bytes.NewReader(body) // a bytes.Reader sets GetBody, so the retry transport can replay it
	}
	req, err := http.NewRequestWithContext(ctx, method, o.baseURL+path, rd)
	if err != nil {
		return nil, fmt.Errorf("%s: build request: %w", o.name, err)
	}
	if ctype != "" {
		req.Header.Set("Content-Type", ctype)
	}
	req.Header.Set("Authorization", "Bearer "+o.apiKey)
	res, err := o.http.Do(req)
	if err != nil {
		return nil, err
	}
	if res.StatusCode >= 300 {
		defer res.Body.Close()
		raw, rerr := io.ReadAll(io.LimitReader(res.Body, 1<<20))
		if rerr != nil {
			return nil, readError(ctx, string(o.name), rerr)
		}
		return nil, fmt.Errorf("%s: %s (status %d)", o.name, errorText(raw), res.StatusCode)
	}
	return res, nil
}

// errorText pulls error.message out of an API error body, or trims the raw body.
func errorText(raw []byte) string {
	var e struct {
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &e) == nil && e.Error != nil && e.Error.Message != "" {
		return e.Error.Message
	}
	s := strings.TrimSpace(string(raw))
	if len(s) > 200 {
		s = s[:200]
	}
	return s
}

// doJSON sends a request and decodes a JSON reply into out.
func (o *openaiBatch) doJSON(ctx context.Context, method, path string, in, out any) error {
	var body []byte
	ctype := ""
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return fmt.Errorf("%s: encode request: %w", o.name, err)
		}
		body, ctype = b, "application/json"
	}
	res, err := o.do(ctx, method, path, ctype, body)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return readError(ctx, string(o.name), err)
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("%s: decode reply: %w", o.name, err)
	}
	return nil
}

func (o *openaiBatch) submit(ctx context.Context, kind batchKind, _ string, lines []batchLine) (string, error) {
	type jl struct {
		CustomID string          `json:"custom_id"`
		Method   string          `json:"method"`
		URL      string          `json:"url"`
		Body     json.RawMessage `json:"body"`
	}
	endpoint := endpointFor(kind)
	var jsonl bytes.Buffer
	for _, l := range lines {
		b, err := json.Marshal(jl{l.customID, "POST", endpoint, l.body})
		if err != nil {
			return "", fmt.Errorf("%s: encode batch line: %w", o.name, err)
		}
		jsonl.Write(b)
		jsonl.WriteByte('\n')
	}

	// 1. Upload the file. Normal retry: uploading twice only leaves an unused file.
	var mp bytes.Buffer
	w := multipart.NewWriter(&mp)
	if err := w.WriteField("purpose", "batch"); err != nil {
		return "", err
	}
	h := textproto.MIMEHeader{}
	h.Set("Content-Disposition", `form-data; name="file"; filename="batch.jsonl"`)
	h.Set("Content-Type", "application/jsonl")
	part, err := w.CreatePart(h)
	if err != nil {
		return "", err
	}
	part.Write(jsonl.Bytes())
	if err := w.Close(); err != nil {
		return "", err
	}
	var up struct {
		ID string `json:"id"`
	}
	res, err := o.do(ctx, http.MethodPost, "/files", w.FormDataContentType(), mp.Bytes())
	if err != nil {
		return "", fmt.Errorf("%s: upload batch file: %w", o.name, err)
	}
	raw, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		return "", readError(ctx, string(o.name), err)
	}
	if err := json.Unmarshal(raw, &up); err != nil || up.ID == "" {
		return "", fmt.Errorf("%s: upload batch file: the reply has no file id", o.name)
	}

	// 2. Create the batch. Never retried: any failure may hide a created batch.
	var created struct {
		ID string `json:"id"`
	}
	err = o.doJSON(withNoRetry(ctx), http.MethodPost, "/batches", map[string]string{
		"input_file_id":     up.ID,
		"endpoint":          endpoint,
		"completion_window": "24h",
	}, &created)
	if err == nil && created.ID == "" {
		err = fmt.Errorf("%s: the reply has no batch id", o.name)
	}
	if err != nil {
		return "", errCreateMayExist(string(o.name), err)
	}
	return created.ID, nil
}

type oaiBatchObj struct {
	Status        string `json:"status"`
	OutputFileID  string `json:"output_file_id"`
	ErrorFileID   string `json:"error_file_id"`
	RequestCounts struct {
		Total     int `json:"total"`
		Completed int `json:"completed"`
		Failed    int `json:"failed"`
	} `json:"request_counts"`
	Errors struct {
		Data []struct {
			Message string `json:"message"`
		} `json:"data"`
	} `json:"errors"`
}

func (o *openaiBatch) get(ctx context.Context, id string) (oaiBatchObj, error) {
	var b oaiBatchObj
	err := o.doJSON(ctx, http.MethodGet, "/batches/"+url.PathEscape(id), nil, &b)
	return b, err
}

func (o *openaiBatch) status(ctx context.Context, id string) (BatchStatus, error) {
	b, err := o.get(ctx, id)
	if err != nil {
		return BatchStatus{}, err
	}
	st := BatchStatus{
		State:     BatchRunning,
		Total:     b.RequestCounts.Total,
		Succeeded: b.RequestCounts.Completed,
		Failed:    b.RequestCounts.Failed,
	}
	switch b.Status {
	case "validating", "in_progress", "finalizing", "cancelling":
	case "completed", "expired", "cancelled":
		st.State = BatchDone
	case "failed":
		st.State = BatchFailed
		var msgs []string
		for _, e := range b.Errors.Data {
			if e.Message != "" {
				msgs = append(msgs, e.Message)
			}
		}
		text := "batch failed"
		if len(msgs) > 0 {
			text = strings.Join(msgs, "; ")
		}
		st.Err = fmt.Errorf("%s: %s", o.name, text)
	default:
		if o.log != nil {
			o.log.Warn("llm: unknown batch status, treating it as running", "provider", string(o.name), "status", b.Status)
		}
	}
	return st, nil
}

type oaiBatchLine struct {
	CustomID string `json:"custom_id"`
	Response *struct {
		StatusCode int             `json:"status_code"`
		Body       json.RawMessage `json:"body"`
	} `json:"response"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func (o *openaiBatch) results(ctx context.Context, kind batchKind, model, id string) iter.Seq2[rawBatchResult, error] {
	return func(yield func(rawBatchResult, error) bool) {
		b, err := o.get(ctx, id)
		if err != nil {
			yield(rawBatchResult{}, err)
			return
		}
		for _, fid := range []string{b.OutputFileID, b.ErrorFileID} {
			if fid == "" {
				continue
			}
			if !o.streamFile(ctx, kind, model, fid, yield) {
				return
			}
		}
	}
}

// streamFile yields every line of one result file. It returns false when the
// consumer stopped or an error was yielded. The body is closed on every path.
func (o *openaiBatch) streamFile(ctx context.Context, kind batchKind, model, fid string, yield func(rawBatchResult, error) bool) bool {
	res, err := o.do(ctx, http.MethodGet, "/files/"+url.PathEscape(fid)+"/content", "", nil)
	if err != nil {
		yield(rawBatchResult{}, err)
		return false
	}
	defer res.Body.Close()
	for line, err := range jsonlLines(res.Body) {
		if err != nil {
			yield(rawBatchResult{}, err)
			return false
		}
		var item oaiBatchLine
		if err := json.Unmarshal(line, &item); err != nil {
			yield(rawBatchResult{}, fmt.Errorf("%s: bad batch result line: %w", o.name, err))
			return false
		}
		if !yield(o.parseLine(kind, model, item), nil) {
			return false
		}
	}
	return true
}

func (o *openaiBatch) parseLine(kind batchKind, model string, item oaiBatchLine) rawBatchResult {
	raw := rawBatchResult{customID: item.CustomID}
	if e := item.Error; e != nil && (e.Code != "" || e.Message != "") {
		switch e.Code {
		case "batch_expired":
			raw.err = ErrBatchItemExpired
		case "batch_cancelled":
			raw.err = ErrBatchItemCanceled
		default:
			raw.err = fmt.Errorf("%s: %s (%s)", o.name, e.Message, e.Code)
		}
		return raw
	}
	r := item.Response
	if r == nil {
		raw.err = fmt.Errorf("%s: batch result line has no response", o.name)
		return raw
	}
	if r.StatusCode != http.StatusOK {
		raw.err = fmt.Errorf("%s: %s (status %d)", o.name, errorText(r.Body), r.StatusCode)
		return raw
	}
	if kind != batchEmbed {
		raw.resp, raw.err = oaiDecodeResponse(model, r.Body, r.StatusCode)
		return raw
	}
	var out oaiEmbedResponse
	if err := json.Unmarshal(r.Body, &out); err != nil {
		raw.err = fmt.Errorf("%s: decode embedding: %w", o.name, err)
		return raw
	}
	if len(out.Data) == 0 {
		raw.err = fmt.Errorf("%s: batch item has no embedding", o.name)
		return raw
	}
	raw.vector = out.Data[0].Embedding
	// The text is not kept, so there is no estimate: a reply without usage means 0 tokens
	// and UsageEstimated stays false.
	raw.resp = Response{InputTokens: out.Usage.PromptTokens}
	return raw
}

func (o *openaiBatch) cancel(ctx context.Context, id string) error {
	return o.doJSON(ctx, http.MethodPost, "/batches/"+url.PathEscape(id)+"/cancel", nil, nil)
}

// moonshotBatchLimits: read on 2026-10-14 from https://platform.kimi.ai/docs/api/batch-create
// (and the files page): the input file is at most 100 MB and 1000 batch files per
// organization. No cap on requests per batch is documented, so maxItems is 0.
var moonshotBatchLimits = batchLimits{maxBytes: 100 << 20}
