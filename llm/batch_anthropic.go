package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"net/url"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicBatch is the Anthropic Message Batches backend.
type anthropicBatch struct {
	client anthropic.Client
}

// newAnthropicBatch builds the backend. The client is set up like newAnthropic's:
// the SDK's own retries are off and the shared transport retries instead. opts come
// last so a test can add option.WithBaseURL.
func newAnthropicBatch(apiKey string, lg *slog.Logger, opts ...option.RequestOption) batchBackend {
	base := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(newProviderHTTPClient(lg)),
		option.WithMaxRetries(0),
	}
	return &anthropicBatch{client: anthropic.NewClient(append(base, opts...)...)}
}

func (a *anthropicBatch) supports(kind batchKind) bool {
	return kind == batchChat || kind == batchChatJSON
}

func (a *anthropicBatch) limits(batchKind) batchLimits {
	return batchLimits{maxItems: 100_000, maxBytes: 256 << 20}
}

// buildChat encodes one item with the live params builder, then copies it into the
// batch params type. Every field the two types share is copied, except ServiceTier
// (a different type) and Stream (never set in a batch).
func (a *anthropicBatch) buildChat(model string, maxTokens int, req Request) ([]byte, error) {
	p := anthropicParams(model, maxTokens, req)
	bp := anthropic.MessageBatchNewParamsRequestParams{
		MaxTokens:     p.MaxTokens,
		Messages:      p.Messages,
		Model:         p.Model,
		Container:     p.Container,
		InferenceGeo:  p.InferenceGeo,
		Temperature:   p.Temperature,
		TopK:          p.TopK,
		TopP:          p.TopP,
		CacheControl:  p.CacheControl,
		Metadata:      p.Metadata,
		OutputConfig:  p.OutputConfig,
		StopSequences: p.StopSequences,
		System:        p.System,
		Thinking:      p.Thinking,
		ToolChoice:    p.ToolChoice,
		Tools:         p.Tools,
	}
	return json.Marshal(bp)
}

func (a *anthropicBatch) buildEmbed(string, string, EmbedKind, int) ([]byte, error) {
	return nil, fmt.Errorf("%w: anthropic has no batch embeddings", ErrBatchUnsupported)
}

// submit writes the request body by hand: the lines are already JSON and may not
// round-trip through the SDK's param structs. The create call is never retried.
func (a *anthropicBatch) submit(ctx context.Context, _ batchKind, _ string, lines []batchLine) (string, error) {
	type item struct {
		CustomID string          `json:"custom_id"`
		Params   json.RawMessage `json:"params"`
	}
	items := make([]item, len(lines))
	for i, l := range lines {
		items[i] = item{CustomID: l.customID, Params: l.body}
	}
	body, err := json.Marshal(struct {
		Requests []item `json:"requests"`
	}{items})
	if err != nil {
		return "", fmt.Errorf("anthropic: encode batch: %w", err)
	}
	var batch anthropic.MessageBatch
	err = a.client.Post(withNoRetry(ctx), "v1/messages/batches", nil, &batch,
		option.WithRequestBody("application/json", bytes.NewReader(body)))
	if err == nil && batch.ID == "" {
		err = fmt.Errorf("anthropic: the reply has no batch id")
	}
	if err != nil {
		return "", errCreateMayExist("anthropic", err)
	}
	return batch.ID, nil
}

func (a *anthropicBatch) status(ctx context.Context, id string) (BatchStatus, error) {
	b, err := a.client.Messages.Batches.Get(ctx, id)
	if err != nil {
		return BatchStatus{}, err
	}
	c := b.RequestCounts
	st := BatchStatus{
		State:     BatchRunning,
		Succeeded: int(c.Succeeded),
		Failed:    int(c.Errored),
		Canceled:  int(c.Canceled),
		Expired:   int(c.Expired),
		Total:     int(c.Succeeded + c.Errored + c.Canceled + c.Expired + c.Processing),
	}
	if b.ProcessingStatus == anthropic.MessageBatchProcessingStatusEnded {
		st.State = BatchDone
	}
	return st, nil
}

// results reads the results file line by line. It does not use the SDK's
// ResultsStreaming: that has a 64 KB line cap, and one long reply is a longer line.
func (a *anthropicBatch) results(ctx context.Context, _ batchKind, model, id string) iter.Seq2[rawBatchResult, error] {
	return func(yield func(rawBatchResult, error) bool) {
		var res *http.Response
		err := a.client.Get(ctx, "v1/messages/batches/"+url.PathEscape(id)+"/results", nil, &res,
			option.WithHeader("Accept", "application/x-jsonl"))
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
			var item anthropic.MessageBatchIndividualResponse
			if err := json.Unmarshal(line, &item); err != nil {
				yield(rawBatchResult{}, fmt.Errorf("anthropic: bad batch result line: %w", err))
				return
			}
			raw := rawBatchResult{customID: item.CustomID}
			switch item.Result.Type {
			case "succeeded":
				raw.resp, raw.err = anthropicResponse(model, &item.Result.Message)
			case "errored":
				e := item.Result.Error.Error
				raw.err = fmt.Errorf("anthropic: %s: %s", e.Type, e.Message)
			case "canceled":
				raw.err = ErrBatchItemCanceled
			case "expired":
				raw.err = ErrBatchItemExpired
			default:
				raw.err = fmt.Errorf("anthropic: unknown batch result type %q", item.Result.Type)
			}
			if !yield(raw, nil) {
				return
			}
		}
	}
}

func (a *anthropicBatch) cancel(ctx context.Context, id string) error {
	_, err := a.client.Messages.Batches.Cancel(ctx, id)
	return err
}
