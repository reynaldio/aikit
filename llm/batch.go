package llm

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/url"
	"strings"
	"time"
)

// Batcher runs many requests as one provider batch, at a lower price. A batch is
// not instant: it ends within 24 hours (Gemini: 48 hours). Submit, then poll Status
// (or use Wait), then read Results.
//
// Results come in the provider's order, not the order of the items. Match them on
// CustomID. There is no failover: one batch uses one model.
//
// If a create call fails or its reply is lost, the batch may still exist at the
// provider. The error says so. Check the provider console before you submit again.
type Batcher interface {
	// SubmitChat sends a batch of chat requests and returns its ID.
	SubmitChat(ctx context.Context, b ChatBatch) (BatchID, error)
	// SubmitEmbed sends a batch of texts to embed and returns its ID.
	SubmitEmbed(ctx context.Context, b EmbedBatch) (BatchID, error)
	// Status reports how far a batch has got.
	Status(ctx context.Context, id BatchID) (BatchStatus, error)
	// Results streams the items of a finished batch. Before the batch is done it
	// yields ErrBatchNotDone.
	Results(ctx context.Context, id BatchID) iter.Seq2[BatchResult, error]
	// Cancel asks the provider to stop a batch.
	Cancel(ctx context.Context, id BatchID) error
	// Enabled reports whether at least one provider can batch.
	Enabled() bool
}

// ChatBatch is a set of chat requests that share one model.
type ChatBatch struct {
	// Task picks the model through the profiles, as for Complete. Ignored when Model is set.
	Task Task
	// Model overrides Task and the profiles. Nil or empty = no override.
	Model *ModelRef
	// Items must each have a unique CustomID. An item's Request.Model must be
	// empty or the batch model, Request.Task empty or equal to Task, and
	// Request.OnEvent nil. Either every item has a JSONSchema or none does.
	Items []ChatItem
}

// ChatItem is one request in a ChatBatch.
type ChatItem struct {
	// CustomID is 1 to 64 characters of A-Z, a-z, 0-9, "_" and "-".
	CustomID string
	Request  Request
}

// EmbedBatch is a set of texts to embed with one model.
type EmbedBatch struct {
	// Model overrides Config.EmbedModel. Nil or empty = no override.
	Model *ModelRef
	// Kind is an optional hint, as in EmbedRequest.
	Kind EmbedKind
	// Dimensions asks for shorter vectors. 0 = the model's default size.
	Dimensions int
	Items      []EmbedItem
}

// EmbedItem is one text in an EmbedBatch.
type EmbedItem struct {
	// CustomID follows the same rule as ChatItem.CustomID.
	CustomID string
	Text     string
}

// BatchID names a batch. It holds the provider, the kind of batch, the model and
// the provider's own ID, so no state is kept in this library. Treat it as opaque
// and store it as a string.
type BatchID string

// Provider returns the provider that runs the batch, or "" for a malformed ID.
func (id BatchID) Provider() Provider {
	p, _, _, _, err := parseBatchID(id)
	if err != nil {
		return ""
	}
	return p
}

// Model returns the model the batch runs on, or "" for a malformed ID.
func (id BatchID) Model() string {
	_, _, m, _, err := parseBatchID(id)
	if err != nil {
		return ""
	}
	return m
}

// BatchState is where a batch is in its life.
type BatchState string

const (
	// BatchRunning means the provider has not finished the batch.
	BatchRunning BatchState = "running"
	// BatchDone means results can be read. Some items may still have failed.
	BatchDone BatchState = "done"
	// BatchFailed means the whole batch failed. BatchStatus.Err says why.
	BatchFailed BatchState = "failed"
)

// BatchStatus is the progress of a batch. Counts are what the provider reports; a
// count the provider does not give is 0.
type BatchStatus struct {
	State     BatchState
	Total     int
	Succeeded int
	Failed    int
	Canceled  int
	Expired   int
	// Err is set when State is BatchFailed.
	Err error
}

// BatchResult is one item of a finished batch.
type BatchResult struct {
	CustomID string
	// Response is the chat reply. Provider and Model are set. On a refusal or a
	// schema violation it keeps the text and token counts.
	Response Response
	// Vector is the embedding, unit length (a zero vector stays zero). Embed batches only.
	Vector []float32
	// Err is this item's own failure. Other items may have succeeded.
	Err error
}

// Wait polls Status every `every` (0 or less = 30 seconds) until the batch is done
// or failed. It returns the last status. A failed batch is not an error here:
// check the state. It stops with ctx.Err() when ctx ends.
func Wait(ctx context.Context, b Batcher, id BatchID, every time.Duration) (BatchStatus, error) {
	if every <= 0 {
		every = 30 * time.Second
	}
	for {
		st, err := b.Status(ctx, id)
		if err != nil {
			return st, err
		}
		if st.State == BatchDone || st.State == BatchFailed {
			return st, nil
		}
		timer := time.NewTimer(every)
		select {
		case <-ctx.Done():
			timer.Stop()
			return st, ctx.Err()
		case <-timer.C:
		}
	}
}

// CollectResults reads a Results sequence into a slice. On the first error it
// returns what it has so far and that error.
func CollectResults(seq iter.Seq2[BatchResult, error]) ([]BatchResult, error) {
	var out []BatchResult
	for r, err := range seq {
		if err != nil {
			return out, err
		}
		out = append(out, r)
	}
	return out, nil
}

// batchKind is the kind of batch inside a BatchID.
type batchKind string

const (
	batchChat     batchKind = "chat"
	batchChatJSON batchKind = "chatjson" // every item had a JSONSchema
	batchEmbed    batchKind = "embed"
)

// batchLimits are a provider's caps for one batch.
type batchLimits struct {
	maxItems int   // 0 = no cap
	maxBytes int64 // total encoded bytes of all lines; 0 = no cap
}

// batchLine is one item ready to send. body is the provider's own JSON for the
// request part only (the backend wraps it with custom_id / key on submit).
type batchLine struct {
	customID string
	body     []byte
}

// batchBackend is one provider's batch endpoint.
type batchBackend interface {
	// supports reports whether this backend can batch kind. batchChatJSON is
	// supported wherever batchChat is.
	supports(kind batchKind) bool
	limits(kind batchKind) batchLimits
	// buildChat encodes one chat item with the live request builder.
	buildChat(model string, maxTokens int, req Request) ([]byte, error)
	// buildEmbed encodes one embed text with the live embed builder.
	buildEmbed(model, text string, kind EmbedKind, dims int) ([]byte, error)
	// submit uploads (if needed) and creates the batch. The create call must use
	// withNoRetry(ctx) and wrap its failures with errCreateMayExist. Returns the
	// provider's own batch id.
	submit(ctx context.Context, kind batchKind, model string, lines []batchLine) (string, error)
	status(ctx context.Context, providerID string) (BatchStatus, error)
	// results streams the provider's result lines. Called only when status is BatchDone.
	results(ctx context.Context, kind batchKind, model, providerID string) iter.Seq2[rawBatchResult, error]
	cancel(ctx context.Context, providerID string) error
}

// rawBatchResult is one result line as the backend parsed it.
type rawBatchResult struct {
	customID string
	resp     Response  // chat: from the live parser; embed: InputTokens only
	vector   []float32 // embed only, not yet normalized
	err      error     // live parser error (a refusal: resp filled) or the provider's item error
}

// errCreateMayExist wraps a create-call failure: the batch may still have been created.
func errCreateMayExist(provider string, err error) error {
	return fmt.Errorf("%s: create batch failed; the batch may still have been created, check the provider console before resubmitting: %w", provider, err)
}

// maxJSONLLine is the longest result line jsonlLines accepts.
const maxJSONLLine = 256 << 20

// jsonlLines yields each non-empty line of r. There is no line length cap below
// 256 MB: a long reply is a long line, so this is not a bufio.Scanner.
func jsonlLines(r io.Reader) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		br := bufio.NewReaderSize(r, 64<<10)
		var line []byte
		for {
			chunk, err := br.ReadSlice('\n')
			if len(line)+len(chunk) > maxJSONLLine {
				yield(nil, fmt.Errorf("llm: batch result line is over %d bytes", maxJSONLLine))
				return
			}
			line = append(line, chunk...)
			if errors.Is(err, bufio.ErrBufferFull) {
				continue // the line goes on
			}
			if t := strings.TrimSpace(string(line)); t != "" {
				if !yield([]byte(t), nil) {
					return
				}
			}
			line = line[:0]
			if err != nil {
				if err != io.EOF {
					yield(nil, err)
				}
				return
			}
		}
	}
}

// makeBatchID builds a BatchID.
func makeBatchID(p Provider, kind batchKind, model, providerID string) BatchID {
	return BatchID(string(p) + ":" + string(kind) + ":" + url.QueryEscape(model) + ":" + providerID)
}

// parseBatchID splits a BatchID. Any problem is ErrBadBatchID.
func parseBatchID(id BatchID) (p Provider, kind batchKind, model, providerID string, err error) {
	parts := strings.SplitN(string(id), ":", 4)
	if len(parts) != 4 || parts[0] == "" || parts[1] == "" || parts[2] == "" || parts[3] == "" {
		return "", "", "", "", ErrBadBatchID
	}
	switch k := batchKind(parts[1]); k {
	case batchChat, batchChatJSON, batchEmbed:
		kind = k
	default:
		return "", "", "", "", ErrBadBatchID
	}
	model, uerr := url.QueryUnescape(parts[2])
	if uerr != nil || model == "" {
		return "", "", "", "", ErrBadBatchID
	}
	return Provider(parts[0]), kind, model, parts[3], nil
}

// defaultBatchTimeout bounds a Status or Cancel call when the context has no
// deadline and Config.RequestTimeout is 0.
const defaultBatchTimeout = 60 * time.Second

// NewBatcher builds a Batcher from cfg. It reads the key fields, OpenAIBaseURL,
// MoonshotBaseURL, Profiles, EmbedModel, MaxTokens, RequestTimeout and Logger.
// Batches run on Anthropic, OpenAI (its own endpoint only), Google and Moonshot.
// If none of them has a key, it returns a Batcher that is not enabled and whose
// calls return ErrNotConfigured.
func NewBatcher(cfg Config) Batcher {
	backends := map[Provider]batchBackend{}
	// One block per provider goes here; each is added with its backend:
	//   Anthropic when cfg.AnthropicAPIKey != "" (added below)
	//   OpenAI    when cfg.OpenAIAPIKey != "" and cfg.OpenAIBaseURL == ""
	//   Google    when cfg.GoogleAPIKey != ""
	//   Moonshot  when cfg.MoonshotAPIKey != ""
	// DeepSeek never has a backend.
	if cfg.AnthropicAPIKey != "" {
		backends[ProviderAnthropic] = newAnthropicBatch(cfg.AnthropicAPIKey, cfg.Logger)
	}
	return newBatcher(cfg, backends)
}

// newBatcher is NewBatcher with the backends given (tests pass a fake).
func newBatcher(cfg Config, backends map[Provider]batchBackend) Batcher {
	if len(backends) == 0 {
		return noopBatcher{}
	}
	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1024
	}
	return &batcher{backends: backends, cfg: cfg, maxTokens: maxTokens}
}

type noopBatcher struct{}

func (noopBatcher) SubmitChat(context.Context, ChatBatch) (BatchID, error) {
	return "", ErrNotConfigured
}
func (noopBatcher) SubmitEmbed(context.Context, EmbedBatch) (BatchID, error) {
	return "", ErrNotConfigured
}
func (noopBatcher) Status(context.Context, BatchID) (BatchStatus, error) {
	return BatchStatus{}, ErrNotConfigured
}
func (noopBatcher) Cancel(context.Context, BatchID) error { return ErrNotConfigured }
func (noopBatcher) Enabled() bool                         { return false }
func (noopBatcher) Results(context.Context, BatchID) iter.Seq2[BatchResult, error] {
	return func(yield func(BatchResult, error) bool) { yield(BatchResult{}, ErrNotConfigured) }
}

type batcher struct {
	backends  map[Provider]batchBackend
	cfg       Config
	maxTokens int
}

func (b *batcher) Enabled() bool { return len(b.backends) > 0 }

// batchProfileOrder is the search order when the task's own profile is empty.
var batchProfileOrder = []Profile{ProfileFast, ProfileChat, ProfileDeep, ProfileVision, ProfileDocument, ProfileTranscribe, ProfileSearch}

// pickChatModel resolves the batch model. There is no usable() swap: a profile on a
// provider with no batch backend is an error later, not replaced.
func (b *batcher) pickChatModel(cb ChatBatch) ModelRef {
	if cb.Model != nil && !cb.Model.empty() {
		return *cb.Model
	}
	prof, ok := defaultTaskProfile[cb.Task]
	if !ok {
		prof = ProfileChat
	}
	if ref := b.cfg.Profiles[prof]; !ref.empty() {
		return ref
	}
	for _, p := range batchProfileOrder {
		if ref := b.cfg.Profiles[p]; !ref.empty() {
			return ref
		}
	}
	return ModelRef{}
}

// backendFor applies the unsupported and not-configured rules for a provider.
func (b *batcher) backendFor(p Provider, kind batchKind) (batchBackend, error) {
	switch p {
	case ProviderAnthropic, ProviderOpenAI, ProviderGoogle, ProviderMoonshot:
	default: // DeepSeek and any unknown provider
		return nil, fmt.Errorf("%w: %s", ErrBatchUnsupported, p)
	}
	if p == ProviderOpenAI && b.cfg.OpenAIBaseURL != "" {
		return nil, fmt.Errorf("%w: openai with a custom base URL", ErrBatchUnsupported)
	}
	if kind == batchEmbed && (p == ProviderAnthropic || p == ProviderMoonshot) {
		return nil, fmt.Errorf("%w: %s has no batch embeddings", ErrBatchUnsupported, p)
	}
	be, ok := b.backends[p]
	if !ok {
		return nil, ErrNotConfigured
	}
	if !be.supports(kind) {
		return nil, fmt.Errorf("%w: %s", ErrBatchUnsupported, p)
	}
	return be, nil
}

// validCustomID reports whether id matches ^[A-Za-z0-9_-]{1,64}$.
func validCustomID(id string) bool {
	if len(id) == 0 || len(id) > 64 {
		return false
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-') {
			return false
		}
	}
	return true
}

// checkLimits applies the provider caps to the built lines.
func checkLimits(lim batchLimits, lines []batchLine) error {
	if lim.maxItems > 0 && len(lines) > lim.maxItems {
		return fmt.Errorf("%w: %d items, limit %d", ErrBatchTooLarge, len(lines), lim.maxItems)
	}
	if lim.maxBytes > 0 {
		var sum int64
		for _, l := range lines {
			sum += int64(len(l.body))
		}
		if sum > lim.maxBytes {
			return fmt.Errorf("%w: %d bytes, limit %d", ErrBatchTooLarge, sum, lim.maxBytes)
		}
	}
	return nil
}

func (b *batcher) SubmitChat(ctx context.Context, cb ChatBatch) (BatchID, error) {
	ref := b.pickChatModel(cb)
	if ref.empty() {
		return "", ErrNotConfigured
	}
	be, err := b.backendFor(ref.Provider, batchChat)
	if err != nil {
		return "", err
	}
	if len(cb.Items) == 0 {
		return "", errors.New("llm: batch has no items")
	}
	seen := make(map[string]bool, len(cb.Items))
	kind := batchChat
	if len(cb.Items[0].Request.JSONSchema) > 0 {
		kind = batchChatJSON
	}
	for i, it := range cb.Items {
		if !validCustomID(it.CustomID) {
			return "", fmt.Errorf("llm: batch item %d: CustomID %q must be 1-64 characters of A-Z, a-z, 0-9, _ and -", i, it.CustomID)
		}
		if seen[it.CustomID] {
			return "", fmt.Errorf("llm: batch item %d: duplicate CustomID %q", i, it.CustomID)
		}
		seen[it.CustomID] = true
		r := it.Request
		if r.Model != nil && !r.Model.empty() && *r.Model != ref {
			return "", fmt.Errorf("llm: batch item %d: Request.Model differs from the batch model", i)
		}
		if r.Task != "" && r.Task != cb.Task {
			return "", fmt.Errorf("llm: batch item %d: Request.Task differs from the batch Task", i)
		}
		if r.OnEvent != nil {
			return "", fmt.Errorf("llm: batch item %d: OnEvent is not allowed in a batch", i)
		}
		if (len(r.JSONSchema) > 0) != (kind == batchChatJSON) {
			return "", fmt.Errorf("llm: batch item %d: batch mixes items with and without JSONSchema", i)
		}
	}
	lines := make([]batchLine, len(cb.Items))
	for i, it := range cb.Items {
		maxTokens := it.Request.MaxTokens
		if maxTokens <= 0 {
			maxTokens = b.maxTokens
		}
		body, err := be.buildChat(ref.Model, maxTokens, it.Request)
		if err != nil {
			return "", fmt.Errorf("llm: batch item %d: %w", i, err)
		}
		lines[i] = batchLine{customID: it.CustomID, body: body}
	}
	if err := checkLimits(be.limits(kind), lines); err != nil {
		return "", err
	}
	pid, err := be.submit(ctx, kind, ref.Model, lines)
	if err != nil {
		return "", err
	}
	return makeBatchID(ref.Provider, kind, ref.Model, pid), nil
}

func (b *batcher) SubmitEmbed(ctx context.Context, eb EmbedBatch) (BatchID, error) {
	ref := b.cfg.EmbedModel
	if eb.Model != nil && !eb.Model.empty() {
		ref = *eb.Model
	}
	if ref.empty() {
		return "", ErrNotConfigured
	}
	be, err := b.backendFor(ref.Provider, batchEmbed)
	if err != nil {
		return "", err
	}
	if ref.Provider == ProviderGoogle {
		ref.Model = strings.TrimPrefix(ref.Model, "models/")
	}
	if len(eb.Items) == 0 {
		return "", errors.New("llm: batch has no items")
	}
	if eb.Dimensions < 0 {
		return "", errors.New("llm: embed Dimensions is negative")
	}
	switch eb.Kind {
	case "", EmbedQuery, EmbedDocument:
	default:
		return "", fmt.Errorf("llm: unknown EmbedKind %q", eb.Kind)
	}
	seen := make(map[string]bool, len(eb.Items))
	lines := make([]batchLine, len(eb.Items))
	for i, it := range eb.Items {
		if !validCustomID(it.CustomID) {
			return "", fmt.Errorf("llm: batch item %d: CustomID %q must be 1-64 characters of A-Z, a-z, 0-9, _ and -", i, it.CustomID)
		}
		if seen[it.CustomID] {
			return "", fmt.Errorf("llm: batch item %d: duplicate CustomID %q", i, it.CustomID)
		}
		seen[it.CustomID] = true
		if it.Text == "" {
			return "", fmt.Errorf("llm: batch item %d: text is empty", i)
		}
		body, err := be.buildEmbed(ref.Model, it.Text, eb.Kind, eb.Dimensions)
		if err != nil {
			return "", fmt.Errorf("llm: batch item %d: %w", i, err)
		}
		lines[i] = batchLine{customID: it.CustomID, body: body}
	}
	if err := checkLimits(be.limits(batchEmbed), lines); err != nil {
		return "", err
	}
	pid, err := be.submit(ctx, batchEmbed, ref.Model, lines)
	if err != nil {
		return "", err
	}
	return makeBatchID(ref.Provider, batchEmbed, ref.Model, pid), nil
}

// withDeadline gives a Status or Cancel call a deadline. A caller deadline always wins.
func (b *batcher) withDeadline(ctx context.Context) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, func() {}
	}
	d := b.cfg.RequestTimeout
	if d == 0 {
		d = defaultBatchTimeout
	}
	return context.WithTimeout(ctx, d)
}

// lookup parses an ID and finds its backend.
func (b *batcher) lookup(id BatchID) (be batchBackend, kind batchKind, model, pid string, p Provider, err error) {
	p, kind, model, pid, err = parseBatchID(id)
	if err != nil {
		return
	}
	be, ok := b.backends[p]
	if !ok {
		err = ErrNotConfigured
	}
	return
}

func (b *batcher) Status(ctx context.Context, id BatchID) (BatchStatus, error) {
	be, _, _, pid, _, err := b.lookup(id)
	if err != nil {
		return BatchStatus{}, err
	}
	ctx, cancel := b.withDeadline(ctx)
	defer cancel()
	return be.status(ctx, pid)
}

func (b *batcher) Cancel(ctx context.Context, id BatchID) error {
	be, _, _, pid, _, err := b.lookup(id)
	if err != nil {
		return err
	}
	ctx, cancel := b.withDeadline(ctx)
	defer cancel()
	return be.cancel(ctx, pid)
}

func (b *batcher) Results(ctx context.Context, id BatchID) iter.Seq2[BatchResult, error] {
	return func(yield func(BatchResult, error) bool) {
		be, kind, model, pid, p, err := b.lookup(id)
		if err != nil {
			yield(BatchResult{}, err)
			return
		}
		st, err := b.Status(ctx, id)
		if err != nil {
			yield(BatchResult{}, err)
			return
		}
		switch st.State {
		case BatchFailed:
			yield(BatchResult{}, fmt.Errorf("llm: batch failed: %w", st.Err))
			return
		case BatchDone:
		default:
			yield(BatchResult{}, ErrBatchNotDone)
			return
		}
		for raw, err := range be.results(ctx, kind, model, pid) {
			if err != nil {
				yield(BatchResult{}, err)
				return
			}
			res := BatchResult{CustomID: raw.customID, Response: raw.resp, Err: raw.err}
			res.Response.Provider = p
			res.Response.Model = model
			if raw.err == nil {
				switch kind {
				case batchChatJSON:
					res.Response, res.Err = checkJSON(res.Response)
				case batchEmbed:
					if len(raw.vector) == 0 {
						res.Err = errors.New("llm: batch item has an empty vector")
					} else {
						v := append([]float32(nil), raw.vector...)
						normalizeVec(v)
						res.Vector = v
					}
				}
			}
			if !yield(res, nil) {
				return
			}
		}
	}
}
