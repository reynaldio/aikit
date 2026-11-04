package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
)

// anthropicProvider is the Claude-backed provider (ADR-0008). Model-agnostic: the
// router passes the concrete model per call. It prompt-caches the reused system/memory
// context to cut cost — high leverage given Nathan injects memory every turn.
type anthropicProvider struct {
	client anthropic.Client
}

func newAnthropic(apiKey string, lg *slog.Logger, opts ...option.RequestOption) provider {
	// The SDK's own retries are off so that one policy applies to every provider
	// (see docs/superpowers/specs/2026-10-13-llm-retries-design.md). The SDK sets
	// GetBody. For non-streaming calls it also puts its per-attempt timeout in
	// the request context, so the shared transport's deadline check sees it;
	// NewStreaming sets no request timeout. opts come last so a test can add
	// option.WithBaseURL.
	base := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(newProviderHTTPClient(lg)),
		option.WithMaxRetries(0),
	}
	return &anthropicProvider{client: anthropic.NewClient(append(base, opts...)...)}
}

// splitSystemAndMessages folds "system" turns into the cacheable system prefix and
// converts the rest into Claude message params (user turns may carry images).
func splitSystemAndMessages(req Request) (string, []anthropic.MessageParam) {
	systemText := req.SystemCacheable
	var msgs []anthropic.MessageParam
	echoes := anthropicEchoTurns(req)
	for i, m := range req.Messages {
		switch m.Role {
		case "system":
			if systemText != "" {
				systemText += "\n\n"
			}
			systemText += m.Content
		case "assistant":
			if t, ok := echoes[i]; ok {
				msgs = append(msgs, anthropic.NewAssistantMessage(t.blocks...))
				continue
			}
			blocks := []anthropic.ContentBlockParamUnion{}
			if m.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				// normalizeToolInput: a nil/empty Input would otherwise serialise as
				// `null` (or fail the whole request's marshal), so an echoed no-arg
				// call stays a valid `{}` object.
				blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, normalizeToolInput(tc.Input), tc.Name))
			}
			if len(blocks) > 0 {
				msgs = append(msgs, anthropic.NewAssistantMessage(blocks...))
			}
		default: // "user"
			blocks := userBlocks(m)
			blocks = append(blocks, anthropicToolResults(m.ToolResults)...)
			if len(blocks) > 0 {
				msgs = append(msgs, anthropic.NewUserMessage(blocks...))
			}
		}
	}
	return systemText, msgs
}

// userBlocks builds a user turn's content blocks — images/documents first, then text.
func userBlocks(m Message) []anthropic.ContentBlockParamUnion {
	var blocks []anthropic.ContentBlockParamUnion
	for _, img := range m.Images {
		blocks = append(blocks, anthropic.NewImageBlockBase64(img.MediaType, img.Base64))
	}
	for _, doc := range m.Documents {
		if doc.MediaType == "application/pdf" {
			blocks = append(blocks, anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{Data: doc.Base64}))
		}
	}
	if m.Content != "" {
		blocks = append(blocks, anthropic.NewTextBlock(m.Content))
	}
	return blocks
}

// anthropicProviderData is the Response.ProviderData of a Claude turn with search steps.
type anthropicProviderData struct {
	Provider Provider          `json:"provider"`
	Content  []json.RawMessage `json:"content"`
}

// anthropicTurnData returns the turn to echo, or nil when it ran no search. Claude
// needs the search blocks back with their encrypted fields. Without them it has lost
// the results and searches again, and a search that was waiting on a client tool's
// result never runs.
func anthropicTurnData(content []anthropic.ContentBlockUnion) []byte {
	need := false
	for _, block := range content {
		need = need || block.Type == "server_tool_use" || block.Type == "web_search_tool_result"
	}
	if !need {
		return nil
	}
	raw, ok := anthropicEchoJSON(content)
	if !ok {
		return nil
	}
	data, err := json.Marshal(anthropicProviderData{Provider: ProviderAnthropic, Content: raw})
	if err != nil {
		return nil
	}
	return data
}

// anthropicEchoJSON writes a reply's blocks in the form Claude takes them back. It
// builds them from the fields because RawJSON can't be sent: after a stream the SDK
// has re-marshaled the block with every field of the union. false means a block of a
// type this code does not know, which can't be sent back safely.
func anthropicEchoJSON(content []anthropic.ContentBlockUnion) ([]json.RawMessage, bool) {
	out := make([]json.RawMessage, 0, len(content))
	for _, block := range content {
		var v any
		switch block.Type {
		case "text":
			// The API rejects an empty text block, which a stream can leave behind.
			if block.Text == "" {
				continue
			}
			type citation struct {
				Type           string `json:"type"`
				URL            string `json:"url"`
				Title          string `json:"title"`
				EncryptedIndex string `json:"encrypted_index"`
				CitedText      string `json:"cited_text"`
			}
			var citations []citation
			for _, c := range block.Citations {
				if c.Type != "web_search_result_location" {
					return nil, false
				}
				citations = append(citations, citation{c.Type, c.URL, c.Title, c.EncryptedIndex, c.CitedText})
			}
			v = struct {
				Type      string     `json:"type"`
				Text      string     `json:"text"`
				Citations []citation `json:"citations,omitempty"`
			}{block.Type, block.Text, citations}
		case "thinking":
			v = struct {
				Type      string `json:"type"`
				Thinking  string `json:"thinking"`
				Signature string `json:"signature"`
			}{block.Type, block.Thinking, block.Signature}
		case "redacted_thinking":
			v = struct {
				Type string `json:"type"`
				Data string `json:"data"`
			}{block.Type, block.Data}
		case "tool_use", "server_tool_use":
			v = struct {
				Type  string          `json:"type"`
				ID    string          `json:"id"`
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
			}{block.Type, block.ID, block.Name, normalizeToolInput(block.Input)}
		case "web_search_tool_result":
			var result any
			if code := block.Content.ErrorCode; code != "" {
				result = struct {
					Type      string `json:"type"`
					ErrorCode string `json:"error_code"`
				}{"web_search_tool_result_error", string(code)}
			} else {
				type page struct {
					Type             string `json:"type"`
					URL              string `json:"url"`
					Title            string `json:"title"`
					EncryptedContent string `json:"encrypted_content"`
					PageAge          string `json:"page_age,omitempty"`
				}
				pages := []page{}
				for _, p := range block.Content.OfWebSearchResultBlockArray {
					pages = append(pages, page{string(p.Type), p.URL, p.Title, p.EncryptedContent, p.PageAge})
				}
				result = pages
			}
			v = struct {
				Type      string `json:"type"`
				ToolUseID string `json:"tool_use_id"`
				Content   any    `json:"content"`
			}{block.Type, block.ToolUseID, result}
		default:
			return nil, false
		}
		b, err := json.Marshal(v)
		if err != nil {
			return nil, false
		}
		out = append(out, b)
	}
	return out, true
}

// anthropicEchoTurn is an assistant turn that can go back as Claude sent it.
type anthropicEchoTurn struct {
	blocks  []anthropic.ContentBlockParamUnion
	pending map[string]bool // searches with no result in this turn: they run next request
	orphans map[string]bool // results whose search was in the turn before
}

// anthropicEchoBlocks reads m.ProviderData when it is a Claude turn whose tool calls
// match m.ToolCalls in order. Otherwise the caller rebuilds the turn.
func anthropicEchoBlocks(m Message) (anthropicEchoTurn, bool) {
	var none anthropicEchoTurn
	if len(m.ProviderData) == 0 {
		return none, false
	}
	var d anthropicProviderData
	if json.Unmarshal(m.ProviderData, &d) != nil || d.Provider != ProviderAnthropic || len(d.Content) == 0 {
		return none, false
	}
	t := anthropicEchoTurn{pending: map[string]bool{}, orphans: map[string]bool{}}
	n := 0
	for _, raw := range d.Content {
		var b struct {
			Type      string `json:"type"`
			ID        string `json:"id"`
			ToolUseID string `json:"tool_use_id"`
		}
		if json.Unmarshal(raw, &b) != nil {
			return none, false
		}
		switch b.Type {
		case "tool_use":
			if n >= len(m.ToolCalls) || m.ToolCalls[n].ID != b.ID {
				return none, false
			}
			n++
		case "server_tool_use":
			t.pending[b.ID] = true
		case "web_search_tool_result":
			if t.pending[b.ToolUseID] {
				delete(t.pending, b.ToolUseID)
			} else {
				t.orphans[b.ToolUseID] = true
			}
		}
		t.blocks = append(t.blocks, param.Override[anthropic.ContentBlockParamUnion](raw))
	}
	if n != len(m.ToolCalls) {
		return none, false
	}
	return t, true
}

// anthropicEchoTurns picks the assistant turns to send back as Claude sent them, by
// index in req.Messages. Search steps need the search tool (a step still waiting to
// run is a 400 without it) and must pair up across the history, so each pick is
// checked against its neighbours and any other turn is rebuilt. Dropping a turn can
// break the turn next to it, so the check repeats until nothing changes.
func anthropicEchoTurns(req Request) map[int]anthropicEchoTurn {
	if !req.WebSearch {
		return nil
	}
	// Only real turns are neighbours: system turns are folded out of the messages.
	var turns []int
	for i, m := range req.Messages {
		if m.Role != "system" {
			turns = append(turns, i)
		}
	}
	echoes := map[int]anthropicEchoTurn{}
	for _, i := range turns {
		if t, ok := anthropicEchoBlocks(req.Messages[i]); ok {
			echoes[i] = t
		}
	}
	for changed := true; changed; {
		changed = false
		for at, i := range turns {
			if t, ok := echoes[i]; ok && !anthropicEchoPairs(req.Messages, turns, at, t, echoes) {
				delete(echoes, i)
				changed = true
			}
		}
	}
	return echoes
}

// anthropicEchoPairs checks a turn's open search steps against the turns around it.
func anthropicEchoPairs(msgs []Message, turns []int, at int, t anthropicEchoTurn, echoes map[int]anthropicEchoTurn) bool {
	// A search waiting to run needs a next message of tool results only. Anything
	// else, consecutive user turns included since they merge, ends the turn with the
	// search unresolved. If the reply to it is in the history, it must hold the results.
	if len(t.pending) > 0 {
		if at+1 >= len(turns) {
			return false
		}
		next := msgs[turns[at+1]]
		if next.Role != "user" || len(next.ToolResults) == 0 || next.Content != "" || len(next.Images) > 0 || len(next.Documents) > 0 {
			return false
		}
		if at+2 < len(turns) {
			reply, ok := echoes[turns[at+2]]
			if !ok {
				return false
			}
			for id := range t.pending {
				if !reply.orphans[id] {
					return false
				}
			}
		}
	}
	// A result whose search came earlier needs that search in the turn before.
	if len(t.orphans) > 0 {
		if at < 2 {
			return false
		}
		before, ok := echoes[turns[at-2]]
		if !ok {
			return false
		}
		for id := range t.orphans {
			if !before.pending[id] {
				return false
			}
		}
	}
	return true
}

// defaultWebSearchMaxUses caps Claude's searches when Request.WebSearchMaxUses is 0.
const defaultWebSearchMaxUses = 5

// webSearchTool builds Claude's server-side web-search tool: it runs the search itself
// and returns a synthesized, cited answer in one request (no client tool loop needed).
// loc may be nil; each of its fields is optional and is sent only when set.
func webSearchTool(loc *UserLocation, maxUses int) anthropic.ToolUnionParam {
	if maxUses <= 0 {
		maxUses = defaultWebSearchMaxUses
	}
	ws := anthropic.WebSearchTool20250305Param{MaxUses: anthropic.Int(int64(maxUses))}
	if loc != nil {
		ul := anthropic.UserLocationParam{}
		if loc.City != "" {
			ul.City = anthropic.String(loc.City)
		}
		if loc.Region != "" {
			ul.Region = anthropic.String(loc.Region)
		}
		if loc.Country != "" {
			ul.Country = anthropic.String(loc.Country)
		}
		if loc.Timezone != "" {
			ul.Timezone = anthropic.String(loc.Timezone)
		}
		ws.UserLocation = ul
	}
	return anthropic.ToolUnionParam{OfWebSearchTool20250305: &ws}
}

// outputConfig maps effort and structured outputs onto Claude's single output_config
// field. Reports false when the caller asked for neither, so the request is sent
// without the field and the model's own defaults apply.
func outputConfig(req Request) (anthropic.OutputConfigParam, bool) {
	if req.Effort == "" && len(req.JSONSchema) == 0 {
		return anthropic.OutputConfigParam{}, false
	}
	oc := anthropic.OutputConfigParam{}
	if req.Effort != "" {
		oc.Effort = anthropic.OutputConfigEffort(req.Effort)
	}
	if len(req.JSONSchema) > 0 {
		oc.Format = anthropic.JSONOutputFormatParam{Schema: req.JSONSchema}
	}
	return oc, true
}

// applyHistoryCache marks the last content block of the last message cacheable,
// so the next round's identical prefix is a cache READ rather than a full-price
// resend. Only the final block is marked: Anthropic allows 4 breakpoints per
// request, the system prefix already spends one, and a breakpoint earlier in the
// history buys nothing the rolling one at the end does not already cover.
//
// GetCacheControl returns a pointer into whichever block variant is set (text,
// tool_result, image, ...), so this works uniformly without a type switch. It
// returns nil for a block that cannot carry cache control, which is a no-op.
func applyHistoryCache(msgs []anthropic.MessageParam) {
	if len(msgs) == 0 {
		return
	}
	blocks := msgs[len(msgs)-1].Content
	if len(blocks) == 0 {
		return
	}
	if cc := blocks[len(blocks)-1].GetCacheControl(); cc != nil {
		*cc = anthropic.NewCacheControlEphemeralParam()
	}
}

// anthropicParams builds the request body for one chat call. The live path and
// the batch path both use it, so their bodies match.
func anthropicParams(model string, maxTokens int, req Request) anthropic.MessageNewParams {
	// Split the request into the (cacheable) system context and the turn messages.
	systemText, msgs := splitSystemAndMessages(req)
	if req.CacheHistory {
		applyHistoryCache(msgs)
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: int64(maxTokens),
		Messages:  msgs,
	}
	if systemText != "" {
		params.System = []anthropic.TextBlockParam{{Text: systemText}}
		if !req.NoSystemCache {
			// Cache the system/memory prefix (~90% off on cache reads).
			params.System[0].CacheControl = anthropic.NewCacheControlEphemeralParam()
		}
	}
	if req.WebSearch {
		params.Tools = []anthropic.ToolUnionParam{webSearchTool(req.UserLocation, req.WebSearchMaxUses)}
	}
	if tools := anthropicTools(req.Tools); len(tools) > 0 {
		params.Tools = append(params.Tools, tools...)
	}
	if oc, ok := outputConfig(req); ok {
		params.OutputConfig = oc
	}
	return params
}

// maxPauseContinuations caps how many times one call continues a paused turn. A turn
// still paused after that comes back as StopTruncated.
const maxPauseContinuations = 5

func (a *anthropicProvider) complete(ctx context.Context, model string, maxTokens int, req Request) (Response, error) {
	params := anthropicParams(model, maxTokens, req)

	// A long search turn can come back as pause_turn. It is not finished: send the
	// paused content back as it is, and the next reply holds only what is new.
	var paused *anthropic.Message // every paused reply so far, as one message
	for n := 0; ; n++ {
		msg, sawDelta, err := a.send(ctx, params, req.OnEvent)
		if err != nil {
			err = anthropicError(ctx, err)
			// What was paused, and a stream that ended early, were still billed.
			var r Response
			if paused != nil {
				r = anthropicPartial(paused, true)
			}
			if msg != nil {
				r = addPartial(r, anthropicPartial(msg, sawDelta))
			}
			return r, err
		}
		if paused != nil {
			msg = joinPaused(paused, msg)
		}
		if msg.StopReason != anthropic.StopReasonPauseTurn || n == maxPauseContinuations {
			return anthropicResponse(model, msg)
		}
		last := msg.Content
		if paused != nil {
			last = msg.Content[len(paused.Content):]
		}
		raw, ok := anthropicEchoJSON(last)
		if !ok || len(raw) == 0 {
			return anthropicResponse(model, msg)
		}
		blocks := make([]anthropic.ContentBlockParamUnion, 0, len(raw))
		for _, b := range raw {
			blocks = append(blocks, param.Override[anthropic.ContentBlockParamUnion](b))
		}
		params.Messages = append(params.Messages, anthropic.NewAssistantMessage(blocks...))
		paused = msg
	}
}

// joinPaused adds a continuation to the paused replies: its content follows theirs,
// its usage is added, and its stop reason is the turn's.
func joinPaused(paused, next *anthropic.Message) *anthropic.Message {
	joined := *next
	joined.Content = append(append([]anthropic.ContentBlockUnion(nil), paused.Content...), next.Content...)
	joined.Usage.InputTokens += paused.Usage.InputTokens
	joined.Usage.OutputTokens += paused.Usage.OutputTokens
	joined.Usage.CacheReadInputTokens += paused.Usage.CacheReadInputTokens
	joined.Usage.CacheCreationInputTokens += paused.Usage.CacheCreationInputTokens
	joined.Usage.ServerToolUse.WebSearchRequests += paused.Usage.ServerToolUse.WebSearchRequests
	return &joined
}

// addPartial adds an unfinished continuation to the paused replies before it.
func addPartial(paused, next Response) Response {
	next.Text = paused.Text + next.Text
	next.InputTokens += paused.InputTokens
	next.OutputTokens += paused.OutputTokens
	next.CachedTokens += paused.CachedTokens
	next.CacheWriteTokens += paused.CacheWriteTokens
	next.WebSearchRequests += paused.WebSearchRequests
	next.Citations = dedupeCitations(append(paused.Citations, next.Citations...))
	return next
}

// anthropicError wraps an SDK error in a *ProviderError: an API error with its
// status, or a transport failure as Status 0. An error event mid-stream comes with
// the stream's own 200, so its status is taken from its error type instead. Other
// errors, and any error once ctx has ended, are returned as they are.
func anthropicError(ctx context.Context, err error) error {
	if err == nil || ctx.Err() != nil {
		return err
	}
	var ae *anthropic.Error
	if errors.As(err, &ae) {
		status := ae.StatusCode
		if status < 300 {
			status = anthropicTypeStatus(string(ae.Type()))
		}
		return statusError(ProviderAnthropic, status, err)
	}
	if isTransport(err) {
		return statusError(ProviderAnthropic, 0, err)
	}
	return err
}

// send issues the request, non-streaming when the SDK allows it and as a stream
// otherwise.
//
// The SDK refuses a non-streaming Messages.New whose estimated duration exceeds 10
// minutes ("streaming is required for operations that may take longer than 10
// minutes"): with no request timeout it estimates 1h × max_tokens / 128000, so any
// max_tokens above ~21333 — or above a model's own non-streaming cap — fails before
// a byte is sent. Asking the SDK's own exported check, with the client's options,
// keeps this in lockstep with whatever rule the pinned SDK version applies.
//
// A caller that wants text as it arrives (onEvent != nil) always takes the stream.
//
// On error, the message is the partial one a stream built so far (nil when it took
// no event). The non-streaming call never returns a partial message.
func (a *anthropicProvider) send(ctx context.Context, params anthropic.MessageNewParams, onEvent func(StreamEvent)) (msg *anthropic.Message, sawDelta bool, err error) {
	if onEvent != nil {
		return a.stream(ctx, params, onEvent)
	}
	if _, err := anthropic.CalculateNonStreamingTimeout(int(params.MaxTokens), params.Model, a.client.Options); err != nil {
		return a.stream(ctx, params, nil)
	}
	msg, err = a.client.Messages.New(ctx, params)
	if err != nil {
		return nil, false, err
	}
	return msg, false, nil
}

// stream sends params as a stream and assembles the events into the same Message a
// non-streaming call returns, so the caller cannot tell the two apart. A stream
// that ends before message_stop (a dropped connection, a proxy cutting the body)
// is an error, never a silently partial reply. Once message_stop has arrived the
// reply is whole and billed, so the loop stops reading: nothing after it — a
// connection held open, a cancel landing meanwhile — can discard it.
//
// When onEvent is set, each non-empty text_delta is passed on once Accumulate has
// taken it, so the pieces sent are exactly what anthropicResponse later joins into
// Text. Thinking, signature, tool-input and citation deltas are never sent. Client
// tool_use blocks also send tool_start and tool_ready (see anthropicEvents).
//
// On every error return the message built so far comes back too (nil when no event
// was taken), with sawDelta, so the caller can report the usage already billed.
// sawDelta says a message_delta was taken, so the output count is the real one.
func (a *anthropicProvider) stream(ctx context.Context, params anthropic.MessageNewParams, onEvent func(StreamEvent)) (*anthropic.Message, bool, error) {
	s := a.client.Messages.NewStreaming(ctx, params)
	defer func() { _ = s.Close() }()

	var (
		msg      anthropic.Message
		n        int // events that Accumulate took
		sawDelta bool
	)
	partial := func() *anthropic.Message {
		if n > 0 {
			return &msg
		}
		return nil
	}
	for s.Next() {
		ev := s.Current()
		repairToolInputs(&msg, ev)
		if err := msg.Accumulate(ev); err != nil {
			return partial(), sawDelta, err
		}
		n++
		if ev.Type == "message_delta" {
			sawDelta = true
		}
		if onEvent != nil {
			for _, out := range anthropicEvents(&msg, ev) {
				onEvent(out)
				// The SDK stream never looks at ctx, so a callback that cancels it must
				// stop the loop here, even when the next events are already buffered.
				if err := ctx.Err(); err != nil {
					return partial(), sawDelta, err
				}
			}
		}
		if ev.Type == "message_stop" {
			return &msg, sawDelta, nil
		}
	}
	if err := s.Err(); err != nil {
		return partial(), sawDelta, err
	}
	if err := ctx.Err(); err != nil {
		return partial(), sawDelta, err
	}
	// Wrapping io.ErrUnexpectedEOF classifies this like a connection dropped
	// mid-body on the non-streaming path: transient, so failover may retry it.
	return partial(), sawDelta, fmt.Errorf("anthropic: stream ended before message_stop: %w", io.ErrUnexpectedEOF)
}

// anthropicPartial maps a message that a stream left unfinished onto a Response
// with text and token counts only. Input and cache counts come from message_start
// and are real. The output count in message_start is a placeholder, so unless a
// message_delta was taken it is replaced by a guess over everything produced.
func anthropicPartial(msg *anthropic.Message, sawDelta bool) Response {
	r := anthropicUsage(msg.Usage)
	var text, all string
	for _, block := range msg.Content {
		// Read the fields, not AsAny: an unfinished block has not been re-marshaled.
		if block.Type == "text" {
			text += block.Text
		}
		all += block.Text + block.Thinking + string(block.Input)
	}
	r.Text = text
	r.Citations = anthropicCitations(msg.Content)
	if !sawDelta {
		r.OutputTokens = estimateTokens(all)
		r.UsageEstimated = true
		r.WebSearchRequests = anthropicSearchResults(msg.Content)
	}
	return r
}

// anthropicEvents maps one stream event, already taken by Accumulate, onto the
// events for Request.OnEvent: text from a non-empty text_delta or from the
// initial text of a text block, tool_start when a client tool_use block opens,
// and tool_ready when it closes. Server tool blocks (web search) and thinking
// produce nothing.
func anthropicEvents(msg *anthropic.Message, ev anthropic.MessageStreamEventUnion) []StreamEvent {
	switch ev.Type {
	case "content_block_start":
		switch cb := ev.ContentBlock; cb.Type {
		case "tool_use":
			return []StreamEvent{{Kind: StreamToolStart, ToolCallID: cb.ID, ToolName: cb.Name}}
		case "text":
			if cb.Text != "" {
				return []StreamEvent{{Kind: StreamText, Text: cb.Text}}
			}
		}
	case "content_block_delta":
		if ev.Delta.Type == "text_delta" && ev.Delta.Text != "" {
			return []StreamEvent{{Kind: StreamText, Text: ev.Delta.Text}}
		}
	case "content_block_stop":
		if i := ev.Index; i >= 0 && i < int64(len(msg.Content)) && msg.Content[i].Type == "tool_use" {
			return []StreamEvent{{Kind: StreamToolReady, ToolCallID: msg.Content[i].ID, ToolName: msg.Content[i].Name}}
		}
	}
	return nil
}

// streams reports that this provider sends text to Request.OnEvent as it arrives.
func (a *anthropicProvider) streams() bool { return true }

var _ streamer = (*anthropicProvider)(nil)

// repairToolInputs runs before Accumulate sees a content_block_stop or
// message_stop, the two events on which the SDK re-marshals the accumulated
// blocks. A reply cut off by max_tokens mid tool input leaves that block's
// input_json_delta concatenation as invalid JSON, and the re-marshal would turn
// an ordinary truncation into a hard error with no usage. The non-streaming path
// reports the same situation as StopTruncated, so the half-written input is
// replaced with {} and the message (stop reason, usage) survives intact.
func repairToolInputs(msg *anthropic.Message, ev anthropic.MessageStreamEventUnion) {
	fix := func(cb *anthropic.ContentBlockUnion) {
		if len(cb.Input) > 0 && !json.Valid(cb.Input) {
			cb.Input = json.RawMessage("{}")
		}
	}
	switch ev.Type {
	case "content_block_stop":
		if i := ev.Index; i >= 0 && i < int64(len(msg.Content)) {
			fix(&msg.Content[i])
		}
	case "message_stop":
		for i := range msg.Content {
			fix(&msg.Content[i])
		}
	}
}

// anthropicResponse maps a complete Claude message onto Response. Both the
// non-streaming and the streaming path end here, so they report identically.
func anthropicResponse(model string, resp *anthropic.Message) (Response, error) {
	var text string
	for _, block := range resp.Content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			text += tb.Text
		}
	}
	out := anthropicUsage(resp.Usage)
	out.Text = text
	out.Citations = anthropicCitations(resp.Content)
	out.ProviderData = anthropicTurnData(resp.Content)
	out.ToolCalls = anthropicToolCalls(resp.Content)
	out.StopReason = anthropicStopReason(resp.StopReason)
	// A safety refusal arrives as a successful 200 with an empty or partial body, so it
	// must be turned into an error here — otherwise every caller silently receives "".
	// The partial text and the usage still ride along: a mid-stream refusal bills what it
	// streamed, and callers that meter cost should see it.
	if resp.StopReason == anthropic.StopReasonRefusal {
		return out, &RefusalError{
			Provider:    ProviderAnthropic,
			Model:       model,
			Category:    string(resp.StopDetails.Category),
			Explanation: resp.StopDetails.Explanation,
		}
	}
	return out, nil
}

// anthropicSearchResults counts the searches that returned, for a stream that ended
// before the real count (it only comes in message_delta). A failed search is not
// billed. A search still running when the stream ended is not seen.
func anthropicSearchResults(content []anthropic.ContentBlockUnion) int {
	n := 0
	for _, block := range content {
		if block.Type == "web_search_tool_result" && block.Content.ErrorCode == "" {
			n++
		}
	}
	return n
}

// anthropicCitations collects the web-search citations of the text blocks. It reads
// the fields, not AsAny, so it also works on an unfinished block.
func anthropicCitations(content []anthropic.ContentBlockUnion) []Citation {
	var all []Citation
	for _, block := range content {
		if block.Type != "text" {
			continue
		}
		for _, c := range block.Citations {
			if c.Type == "web_search_result_location" {
				all = append(all, Citation{URL: c.URL, Title: c.Title, CitedText: c.CitedText})
			}
		}
	}
	return dedupeCitations(all)
}

// anthropicInputSchema maps a JSON Schema onto the SDK's tool input schema.
// `properties` and `required` have typed fields (`required` arrives as []string
// from a Go literal but []any once it has been through JSON, so both are
// accepted); EVERY OTHER key rides through ExtraFields verbatim.
//
// Forwarding the rest is not tidiness. OpenAI and Google pass ToolDef.Schema
// wholesale, so a schema using $defs/$ref/additionalProperties/enum that worked
// on those two would arrive at Claude with its definitions stripped and its
// $refs dangling — one tool, two behaviours, no error anywhere.
//
// `type` is excluded because the SDK owns it (constant "object").
func anthropicInputSchema(schema map[string]any) anthropic.ToolInputSchemaParam {
	out := anthropic.ToolInputSchemaParam{}
	if props, ok := schema["properties"]; ok {
		out.Properties = props
	}
	switch req := schema["required"].(type) {
	case []string:
		out.Required = req
	case []any:
		for _, r := range req {
			if s, ok := r.(string); ok {
				out.Required = append(out.Required, s)
			}
		}
	}
	for k, v := range schema {
		switch k {
		case "properties", "required", "type":
			continue
		}
		if out.ExtraFields == nil {
			out.ExtraFields = make(map[string]any, len(schema))
		}
		out.ExtraFields[k] = v
	}
	return out
}

// anthropicTools maps tool declarations onto the SDK's tool union.
func anthropicTools(defs []ToolDef) []anthropic.ToolUnionParam {
	if len(defs) == 0 {
		return nil
	}
	out := make([]anthropic.ToolUnionParam, 0, len(defs))
	for _, d := range defs {
		t := anthropic.ToolParam{
			Name:        d.Name,
			InputSchema: anthropicInputSchema(d.Schema),
		}
		if d.Description != "" {
			t.Description = anthropic.String(d.Description)
		}
		out = append(out, anthropic.ToolUnionParam{OfTool: &t})
	}
	return out
}

// anthropicToolResults renders one round's results as content blocks. The caller
// puts them all in a single user turn.
func anthropicToolResults(results []ToolResult) []anthropic.ContentBlockParamUnion {
	out := make([]anthropic.ContentBlockParamUnion, 0, len(results))
	for _, r := range results {
		out = append(out, anthropic.NewToolResultBlock(r.ToolCallID, r.Content, r.IsError))
	}
	return out
}

// anthropicToolCalls pulls tool_use blocks out of a reply.
func anthropicToolCalls(content []anthropic.ContentBlockUnion) []ToolCall {
	var out []ToolCall
	for _, block := range content {
		if tu, ok := block.AsAny().(anthropic.ToolUseBlock); ok {
			out = append(out, ToolCall{ID: tu.ID, Name: tu.Name, Input: normalizeToolInput(tu.Input)})
		}
	}
	return out
}

// anthropicStopReason maps the SDK's stop reason onto ours.
//
// A refusal DOES reach this function — the response is returned with its
// RefusalError, not instead of one — and falls to the default arm, so a refused
// turn reports StopEndTurn. That is deliberate rather than a mapping gap: the
// error takes precedence, and Response.StopReason is only meaningful when
// Complete returned err == nil. Adding a StopRefused would invite callers to
// branch on the stop reason of a call that failed.
func anthropicStopReason(sr anthropic.StopReason) StopReason {
	switch sr {
	case anthropic.StopReasonToolUse:
		return StopToolUse
	case anthropic.StopReasonMaxTokens, anthropic.StopReasonPauseTurn:
		// A paused turn that was not continued (a batch item, or past the limit) is
		// incomplete in the same way.
		return StopTruncated
	default:
		return StopEndTurn
	}
}

// anthropicUsage maps Claude's four reported token dimensions onto Response.
// Claude reports cache reads and cache writes SEPARATELY from InputTokens, and
// each bills at its own rate — folding either into InputTokens would misprice
// the call in both directions (reads cost ~0.1x, writes ~1.25x).
func anthropicUsage(u anthropic.Usage) Response {
	return Response{
		InputTokens:       int(u.InputTokens),
		OutputTokens:      int(u.OutputTokens),
		CachedTokens:      int(u.CacheReadInputTokens),
		CacheWriteTokens:  int(u.CacheCreationInputTokens),
		WebSearchRequests: int(u.ServerToolUse.WebSearchRequests),
	}
}
