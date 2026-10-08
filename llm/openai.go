package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
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
	timeout time.Duration // Config.RequestTimeout: deadline for a ctx without one; 0 = scaled
	// strictSchema sends Request.JSONSchema as response_format json_schema with
	// strict:true. Only OpenAI's own endpoint gets it: DeepSeek rejects json_schema
	// with a 400, Moonshot supports it only on some Kimi models, and a custom
	// OpenAIBaseURL could be any backend. Those stay on the router's JSON check.
	strictSchema bool
}

const openaiBaseURL = "https://api.openai.com/v1"

func newOpenAI(apiKey, baseURL string, timeout time.Duration) provider {
	if baseURL == "" {
		baseURL = openaiBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")
	return &openaiProvider{
		apiKey:       apiKey,
		baseURL:      baseURL,
		http:         newProviderHTTPClient(),
		timeout:      timeout,
		strictSchema: baseURL == openaiBaseURL,
	}
}

// enforcesJSONSchema reports whether this backend will constrain the reply to
// schema: OpenAI itself, and only for a schema strict mode accepts.
func (o *openaiProvider) enforcesJSONSchema(schema map[string]any) bool {
	if !o.strictSchema {
		return false
	}
	_, ok := oaiStrictSchema(schema)
	return ok
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
	MaxTokens           int                `json:"max_tokens,omitempty"`
	MaxCompletionTokens int                `json:"max_completion_tokens,omitempty"`
	Messages            []oaiMessage       `json:"messages"`
	Tools               []oaiTool          `json:"tools,omitempty"`
	ResponseFormat      *oaiResponseFormat `json:"response_format,omitempty"`
}

// oaiResponseFormat is response_format {type:"json_schema", json_schema:{…}}.
type oaiResponseFormat struct {
	Type       string        `json:"type"`
	JSONSchema oaiJSONSchema `json:"json_schema"`
}

type oaiJSONSchema struct {
	Name   string         `json:"name"` // ^[a-zA-Z0-9_-]{1,64}$
	Schema map[string]any `json:"schema"`
	Strict bool           `json:"strict"`
}

// oaiSchemaFormat returns the response_format for req, or nil to send none: no
// schema, a backend that may not take json_schema, or a schema strict mode
// rejects. Non-strict json_schema is never sent — it is advice, not enforcement,
// and the router's JSON check already covers the unenforced case.
func (o *openaiProvider) oaiSchemaFormat(req Request) *oaiResponseFormat {
	if len(req.JSONSchema) == 0 || !o.strictSchema {
		return nil
	}
	strict, ok := oaiStrictSchema(req.JSONSchema)
	if !ok {
		return nil
	}
	return &oaiResponseFormat{Type: "json_schema", JSONSchema: oaiJSONSchema{Name: "response", Schema: strict, Strict: true}}
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
	ctx, cancel := withDefaultDeadline(ctx, o.timeout, maxTokens)
	defer cancel()
	oaiReq := oaiRequest{Model: model, Messages: oaiMessages(req), Tools: oaiTools(req.Tools), ResponseFormat: o.oaiSchemaFormat(req)}
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
	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return Response{}, readError(ctx, "openai", err)
	}

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

// oaiStrictUnsupported are the keywords OpenAI's strict mode rejects outright. A
// schema using one cannot be enforced there, so it is not sent at all.
var oaiStrictUnsupported = map[string]bool{
	"allOf": true, "not": true, "if": true, "then": true, "else": true,
	"dependentRequired": true, "dependentSchemas": true,
	"patternProperties": true, "unevaluatedProperties": true,
}

// oaiStrictSchema rewrites a caller's schema into the form strict mode requires,
// or reports false when it cannot be expressed there. It returns a copy; the
// caller's map is never mutated.
//
// Strict mode wants every object closed (additionalProperties:false) and every
// property required. A property the caller left OPTIONAL is therefore made
// required-but-nullable — OpenAI's documented spelling of "optional" — so the
// model sends null instead of omitting the key. Unmarshalling into Go treats both
// the same. An object the caller left open (additionalProperties true or a
// schema), a root that is not an object, or a root anyOf cannot be expressed.
func oaiStrictSchema(schema map[string]any) (map[string]any, bool) {
	if !schemaHasType(schema, "object") {
		return nil, false
	}
	if _, has := schema["anyOf"]; has {
		return nil, false
	}
	out, ok := oaiStrictNode(schema)
	if !ok {
		return nil, false
	}
	return out.(map[string]any), true
}

// oaiStrictNode converts one subschema position: a schema object is rewritten,
// a list of subschemas is converted element-wise, anything else is a literal.
func oaiStrictNode(v any) (any, bool) {
	switch n := v.(type) {
	case map[string]any:
		return oaiStrictObject(n)
	case []map[string]any:
		list := make([]any, len(n))
		for i, s := range n {
			list[i] = s
		}
		return oaiStrictNode(list)
	case []any:
		out := make([]any, len(n))
		for i, s := range n {
			c, ok := oaiStrictNode(s)
			if !ok {
				return nil, false
			}
			out[i] = c
		}
		return out, true
	default:
		return v, true
	}
}

func oaiStrictObject(s map[string]any) (any, bool) {
	out := make(map[string]any, len(s)+2)
	for k, v := range s {
		if oaiStrictUnsupported[k] {
			return nil, false
		}
		switch k {
		case "properties", "$defs":
			byName, ok := oaiStrictByName(v)
			if !ok {
				return nil, false
			}
			out[k] = byName
		case "items", "anyOf", "prefixItems":
			c, ok := oaiStrictNode(v)
			if !ok {
				return nil, false
			}
			out[k] = c
		default:
			out[k] = v // keywords and literal values (enum, required, …)
		}
	}
	if !schemaHasType(s, "object") && s["properties"] == nil {
		return out, true
	}
	if ap, has := s["additionalProperties"]; has && ap != false {
		return nil, false // an open object or a map: strict mode can't say that
	}
	out["additionalProperties"] = false
	oaiRequireAll(out)
	return out, true
}

// oaiStrictByName converts a NAME -> subschema map; the names are kept as-is.
func oaiStrictByName(v any) (any, bool) {
	m, ok := v.(map[string]any)
	if !ok {
		return v, true
	}
	out := make(map[string]any, len(m))
	for name, sub := range m {
		c, ok := oaiStrictNode(sub)
		if !ok {
			return nil, false
		}
		out[name] = c
	}
	return out, true
}

// oaiRequireAll lists every property in required, making each one the caller had
// not required nullable. The caller's own required order is kept, and the newly
// required names follow in sorted order so the request body is deterministic.
func oaiRequireAll(obj map[string]any) {
	props, _ := obj["properties"].(map[string]any)
	required := []any{}
	seen := map[string]bool{}
	for _, r := range schemaStrings(obj["required"]) {
		if _, ok := props[r]; ok && !seen[r] {
			required = append(required, r)
			seen[r] = true
		}
	}
	var optional []string
	for name := range props {
		if !seen[name] {
			optional = append(optional, name)
		}
	}
	sort.Strings(optional)
	for _, name := range optional {
		props[name] = schemaNullable(props[name])
		required = append(required, name)
	}
	obj["required"] = required
}

// schemaNullable widens a subschema to also accept null.
func schemaNullable(v any) any {
	s, ok := v.(map[string]any)
	if !ok {
		return v
	}
	_, isRef := s["$ref"]
	t, hasType := s["type"]
	if isRef || !hasType {
		// $ref may carry no siblings, and an untyped schema (anyOf, enum alone)
		// has no type to widen: say "this, or null" as a union instead.
		return map[string]any{"anyOf": []any{s, map[string]any{"type": "null"}}}
	}
	if schemaHasType(s, "null") {
		return s
	}
	out := make(map[string]any, len(s))
	for k, v := range s {
		out[k] = v
	}
	switch t := t.(type) {
	case string:
		out["type"] = []any{t, "null"}
	default:
		out["type"] = append(schemaAnyList(t), "null")
	}
	if enum, ok := s["enum"]; ok {
		out["enum"] = append(schemaAnyList(enum), nil) // null must be an allowed value too
	}
	return out
}

// schemaHasType reports whether a schema's type is, or includes, want.
func schemaHasType(s map[string]any, want string) bool {
	for _, t := range schemaStrings(s["type"]) {
		if t == want {
			return true
		}
	}
	return false
}

// schemaStrings reads a string or list-of-strings keyword (type, required).
func schemaStrings(v any) []string {
	switch v := v.(type) {
	case string:
		return []string{v}
	case []string:
		return v
	case []any:
		var out []string
		for _, e := range v {
			if s, ok := e.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

// schemaAnyList copies a list keyword as []any, whether it arrived decoded or as a
// typed Go literal ([]string), so appending never aliases the caller's slice.
func schemaAnyList(v any) []any {
	switch v := v.(type) {
	case []any:
		return append([]any(nil), v...)
	case []string:
		out := make([]any, len(v))
		for i, s := range v {
			out[i] = s
		}
		return out
	}
	return []any{v}
}
