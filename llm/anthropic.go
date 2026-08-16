package llm

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicProvider is the Claude-backed provider (ADR-0008). Model-agnostic: the
// router passes the concrete model per call. It prompt-caches the reused system/memory
// context to cut cost — high leverage given Nathan injects memory every turn.
type anthropicProvider struct {
	client anthropic.Client
}

func newAnthropic(apiKey string) provider {
	return &anthropicProvider{client: anthropic.NewClient(option.WithAPIKey(apiKey))}
}

// splitSystemAndMessages folds "system" turns into the cacheable system prefix and
// converts the rest into Claude message params (user turns may carry images).
func splitSystemAndMessages(req Request) (string, []anthropic.MessageParam) {
	systemText := req.SystemCacheable
	var msgs []anthropic.MessageParam
	for _, m := range req.Messages {
		switch m.Role {
		case "system":
			if systemText != "" {
				systemText += "\n\n"
			}
			systemText += m.Content
		case "assistant":
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

// webSearchTool builds Claude's server-side web-search tool: it runs the search itself
// and returns a synthesized, cited answer in one request (no client tool loop needed).
// loc may be nil; each of its fields is optional and is sent only when set.
func webSearchTool(loc *UserLocation) anthropic.ToolUnionParam {
	ws := anthropic.WebSearchTool20250305Param{MaxUses: anthropic.Int(5)}
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

func (a *anthropicProvider) complete(ctx context.Context, model string, maxTokens int, req Request) (Response, error) {
	// Split the request into the (cacheable) system context and the turn messages.
	systemText, msgs := splitSystemAndMessages(req)

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: int64(maxTokens),
		Messages:  msgs,
	}
	if systemText != "" {
		// Cache the system/memory prefix (~90% off on cache reads).
		params.System = []anthropic.TextBlockParam{{
			Text:         systemText,
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		}}
	}
	if req.WebSearch {
		params.Tools = []anthropic.ToolUnionParam{webSearchTool(req.UserLocation)}
	}
	if tools := anthropicTools(req.Tools); len(tools) > 0 {
		params.Tools = append(params.Tools, tools...)
	}
	if oc, ok := outputConfig(req); ok {
		params.OutputConfig = oc
	}

	resp, err := a.client.Messages.New(ctx, params)
	if err != nil {
		return Response{}, err
	}

	var text string
	for _, block := range resp.Content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			text += tb.Text
		}
	}
	out := Response{
		Text:         text,
		InputTokens:  int(resp.Usage.InputTokens),
		OutputTokens: int(resp.Usage.OutputTokens),
		CachedTokens: int(resp.Usage.CacheReadInputTokens),
		ToolCalls:    anthropicToolCalls(resp.Content),
		StopReason:   anthropicStopReason(resp.StopReason),
	}
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
	case anthropic.StopReasonMaxTokens:
		return StopTruncated
	default:
		return StopEndTurn
	}
}
