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
			msgs = append(msgs, anthropic.NewAssistantMessage(anthropic.NewTextBlock(m.Content)))
		default: // "user"
			msgs = append(msgs, anthropic.NewUserMessage(userBlocks(m)...))
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
		// Claude's server-side web search: it runs the search itself and returns a
		// synthesized, cited answer in one request (no client tool loop needed).
		ws := anthropic.WebSearchTool20250305Param{MaxUses: anthropic.Int(5)}
		if loc := req.UserLocation; loc != nil {
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
		params.Tools = []anthropic.ToolUnionParam{{OfWebSearchTool20250305: &ws}}
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
	return Response{
		Text:         text,
		InputTokens:  int(resp.Usage.InputTokens),
		OutputTokens: int(resp.Usage.OutputTokens),
		CachedTokens: int(resp.Usage.CacheReadInputTokens),
	}, nil
}
