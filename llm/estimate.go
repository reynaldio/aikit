package llm

import (
	"encoding/json"
	"strings"
	"unicode/utf8"
)

// estimateTokens guesses the token count of s: one token per 4 runes, rounded up.
// It is the only token guess in the package.
func estimateTokens(s string) int {
	return (utf8.RuneCountInString(s) + 3) / 4
}

// estimateRequestTokens guesses the input tokens of req. It joins the text parts
// first and guesses once, so the rounding does not add up. Images and documents
// add nothing.
func estimateRequestTokens(req Request) int {
	var b strings.Builder
	b.WriteString(req.SystemCacheable)
	for _, m := range req.Messages {
		b.WriteString(m.Content)
		for _, tc := range m.ToolCalls {
			b.WriteString(tc.Name)
			b.Write(tc.Input)
		}
		for _, tr := range m.ToolResults {
			b.WriteString(tr.Content)
		}
	}
	for _, td := range req.Tools {
		b.WriteString(td.Name)
		b.WriteString(td.Description)
		if td.Schema != nil {
			if schema, err := json.Marshal(td.Schema); err == nil {
				b.Write(schema)
			}
		}
	}
	return estimateTokens(b.String())
}
