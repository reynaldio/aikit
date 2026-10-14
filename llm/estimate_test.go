package llm

import (
	"encoding/json"
	"testing"
)

func TestEstimateTokens(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{{"", 0}, {"abc", 1}, {"abcd", 1}, {"abcde", 2}, {"éé", 1}, {"éééé", 1}, {"ééééé", 2}}
	for _, tc := range cases {
		if got := estimateTokens(tc.in); got != tc.want {
			t.Errorf("estimateTokens(%q) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

func TestEstimateRequestTokens(t *testing.T) {
	schema := map[string]any{"type": "object"}
	sj, _ := json.Marshal(schema)
	req := Request{
		SystemCacheable: "sys",
		Messages: []Message{
			{Content: "hello", ToolCalls: []ToolCall{{Name: "find", Input: json.RawMessage(`{"q":1}`)}}},
			{ToolResults: []ToolResult{{Content: "result"}}},
		},
		Tools: []ToolDef{{Name: "find", Description: "finds", Schema: schema}},
	}
	joined := "sys" + "hello" + "find" + `{"q":1}` + "result" + "find" + "finds" + string(sj)
	if got, want := estimateRequestTokens(req), estimateTokens(joined); got != want {
		t.Fatalf("got %d, want %d", got, want)
	}
}

func TestEstimateRequestTokensImageOnly(t *testing.T) {
	req := Request{Messages: []Message{{
		Images:    []Image{{Base64: "AAAAAAAAAAAAAAAA", MediaType: "image/png"}},
		Documents: []Document{{Base64: "BBBBBBBB", MediaType: "application/pdf"}},
	}}}
	if got := estimateRequestTokens(req); got != 0 {
		t.Fatalf("got %d, want 0", got)
	}
}
