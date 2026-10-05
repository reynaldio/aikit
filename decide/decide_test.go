package decide

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// serve stands up a fake System One endpoint that records the request body and
// replies with status + body.
func serve(t *testing.T, status int, body string) (Client, *map[string]any, *http.Header) {
	t.Helper()
	got := map[string]any{}
	hdr := http.Header{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/systemone" {
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
		hdr = r.Header.Clone()
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Errorf("request is not JSON: %v", err)
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return New(Config{APIKey: "k", BaseURL: srv.URL + "/v1/"}), &got, &hdr
}

func TestEvaluateRequestShape(t *testing.T) {
	c, got, hdr := serve(t, 200, `{"model":"jev-1.13.0","answers":{"urgent":{"type":"noul","noul":0.9},"dept":{"type":"choice","choice":"billing"},"mood":{"type":"score","score":1}},"usage":{"input_tokens":1,"output_tokens":1}}`)
	_, err := c.Evaluate(context.Background(), Request{
		State: map[string]any{"ticket": "refund please"},
		Questions: map[string]Question{
			"urgent": Noul("Is this urgent?", "needs action today", ""),
			"dept":   Choice("Which team?", map[string]string{"billing": "money", "technical": "bugs"}),
			"mood":   Score("How upset?", "Calm", "Frustrated", "Very angry"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if a := hdr.Get("Authorization"); a != "Bearer k" {
		t.Errorf("Authorization = %q", a)
	}
	want := `{"model":"jev-latest","questions":{"dept":{"criteria":{"billing":"money","technical":"bugs"},"instructions":"Which team?","type":"choice"},"mood":{"criteria":["Calm","Frustrated","Very angry"],"instructions":"How upset?","type":"score"},"urgent":{"criteria":{"true":"needs action today"},"instructions":"Is this urgent?","type":"noul"}},"state":{"ticket":"refund please"}}`
	if b, _ := json.Marshal(*got); string(b) != want {
		t.Errorf("request body\n got %s\nwant %s", b, want)
	}
}

func TestNoulWithoutCriteriaOmitsField(t *testing.T) {
	b, _ := json.Marshal(Noul("Is it spam?", "", ""))
	if strings.Contains(string(b), "criteria") {
		t.Errorf("empty criteria should be omitted: %s", b)
	}
}

func TestEvaluateParsesEachAnswerType(t *testing.T) {
	c, _, _ := serve(t, 200, `{
		"model": "jev-1.13.0",
		"answers": {
			"urgent": {"type": "noul", "noul": 0.95},
			"dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.88, "technical": 0.12}, "confidence": 0.81},
			"mood": {"type": "score", "score": 1.05, "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"}, "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05}, "confidence": 0.92}
		},
		"usage": {"input_tokens": 318, "output_tokens": 34}
	}`)
	resp, err := c.Evaluate(context.Background(), Request{
		Model: "jev-1.13",
		State: "I was charged twice",
		Questions: map[string]Question{
			"urgent": Noul("Is this urgent?", "", ""),
			"dept":   Choice("Which team?", map[string]string{"billing": "", "technical": ""}),
			"mood":   Score("How upset?", "Calm", "Frustrated", "Very angry"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Model != "jev-1.13.0" || resp.InputTokens != 318 || resp.OutputTokens != 34 {
		t.Errorf("meta = %+v", resp)
	}
	if a := resp.Answers["urgent"]; a.Type != TypeNoul || a.Noul != 0.95 {
		t.Errorf("noul = %+v", a)
	}
	if a := resp.Answers["dept"]; a.Choice != "billing" || a.Probabilities["technical"] != 0.12 || a.Confidence != 0.81 {
		t.Errorf("choice = %+v", a)
	}
	if a := resp.Answers["mood"]; a.Score != 1.05 || a.Legend["2"] != "Very angry" || a.Probabilities["1"] != 0.95 {
		t.Errorf("score = %+v", a)
	}
}

func TestEvaluateMissingAnswerIsAnError(t *testing.T) {
	c, _, _ := serve(t, 200, `{"model":"jev-1.13.0","answers":{},"usage":{"input_tokens":300,"output_tokens":0}}`)
	resp, err := c.Evaluate(context.Background(), Request{
		State:     "x",
		Questions: map[string]Question{"urgent": Noul("Urgent?", "", "")},
	})
	if err == nil || !strings.Contains(err.Error(), `"urgent"`) {
		t.Fatalf("err = %v, want missing-answer error", err)
	}
	if resp.InputTokens != 300 {
		t.Errorf("billed tokens dropped: %+v", resp)
	}
}

func TestEvaluateAPIError(t *testing.T) {
	cases := []struct {
		status    int
		body      string
		msg       string
		retryable bool
	}{
		{401, `{"error":{"message":"invalid api key"}}`, "invalid api key", false},
		{422, `{"detail":"criteria must have 2-10 levels"}`, "criteria must have 2-10 levels", false},
		{429, `{"message":"slow down"}`, "slow down", true},
		{529, ``, "", true},
	}
	for _, tc := range cases {
		c, _, _ := serve(t, tc.status, tc.body)
		_, err := c.Evaluate(context.Background(), Request{State: "x", Questions: map[string]Question{"q": Noul("?", "", "")}})
		var ae *APIError
		if !errors.As(err, &ae) {
			t.Fatalf("%d: err = %v, want *APIError", tc.status, err)
		}
		if ae.StatusCode != tc.status || ae.Retryable() != tc.retryable {
			t.Errorf("%d: got %+v retryable=%v", tc.status, ae, ae.Retryable())
		}
		if tc.msg != "" && ae.Message != tc.msg {
			t.Errorf("%d: message = %q, want %q", tc.status, ae.Message, tc.msg)
		}
	}
}

func TestEvaluateValidatesBeforeSending(t *testing.T) {
	c := New(Config{APIKey: "k", BaseURL: "http://127.0.0.1:1"}) // unreachable: must not be dialed
	cases := map[string]Request{
		"no state":     {Questions: map[string]Question{"q": Noul("?", "", "")}},
		"no questions": {State: "x"},
		"one option":   {State: "x", Questions: map[string]Question{"q": Choice("?", map[string]string{"a": ""})}},
		"one level":    {State: "x", Questions: map[string]Question{"q": Score("?", "only")}},
		"bad type":     {State: "x", Questions: map[string]Question{"q": {Type: "rank", Instructions: "?"}}},
		"no prompt":    {State: "x", Questions: map[string]Question{"q": Noul("", "", "")}},
	}
	for name, req := range cases {
		_, err := c.Evaluate(context.Background(), req)
		if err == nil || !strings.HasPrefix(err.Error(), "decide: ") || strings.Contains(err.Error(), "connect") {
			t.Errorf("%s: err = %v, want a validation error", name, err)
		}
	}
}

func TestNewWithoutKeyIsNoop(t *testing.T) {
	c := New(Config{})
	if c.Enabled() {
		t.Error("noop client reports enabled")
	}
	if _, err := c.Evaluate(context.Background(), Request{}); !errors.Is(err, ErrNotConfigured) {
		t.Errorf("err = %v, want ErrNotConfigured", err)
	}
}
