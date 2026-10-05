package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// docSchema is shaped like the schemas dossio sends: an enum, descriptions, and
// additionalProperties:false.
func docSchema() map[string]any {
	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties": map[string]any{
			"docType":   map[string]any{"type": "string", "enum": []any{"NATIONAL_ID", "TAX_ID", "DEED", "CONTRACT", "OTHER"}},
			"docNumber": map[string]any{"type": "string", "description": "the document's identifying number"},
			"docDate":   map[string]any{"type": "string", "description": "YYYY-MM-DD"},
		},
		"required": []any{"docType", "docNumber", "docDate"},
	}
}

// geminiServer is a fake generateContent endpoint. Each request body is recorded
// as a generic map; the first len(fails) requests get those statuses, the rest get
// reply with 200.
type geminiServer struct {
	bodies []map[string]any
	fails  []int
	reply  string
}

func (gs *geminiServer) start(t *testing.T) *googleProvider {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("request is not JSON: %v", err)
		}
		gs.bodies = append(gs.bodies, body)
		if n := len(gs.bodies); n <= len(gs.fails) {
			w.WriteHeader(gs.fails[n-1])
			_, _ = w.Write([]byte(`{"error":{"message":"Invalid JSON payload: unknown field thinkingLevel"}}`))
			return
		}
		_, _ = w.Write([]byte(gs.reply))
	}))
	t.Cleanup(srv.Close)
	return &googleProvider{apiKey: "k", baseURL: srv.URL, http: srv.Client()}
}

const geminiJSONReply = `{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"{\"docType\":\"NATIONAL_ID\",\"docNumber\":\"3171\",\"docDate\":\"2026-01-02\"}"}]}}],"usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":20}}`

func genConfig(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	gc, ok := body["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("no generationConfig in %v", body)
	}
	return gc
}

// textFormat returns generationConfig.responseFormat.text, or nil when absent.
func textFormat(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	rf, _ := genConfig(t, body)["responseFormat"].(map[string]any)
	tf, _ := rf["text"].(map[string]any)
	return tf
}

func userMsg(s string) []Message { return []Message{{Role: "user", Content: s}} }

func TestGeminiSendsSchemaOnlyWhenSet(t *testing.T) {
	gs := &geminiServer{reply: geminiJSONReply}
	g := gs.start(t)
	if _, err := g.complete(context.Background(), "gemini-3.1-pro-preview", 100, Request{Messages: userMsg("read"), JSONSchema: docSchema()}); err != nil {
		t.Fatal(err)
	}
	if _, err := g.complete(context.Background(), "gemini-3.1-pro-preview", 100, Request{Messages: userMsg("read")}); err != nil {
		t.Fatal(err)
	}
	with := textFormat(t, gs.bodies[0])
	if with["mimeType"] != "APPLICATION_JSON" {
		t.Errorf("responseFormat.text.mimeType = %v", with["mimeType"])
	}
	schema, _ := with["schema"].(map[string]any)
	if schema["additionalProperties"] != false || schema["properties"].(map[string]any)["docType"].(map[string]any)["enum"] == nil {
		t.Errorf("schema lost enum or additionalProperties: %v", schema)
	}
	// Neither the new field nor the deprecated ones go out without a schema.
	for _, k := range []string{"responseFormat", "responseMimeType", "responseJsonSchema", "responseSchema"} {
		if _, ok := genConfig(t, gs.bodies[1])[k]; ok {
			t.Errorf("%s sent without a schema", k)
		}
	}
	for _, k := range []string{"responseMimeType", "responseJsonSchema", "responseSchema"} {
		if _, ok := genConfig(t, gs.bodies[0])[k]; ok {
			t.Errorf("deprecated %s sent alongside responseFormat", k)
		}
	}
}

func TestGeminiSchemaSurvivesThinkingRetries(t *testing.T) {
	// Flash models walk the thinking-config candidates on 400; every attempt must
	// still carry the schema.
	gs := &geminiServer{fails: []int{400, 400}, reply: geminiJSONReply}
	g := gs.start(t)
	if _, err := g.complete(context.Background(), "gemini-3.8-flash", 100, Request{Messages: userMsg("read"), JSONSchema: docSchema()}); err != nil {
		t.Fatal(err)
	}
	if len(gs.bodies) != 3 {
		t.Fatalf("attempts = %d, want 3", len(gs.bodies))
	}
	for i, b := range gs.bodies {
		if tf := textFormat(t, b); tf["mimeType"] != "APPLICATION_JSON" || tf["schema"] == nil {
			t.Errorf("attempt %d dropped the schema: %v", i, genConfig(t, b))
		}
	}
}

func TestGeminiJSONSchemaStripsUnsupportedKeywords(t *testing.T) {
	in := map[string]any{
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"type":    "object",
		"properties": map[string]any{
			// A PROPERTY named like a keyword must survive; its own keywords are filtered.
			"pattern": map[string]any{"type": "string", "pattern": "^[0-9]+$", "minLength": 16},
			"kind":    map[string]any{"const": "KTP"},
			"items": map[string]any{"type": "array", "items": map[string]any{
				"type": "object", "additionalProperties": false, "examples": []any{},
				"properties": map[string]any{"n": map[string]any{"type": "integer", "default": 1}},
			}},
			"either": map[string]any{"anyOf": []map[string]any{{"type": "string", "maxLength": 3}, {"type": "null"}}},
		},
		"additionalProperties": false,
	}
	before, _ := json.Marshal(in)
	got, _ := json.Marshal(geminiJSONSchema(in))
	want := `{"additionalProperties":false,"properties":{"either":{"anyOf":[{"type":"string"},{"type":"null"}]},"items":{"items":{"additionalProperties":false,"properties":{"n":{"type":"integer"}},"type":"object"},"type":"array"},"kind":{"enum":["KTP"]},"pattern":{"type":"string"}},"type":"object"}`
	if string(got) != want {
		t.Errorf("geminiJSONSchema\n got %s\nwant %s", got, want)
	}
	if after, _ := json.Marshal(in); string(after) != string(before) {
		t.Error("caller's schema was mutated")
	}
}

func TestGeminiSchemaWithToolsByGeneration(t *testing.T) {
	tools := []ToolDef{{Name: "lookup", Schema: map[string]any{"type": "object"}}}

	// Pre-3: schema + WebSearch is a clear error, sent nowhere.
	gs := &geminiServer{reply: geminiJSONReply}
	g := gs.start(t)
	_, err := g.complete(context.Background(), "gemini-2.5-pro", 100, Request{Messages: userMsg("x"), JSONSchema: docSchema(), WebSearch: true})
	if err == nil || !strings.Contains(err.Error(), "WebSearch") || len(gs.bodies) != 0 {
		t.Errorf("pre-3 schema+WebSearch: err=%v, requests=%d", err, len(gs.bodies))
	}

	// Pre-3: schema + function tools sends the tools and leaves JSON mode off.
	if _, err := g.complete(context.Background(), "gemini-2.5-pro", 100, Request{Messages: userMsg("x"), JSONSchema: docSchema(), Tools: tools}); err != nil {
		t.Fatal(err)
	}
	if textFormat(t, gs.bodies[0]) != nil || gs.bodies[0]["tools"] == nil {
		t.Errorf("pre-3 schema+tools body = %v", gs.bodies[0])
	}

	// Gemini 3: both together, and with grounding.
	for _, req := range []Request{
		{Messages: userMsg("x"), JSONSchema: docSchema(), Tools: tools},
		{Messages: userMsg("x"), JSONSchema: docSchema(), WebSearch: true},
	} {
		gs.bodies = nil
		if _, err := g.complete(context.Background(), "gemini-3.1-pro-preview", 100, req); err != nil {
			t.Fatal(err)
		}
		if textFormat(t, gs.bodies[0])["schema"] == nil || gs.bodies[0]["tools"] == nil {
			t.Errorf("gemini 3 body = %v", gs.bodies[0])
		}
	}
}

func TestSchemaTruncatedReplyIsError(t *testing.T) {
	gs := &geminiServer{reply: `{"candidates":[{"finishReason":"MAX_TOKENS","content":{"parts":[{"text":"{\"docType\":\"NAT"}]}}],"usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":100}}`}
	r := &router{
		providers: map[Provider]provider{ProviderGoogle: gs.start(t)},
		profiles:  map[Profile]ModelRef{ProfileFast: {Provider: ProviderGoogle, Model: "gemini-3.1-pro-preview"}},
		maxTokens: 100,
	}
	resp, err := r.Complete(context.Background(), Request{Task: TaskExtract, Messages: userMsg("x"), JSONSchema: docSchema()})
	if !errors.Is(err, ErrSchemaViolation) {
		t.Fatalf("err = %v, want ErrSchemaViolation", err)
	}
	if resp.OutputTokens != 100 || resp.Model != "gemini-3.1-pro-preview" {
		t.Errorf("violation dropped metering: %+v", resp)
	}
	// Without a schema the same truncation is a normal StopTruncated reply.
	if resp, err := r.Complete(context.Background(), Request{Task: TaskExtract, Messages: userMsg("x")}); err != nil || resp.StopReason != StopTruncated {
		t.Errorf("no schema: resp=%+v err=%v", resp, err)
	}
}

func TestSchemaFailoverSkipsNonEnforcingProviders(t *testing.T) {
	jsonReply := &Response{Text: `{"ok":true}`}
	build := func() (*router, *fakeProvider, *fakeProvider) {
		ds := &fakeProvider{reply: jsonReply}
		g := &fakeProvider{reply: jsonReply}
		return &router{
			providers: map[Provider]provider{
				ProviderAnthropic: &fakeProvider{fail: map[string]error{"opus": errors.New("anthropic: overloaded (status 529)")}},
				ProviderDeepSeek:  ds,
				ProviderGoogle:    g,
			},
			profiles: map[Profile]ModelRef{
				ProfileDeep: {Provider: ProviderAnthropic, Model: "opus"},
				ProfileChat: {Provider: ProviderDeepSeek, Model: "deepseek-chat"},
				ProfileFast: {Provider: ProviderGoogle, Model: "gemini-3.8-flash"},
			},
			fallbacks: map[Profile]ModelRef{ProfileDeep: {Provider: ProviderDeepSeek, Model: "deepseek-chat"}},
			maxTokens: 100,
		}, ds, g
	}

	r, ds, g := build()
	resp, err := r.Complete(context.Background(), Request{Task: TaskReason, Messages: userMsg("x"), JSONSchema: docSchema()})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Provider != ProviderGoogle || len(ds.calls) != 0 || len(g.calls) != 1 {
		t.Errorf("schema failover went to %s (deepseek calls %v, google calls %v)", resp.Provider, ds.calls, g.calls)
	}

	// Control: without a schema the configured DeepSeek fallback is used as before.
	r, ds, _ = build()
	if resp, err := r.Complete(context.Background(), Request{Task: TaskReason, Messages: userMsg("x")}); err != nil || resp.Provider != ProviderDeepSeek || len(ds.calls) != 1 {
		t.Errorf("plain failover: resp=%+v err=%v", resp, err)
	}
}

// openaiServer is a fake chat-completions endpoint recording request bodies.
func openaiServer(t *testing.T, reply string, bodies *[]map[string]any) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		*bodies = append(*bodies, body)
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

const openaiJSONReply = `{"choices":[{"finish_reason":"stop","message":{"content":"{\"docType\":\"NATIONAL_ID\",\"docNumber\":\"3171\",\"docDate\":null}"}}],"usage":{"prompt_tokens":40,"completion_tokens":12}}`

func TestOpenAISendsStrictResponseFormat(t *testing.T) {
	var bodies []map[string]any
	url := openaiServer(t, openaiJSONReply, &bodies)
	strict := &openaiProvider{apiKey: "k", baseURL: url, http: http.DefaultClient, strictSchema: true}
	compat := &openaiProvider{apiKey: "k", baseURL: url, http: http.DefaultClient} // DeepSeek / Moonshot / custom base

	for _, call := range []struct {
		p   *openaiProvider
		req Request
	}{
		{strict, Request{Messages: userMsg("x"), JSONSchema: docSchema()}},
		{strict, Request{Messages: userMsg("x")}},
		{compat, Request{Messages: userMsg("x"), JSONSchema: docSchema()}},
		{strict, Request{Messages: userMsg("x"), JSONSchema: map[string]any{"type": "object", "allOf": []any{}}}},
	} {
		if _, err := call.p.complete(context.Background(), "gpt-5.6-luna", 100, call.req); err != nil {
			t.Fatal(err)
		}
	}

	rf, _ := bodies[0]["response_format"].(map[string]any)
	js, _ := rf["json_schema"].(map[string]any)
	if rf["type"] != "json_schema" || js["strict"] != true || js["name"] != "response" || js["schema"] == nil {
		t.Errorf("response_format = %v", rf)
	}
	for i, why := range map[int]string{1: "without a schema", 2: "on a compatible backend", 3: "for a schema strict mode rejects"} {
		if _, ok := bodies[i]["response_format"]; ok {
			t.Errorf("response_format sent %s", why)
		}
	}
}

func TestNewOpenAIStrictOnlyOnOpenAIEndpoint(t *testing.T) {
	for base, want := range map[string]bool{
		"":                           true,
		"https://api.openai.com/v1":  true,
		"https://api.openai.com/v1/": true,
		"https://api.deepseek.com":   false,
		"https://api.moonshot.ai/v1": false,
		"http://localhost:8000/v1":   false,
	} {
		if got := newOpenAI("k", base).(*openaiProvider).strictSchema; got != want {
			t.Errorf("newOpenAI(%q).strictSchema = %v, want %v", base, got, want)
		}
	}
}

func TestOAIStrictSchema(t *testing.T) {
	in := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"docType": map[string]any{"type": "string", "enum": []string{"KTP", "NPWP"}},
			"docDate": map[string]any{"type": "string"},
			"party":   map[string]any{"$ref": "#/$defs/party"},
			"tags":    map[string]any{"type": "array", "items": map[string]any{"type": "object", "properties": map[string]any{"v": map[string]any{"type": "string"}}}},
		},
		"required": []string{"tags"},
		"$defs":    map[string]any{"party": map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}, "required": []any{"name"}}},
	}
	before, _ := json.Marshal(in)
	out, ok := oaiStrictSchema(in)
	if !ok {
		t.Fatal("dossio-shaped schema rejected")
	}
	got, _ := json.Marshal(out)
	want := `{"$defs":{"party":{"additionalProperties":false,"properties":{"name":{"type":"string"}},"required":["name"],"type":"object"}},"additionalProperties":false,"properties":{"docDate":{"type":["string","null"]},"docType":{"enum":["KTP","NPWP",null],"type":["string","null"]},"party":{"anyOf":[{"$ref":"#/$defs/party"},{"type":"null"}]},"tags":{"items":{"additionalProperties":false,"properties":{"v":{"type":["string","null"]}},"required":["v"],"type":"object"},"type":"array"}},"required":["tags","docDate","docType","party"],"type":"object"}`
	if string(got) != want {
		t.Errorf("oaiStrictSchema\n got %s\nwant %s", got, want)
	}
	if after, _ := json.Marshal(in); string(after) != string(before) {
		t.Error("caller's schema was mutated")
	}

	for name, bad := range map[string]map[string]any{
		"root array":    {"type": "array", "items": map[string]any{"type": "string"}},
		"root anyOf":    {"type": "object", "anyOf": []any{map[string]any{"type": "object"}}},
		"allOf":         {"type": "object", "properties": map[string]any{"a": map[string]any{"allOf": []any{}}}},
		"open object":   {"type": "object", "additionalProperties": true},
		"map of values": {"type": "object", "additionalProperties": map[string]any{"type": "string"}},
	} {
		if _, ok := oaiStrictSchema(bad); ok {
			t.Errorf("%s: accepted, want rejected", name)
		}
	}
}

func TestSchemaFailoverToStrictOpenAI(t *testing.T) {
	// OpenAI is now an eligible schema fallback when it enforces.
	var bodies []map[string]any
	oai := &openaiProvider{apiKey: "k", baseURL: openaiServer(t, openaiJSONReply, &bodies), http: http.DefaultClient, strictSchema: true}
	r := &router{
		providers: map[Provider]provider{
			ProviderAnthropic: &fakeProvider{fail: map[string]error{"opus": errors.New("anthropic: overloaded (status 529)")}},
			ProviderOpenAI:    oai,
		},
		profiles:  map[Profile]ModelRef{ProfileDeep: {Provider: ProviderAnthropic, Model: "opus"}},
		fallbacks: map[Profile]ModelRef{ProfileDeep: {Provider: ProviderOpenAI, Model: "gpt-5.6-luna"}},
		maxTokens: 100,
	}
	resp, err := r.Complete(context.Background(), Request{Task: TaskReason, Messages: userMsg("x"), JSONSchema: docSchema()})
	if err != nil || resp.Provider != ProviderOpenAI || len(bodies) != 1 {
		t.Fatalf("resp=%+v err=%v requests=%d", resp, err, len(bodies))
	}
	// A schema strict mode can't express makes the same OpenAI ineligible.
	bodies = nil
	_, err = r.Complete(context.Background(), Request{Task: TaskReason, Messages: userMsg("x"), JSONSchema: map[string]any{"type": "object", "additionalProperties": true}})
	if err == nil || len(bodies) != 0 {
		t.Errorf("unenforceable schema failed over to OpenAI: err=%v requests=%d", err, len(bodies))
	}
}

func TestSchemaSafetyNetOnNonEnforcingPrimary(t *testing.T) {
	cases := []struct {
		name, text, want string
		toolCall, isErr  bool
	}{
		{"bare JSON", ` {"a":1} `, `{"a":1}`, false, false},
		{"json fence", "```json\n{\"a\":1}\n```", `{"a":1}`, false, false},
		{"bare fence", "```\n[1,2]\n```", `[1,2]`, false, false},
		{"one-line fence", "```{\"a\":1}```", `{"a":1}`, false, false},
		{"prose around JSON", "Here you go:\n{\"a\":1}", "", false, true},
		{"prose", "I could not find a date.", "", false, true},
		{"tool round is exempt", "", "", true, false},
	}
	for _, c := range cases {
		reply := &Response{Text: c.text, OutputTokens: 7}
		if c.toolCall {
			reply.ToolCalls = []ToolCall{{ID: "1", Name: "lookup", Input: json.RawMessage(`{}`)}}
		}
		r := &router{
			providers: map[Provider]provider{ProviderDeepSeek: &fakeProvider{reply: reply}},
			profiles:  map[Profile]ModelRef{ProfileFast: {Provider: ProviderDeepSeek, Model: "deepseek-chat"}},
			maxTokens: 100,
		}
		resp, err := r.Complete(context.Background(), Request{Task: TaskExtract, Messages: userMsg("x"), JSONSchema: docSchema()})
		if c.isErr {
			if !errors.Is(err, ErrSchemaViolation) || resp.OutputTokens != 7 {
				t.Errorf("%s: err=%v resp=%+v, want ErrSchemaViolation with tokens", c.name, err, resp)
			}
			continue
		}
		if err != nil || resp.Text != c.want {
			t.Errorf("%s: text=%q err=%v, want %q", c.name, resp.Text, err, c.want)
		}
	}
}

// TestGeminiSchemaLive is the opt-in smoke test against the real API: a
// dossio-shaped schema (enum included) over a PDF part. Skipped without a key.
// GEMINI_SMOKE_MODEL picks the model (default gemini-flash-latest).
func TestGeminiSchemaLive(t *testing.T) {
	key := os.Getenv("GOOGLE_API_KEY")
	if key == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}
	model := os.Getenv("GEMINI_SMOKE_MODEL")
	if model == "" {
		model = "gemini-flash-latest"
	}
	c := New(Config{GoogleAPIKey: key, Profiles: map[Profile]ModelRef{ProfileDocument: {Provider: ProviderGoogle, Model: model}}})
	resp, err := c.Complete(context.Background(), Request{
		Task:       TaskDocument,
		MaxTokens:  2048,
		NoFallback: true,
		JSONSchema: docSchema(),
		Messages: []Message{{
			Role:      "user",
			Content:   "Extract the document type, number and date from the attached PDF.",
			Documents: []Document{{MediaType: "application/pdf", Base64: base64.StdEncoding.EncodeToString(tinyPDF("KARTU TANDA PENDUDUK  NIK 3171012345678901  Tanggal 2026-01-02"))}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var got struct{ DocType, DocNumber, DocDate string }
	if err := json.Unmarshal([]byte(resp.Text), &got); err != nil {
		t.Fatalf("reply is not JSON: %v\n%s", err, resp.Text)
	}
	if got.DocType == "" || got.DocNumber == "" {
		t.Errorf("empty fields: %+v", got)
	}
	t.Logf("%s: %+v (%d in / %d out tokens)", resp.Model, got, resp.InputTokens, resp.OutputTokens)
}

// tinyPDF builds a one-page PDF showing line in Helvetica, with a correct xref.
func tinyPDF(line string) []byte {
	stream := fmt.Sprintf("BT /F1 12 Tf 40 760 Td (%s) Tj ET", line)
	objs := []string{
		"<< /Type /Catalog /Pages 2 0 R >>",
		"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
		fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(stream), stream),
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
	}
	var b strings.Builder
	b.WriteString("%PDF-1.4\n")
	offsets := make([]int, len(objs))
	for i, o := range objs {
		offsets[i] = b.Len()
		fmt.Fprintf(&b, "%d 0 obj\n%s\nendobj\n", i+1, o)
	}
	xref := b.Len()
	fmt.Fprintf(&b, "xref\n0 %d\n0000000000 65535 f \n", len(objs)+1)
	for _, off := range offsets {
		fmt.Fprintf(&b, "%010d 00000 n \n", off)
	}
	fmt.Fprintf(&b, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objs)+1, xref)
	return []byte(b.String())
}
