package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"strings"
	"testing"
	"time"
)

// fakeBatch is an in-package batchBackend. It records every call.
type fakeBatch struct {
	supported map[batchKind]bool // nil = all
	lim       batchLimits

	submits    []fakeSubmit
	buildCalls []int // maxTokens seen by buildChat
	submitErr  error
	statuses   []BatchStatus // scripted; the last repeats
	statusErr  error
	statusN    int
	raws       []rawBatchResult
	iterErr    error // yielded after raws
	pulled     int
	closed     bool
	cancelN    int
	ctxs       []context.Context
}

type fakeSubmit struct {
	kind  batchKind
	model string
	lines []batchLine
}

func (f *fakeBatch) supports(k batchKind) bool    { return f.supported == nil || f.supported[k] }
func (f *fakeBatch) limits(batchKind) batchLimits { return f.lim }
func (f *fakeBatch) buildChat(model string, maxTokens int, req Request) ([]byte, error) {
	f.buildCalls = append(f.buildCalls, maxTokens)
	return []byte(`{"p":"` + req.Messages[0].Content + `"}`), nil
}
func (f *fakeBatch) buildEmbed(model, text string, kind EmbedKind, dims int) ([]byte, error) {
	return []byte(`{"t":"` + text + `"}`), nil
}
func (f *fakeBatch) submit(ctx context.Context, kind batchKind, model string, lines []batchLine) (string, error) {
	f.submits = append(f.submits, fakeSubmit{kind, model, lines})
	if f.submitErr != nil {
		return "", f.submitErr
	}
	return "batches/abc:1", nil
}
func (f *fakeBatch) status(ctx context.Context, id string) (BatchStatus, error) {
	f.ctxs = append(f.ctxs, ctx)
	n := f.statusN
	f.statusN++
	if f.statusErr != nil {
		return BatchStatus{}, f.statusErr
	}
	if len(f.statuses) == 0 {
		return BatchStatus{State: BatchDone}, nil
	}
	return f.statuses[min(n, len(f.statuses)-1)], nil
}
func (f *fakeBatch) results(ctx context.Context, kind batchKind, model, id string) iter.Seq2[rawBatchResult, error] {
	return func(yield func(rawBatchResult, error) bool) {
		defer func() { f.closed = true }()
		for _, r := range f.raws {
			f.pulled++
			if !yield(r, nil) {
				return
			}
		}
		if f.iterErr != nil {
			yield(rawBatchResult{}, f.iterErr)
		}
	}
}
func (f *fakeBatch) cancel(ctx context.Context, id string) error {
	f.ctxs = append(f.ctxs, ctx)
	f.cancelN++
	return nil
}

func userReq(text string) Request {
	return Request{Messages: []Message{{Role: "user", Content: text}}}
}

func chatItems(n int) []ChatItem {
	var out []ChatItem
	for i := 0; i < n; i++ {
		out = append(out, ChatItem{CustomID: fmt.Sprintf("id%d", i), Request: userReq("hi")})
	}
	return out
}

func testCfg() Config {
	return Config{Profiles: map[Profile]ModelRef{
		ProfileChat: {ProviderAnthropic, "chat-m"},
		ProfileFast: {ProviderAnthropic, "fast-m"},
	}, MaxTokens: 0}
}

func newFakeBatcher(cfg Config, p Provider, f *fakeBatch) Batcher {
	return newBatcher(cfg, map[Provider]batchBackend{p: f})
}

func TestBatchIDRoundTrip(t *testing.T) {
	id := makeBatchID(ProviderOpenAI, batchChatJSON, "ft:gpt-4o:org:x/y", "batches/abc:1")
	if id.Provider() != ProviderOpenAI || id.Model() != "ft:gpt-4o:org:x/y" {
		t.Fatalf("got %q %q", id.Provider(), id.Model())
	}
	p, k, m, pid, err := parseBatchID(id)
	if err != nil || p != ProviderOpenAI || k != batchChatJSON || m != "ft:gpt-4o:org:x/y" || pid != "batches/abc:1" {
		t.Fatalf("parse = %v %v %v %v %v", p, k, m, pid, err)
	}
}

func TestBatchBadIDs(t *testing.T) {
	f := &fakeBatch{}
	b := newFakeBatcher(testCfg(), ProviderAnthropic, f)
	for _, s := range []string{"", "a:b", "anthropic:weird:m:x", "anthropic:chat::x", "anthropic:chat:m:"} {
		id := BatchID(s)
		if id.Provider() != "" || id.Model() != "" {
			t.Errorf("%q: Provider/Model not empty", s)
		}
		if _, err := b.Status(context.Background(), id); !errors.Is(err, ErrBadBatchID) {
			t.Errorf("%q status: %v", s, err)
		}
		if err := b.Cancel(context.Background(), id); !errors.Is(err, ErrBadBatchID) {
			t.Errorf("%q cancel: %v", s, err)
		}
		_, err := CollectResults(b.Results(context.Background(), id))
		if !errors.Is(err, ErrBadBatchID) {
			t.Errorf("%q results: %v", s, err)
		}
	}
}

func TestBatchNotConfigured(t *testing.T) {
	b := NewBatcher(Config{})
	ctx := context.Background()
	if b.Enabled() {
		t.Fatal("enabled")
	}
	if _, err := b.SubmitChat(ctx, ChatBatch{Items: chatItems(1)}); !errors.Is(err, ErrNotConfigured) {
		t.Error(err)
	}
	if _, err := b.SubmitEmbed(ctx, EmbedBatch{}); !errors.Is(err, ErrNotConfigured) {
		t.Error(err)
	}
	if _, err := b.Status(ctx, "x"); !errors.Is(err, ErrNotConfigured) {
		t.Error(err)
	}
	if err := b.Cancel(ctx, "x"); !errors.Is(err, ErrNotConfigured) {
		t.Error(err)
	}
	n := 0
	for _, err := range b.Results(ctx, "x") {
		n++
		if !errors.Is(err, ErrNotConfigured) {
			t.Error(err)
		}
	}
	if n != 1 {
		t.Fatalf("results yielded %d", n)
	}
}

func TestBatchModelPick(t *testing.T) {
	ctx := context.Background()
	submitModel := func(cfg Config, cb ChatBatch) (string, error) {
		f := &fakeBatch{}
		b := newFakeBatcher(cfg, ProviderAnthropic, f)
		cb.Items = chatItems(1)
		_, err := b.SubmitChat(ctx, cb)
		if err != nil {
			return "", err
		}
		return f.submits[0].model, nil
	}
	if m, _ := submitModel(testCfg(), ChatBatch{Model: &ModelRef{ProviderAnthropic, "pin"}, Task: TaskClassify}); m != "pin" {
		t.Errorf("model wins: %q", m)
	}
	if m, _ := submitModel(testCfg(), ChatBatch{Task: TaskClassify}); m != "fast-m" {
		t.Errorf("classify: %q", m)
	}
	if m, _ := submitModel(testCfg(), ChatBatch{}); m != "fast-m" {
		t.Errorf("no Task is the fast profile: %q", m)
	}
	if m, _ := submitModel(testCfg(), ChatBatch{Task: Task("unknown")}); m != "chat-m" {
		t.Errorf("unknown Task is the chat profile: %q", m)
	}
	cfg := Config{Profiles: map[Profile]ModelRef{ProfileDeep: {ProviderAnthropic, "deep-m"}, ProfileVision: {ProviderAnthropic, "vis-m"}}}
	if m, _ := submitModel(cfg, ChatBatch{}); m != "deep-m" {
		t.Errorf("first non-empty: %q", m)
	}
	if _, err := submitModel(Config{}, ChatBatch{}); !errors.Is(err, ErrNotConfigured) {
		t.Errorf("all empty: %v", err)
	}
}

func TestBatchUnsupported(t *testing.T) {
	ctx := context.Background()
	f := &fakeBatch{}
	mk := func(cfg Config, ps ...Provider) Batcher {
		m := map[Provider]batchBackend{}
		for _, p := range ps {
			m[p] = f
		}
		return newBatcher(cfg, m)
	}
	cases := []struct {
		name string
		b    Batcher
		ref  ModelRef
	}{
		{"deepseek", mk(Config{}, ProviderDeepSeek), ModelRef{ProviderDeepSeek, "m"}},
		{"openai base url", mk(Config{OpenAIBaseURL: "http://x"}, ProviderOpenAI), ModelRef{ProviderOpenAI, "m"}},
		{"unknown", mk(Config{}, "weird"), ModelRef{"weird", "m"}},
	}
	for _, c := range cases {
		ref := c.ref
		_, err := c.b.SubmitChat(ctx, ChatBatch{Model: &ref, Items: chatItems(1)})
		if !errors.Is(err, ErrBatchUnsupported) {
			t.Errorf("%s chat: %v", c.name, err)
		}
	}
	for _, p := range []Provider{ProviderAnthropic, ProviderMoonshot, ProviderDeepSeek} {
		b := mk(Config{}, p)
		_, err := b.SubmitEmbed(ctx, EmbedBatch{Model: &ModelRef{p, "m"}, Items: []EmbedItem{{"a", "x"}}})
		if !errors.Is(err, ErrBatchUnsupported) {
			t.Errorf("embed %s: %v", p, err)
		}
	}
	b := mk(Config{OpenAIBaseURL: "http://x"}, ProviderOpenAI)
	if _, err := b.SubmitEmbed(ctx, EmbedBatch{Model: &ModelRef{ProviderOpenAI, "m"}, Items: []EmbedItem{{"a", "x"}}}); !errors.Is(err, ErrBatchUnsupported) {
		t.Errorf("embed openai base: %v", err)
	}
	if len(f.submits) != 0 {
		t.Fatal("sent")
	}
}

func TestBatchNoBackend(t *testing.T) {
	f := &fakeBatch{}
	b := newFakeBatcher(testCfg(), ProviderGoogle, f) // profile is Anthropic
	_, err := b.SubmitChat(context.Background(), ChatBatch{Items: chatItems(1)})
	if !errors.Is(err, ErrNotConfigured) {
		t.Fatal(err)
	}
}

func TestBatchItemChecks(t *testing.T) {
	schema := map[string]any{"type": "object"}
	withSchema := func(id string) ChatItem {
		r := userReq("x")
		r.JSONSchema = schema
		return ChatItem{CustomID: id, Request: r}
	}
	plain := func(id string) ChatItem { return ChatItem{CustomID: id, Request: userReq("x")} }
	other := ModelRef{ProviderAnthropic, "other"}
	cases := []struct {
		name  string
		items []ChatItem
		idx   string // expected "item N" in message; "" = no index
	}{
		{"empty", nil, ""},
		{"empty id", []ChatItem{plain("")}, "item 0"},
		{"65 chars", []ChatItem{plain("a"), plain(strings.Repeat("a", 65))}, "item 1"},
		{"space", []ChatItem{plain("a b")}, "item 0"},
		{"non-ascii", []ChatItem{plain("é")}, "item 0"},
		{"dup", []ChatItem{plain("a"), plain("b"), plain("a")}, "item 2"},
		{"model differs", []ChatItem{{CustomID: "a", Request: Request{Model: &other, Messages: userReq("x").Messages}}}, "item 0"},
		{"task differs", []ChatItem{{CustomID: "a", Request: Request{Task: TaskReason, Messages: userReq("x").Messages}}}, "item 0"},
		{"onevent", []ChatItem{{CustomID: "a", Request: Request{OnEvent: func(StreamEvent) {}, Messages: userReq("x").Messages}}}, "item 0"},
		{"mix 1", []ChatItem{withSchema("a"), plain("b")}, "item 1"},
		{"mix 2", []ChatItem{plain("a"), withSchema("b")}, "item 1"},
	}
	for _, c := range cases {
		f := &fakeBatch{}
		b := newFakeBatcher(testCfg(), ProviderAnthropic, f)
		_, err := b.SubmitChat(context.Background(), ChatBatch{Items: c.items})
		if err == nil {
			t.Errorf("%s: no error", c.name)
			continue
		}
		if c.idx != "" && !strings.Contains(err.Error(), c.idx) {
			t.Errorf("%s: %q lacks %q", c.name, err, c.idx)
		}
		if len(f.submits) != 0 {
			t.Errorf("%s: sent", c.name)
		}
	}
	// the same model and task as the batch are fine
	f := &fakeBatch{}
	b := newFakeBatcher(testCfg(), ProviderAnthropic, f)
	same := ModelRef{ProviderAnthropic, "chat-m"}
	_, err := b.SubmitChat(context.Background(), ChatBatch{Task: TaskChat, Items: []ChatItem{{CustomID: "a", Request: Request{Model: &same, Task: TaskChat, Messages: userReq("x").Messages}}}})
	if err != nil {
		t.Fatal(err)
	}
}

func TestBatchKind(t *testing.T) {
	ctx := context.Background()
	f := &fakeBatch{}
	b := newFakeBatcher(testCfg(), ProviderAnthropic, f)
	r := userReq("x")
	r.JSONSchema = map[string]any{"type": "object"}
	id, err := b.SubmitChat(ctx, ChatBatch{Items: []ChatItem{{"a", r}, {"b", r}}})
	if err != nil {
		t.Fatal(err)
	}
	if f.submits[0].kind != batchChatJSON || !strings.HasPrefix(string(id), "anthropic:chatjson:fast-m:") {
		t.Fatalf("%v %q", f.submits[0].kind, id)
	}
	id, err = b.SubmitChat(ctx, ChatBatch{Items: chatItems(1)})
	if err != nil || f.submits[1].kind != batchChat || !strings.HasPrefix(string(id), "anthropic:chat:") {
		t.Fatalf("%v %q %v", f.submits[1].kind, id, err)
	}
}

func TestBatchMaxTokens(t *testing.T) {
	ctx := context.Background()
	items := chatItems(2)
	items[0].Request.MaxTokens = 77
	f := &fakeBatch{}
	b := newFakeBatcher(Config{Profiles: testCfg().Profiles, MaxTokens: 500}, ProviderAnthropic, f)
	if _, err := b.SubmitChat(ctx, ChatBatch{Items: items}); err != nil {
		t.Fatal(err)
	}
	if f.buildCalls[0] != 77 || f.buildCalls[1] != 500 {
		t.Fatalf("%v", f.buildCalls)
	}
	f = &fakeBatch{}
	b = newFakeBatcher(testCfg(), ProviderAnthropic, f)
	if _, err := b.SubmitChat(ctx, ChatBatch{Items: chatItems(1)}); err != nil {
		t.Fatal(err)
	}
	if f.buildCalls[0] != 1024 {
		t.Fatalf("%v", f.buildCalls)
	}
}

func TestBatchLimits(t *testing.T) {
	ctx := context.Background()
	f := &fakeBatch{lim: batchLimits{maxItems: 2}}
	b := newFakeBatcher(testCfg(), ProviderAnthropic, f)
	if _, err := b.SubmitChat(ctx, ChatBatch{Items: chatItems(3)}); !errors.Is(err, ErrBatchTooLarge) {
		t.Fatal(err)
	}
	f = &fakeBatch{lim: batchLimits{maxBytes: 10}}
	b = newFakeBatcher(testCfg(), ProviderAnthropic, f)
	if _, err := b.SubmitChat(ctx, ChatBatch{Items: chatItems(3)}); !errors.Is(err, ErrBatchTooLarge) {
		t.Fatal(err)
	}
	if len(f.submits) != 0 {
		t.Fatal("sent")
	}
	// Bodies are 30 bytes, under 100. The line overhead (3 x 103) puts it over.
	f = &fakeBatch{lim: batchLimits{maxBytes: 100}}
	b = newFakeBatcher(testCfg(), ProviderAnthropic, f)
	if _, err := b.SubmitChat(ctx, ChatBatch{Items: chatItems(3)}); !errors.Is(err, ErrBatchTooLarge) {
		t.Fatalf("overhead: %v", err)
	}
	f = &fakeBatch{lim: batchLimits{maxBytes: 400}}
	b = newFakeBatcher(testCfg(), ProviderAnthropic, f)
	if _, err := b.SubmitChat(ctx, ChatBatch{Items: chatItems(3)}); err != nil {
		t.Fatalf("under the limit: %v", err)
	}
}

func TestBatchFailedWithoutErr(t *testing.T) {
	f := &fakeBatch{statuses: []BatchStatus{{State: BatchFailed}}}
	b := newFakeBatcher(testCfg(), ProviderAnthropic, f)
	id, err := b.SubmitChat(context.Background(), ChatBatch{Items: chatItems(1)})
	if err != nil {
		t.Fatal(err)
	}
	var got error
	for _, err := range b.Results(context.Background(), id) {
		got = err
	}
	if got == nil || strings.Contains(got.Error(), "%!w") || !strings.Contains(got.Error(), "llm: batch failed") {
		t.Fatalf("err = %v", got)
	}
}

func embedCfg(model ModelRef) Config { return Config{EmbedModel: model} }

func TestBatchEmbedChecks(t *testing.T) {
	ctx := context.Background()
	f := &fakeBatch{}
	b := newFakeBatcher(embedCfg(ModelRef{ProviderGoogle, "m"}), ProviderGoogle, f)
	bad := []EmbedBatch{
		{Items: nil},
		{Items: []EmbedItem{{"a", ""}}},
		{Items: []EmbedItem{{"a", "x"}}, Dimensions: -1},
		{Items: []EmbedItem{{"a", "x"}}, Kind: "weird"},
		{Items: []EmbedItem{{"a", "x"}, {"a", "y"}}},
		{Items: []EmbedItem{{"a b", "x"}}},
	}
	for i, eb := range bad {
		if _, err := b.SubmitEmbed(ctx, eb); err == nil {
			t.Errorf("case %d: no error", i)
		}
	}
	if len(f.submits) != 0 {
		t.Fatal("sent")
	}
	// models/ prefix stripped, default model used
	pin := ModelRef{ProviderGoogle, "models/gemini-embedding"}
	id, err := b.SubmitEmbed(ctx, EmbedBatch{Model: &pin, Items: []EmbedItem{{"a", "x"}}})
	if err != nil || f.submits[0].model != "gemini-embedding" || f.submits[0].kind != batchEmbed {
		t.Fatalf("%v %+v", err, f.submits)
	}
	if id.Model() != "gemini-embedding" {
		t.Fatal(id)
	}
	if _, err := b.SubmitEmbed(ctx, EmbedBatch{Items: []EmbedItem{{"a", "x"}}}); err != nil || f.submits[1].model != "m" {
		t.Fatalf("%v %+v", err, f.submits)
	}
	b2 := newFakeBatcher(Config{}, ProviderGoogle, f)
	if _, err := b2.SubmitEmbed(ctx, EmbedBatch{Items: []EmbedItem{{"a", "x"}}}); !errors.Is(err, ErrNotConfigured) {
		t.Fatal(err)
	}
}

func TestBatchResultsNotDone(t *testing.T) {
	ctx := context.Background()
	id := makeBatchID(ProviderAnthropic, batchChat, "m", "p1")
	f := &fakeBatch{statuses: []BatchStatus{{State: BatchRunning}}, raws: []rawBatchResult{{customID: "a"}}}
	b := newFakeBatcher(testCfg(), ProviderAnthropic, f)
	got, err := CollectResults(b.Results(ctx, id))
	if !errors.Is(err, ErrBatchNotDone) || len(got) != 0 || f.pulled != 0 {
		t.Fatalf("%v %v %d", err, got, f.pulled)
	}
	boom := errors.New("boom")
	f = &fakeBatch{statuses: []BatchStatus{{State: BatchFailed, Err: boom}}}
	b = newFakeBatcher(testCfg(), ProviderAnthropic, f)
	if _, err := CollectResults(b.Results(ctx, id)); !errors.Is(err, boom) {
		t.Fatal(err)
	}
	f = &fakeBatch{statusErr: boom}
	b = newFakeBatcher(testCfg(), ProviderAnthropic, f)
	if _, err := CollectResults(b.Results(ctx, id)); !errors.Is(err, boom) {
		t.Fatal(err)
	}
}

func TestBatchResultsReplySteps(t *testing.T) {
	ctx := context.Background()
	collect := func(kind batchKind, raws ...rawBatchResult) []BatchResult {
		f := &fakeBatch{raws: raws}
		b := newFakeBatcher(testCfg(), ProviderAnthropic, f)
		got, err := CollectResults(b.Results(ctx, makeBatchID(ProviderAnthropic, kind, "m", "p1")))
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	r := collect(batchChatJSON, rawBatchResult{customID: "a", resp: Response{Text: "```json\n{\"a\":1}\n```"}})[0]
	if r.Err != nil || r.Response.Text != `{"a":1}` || r.Response.Provider != ProviderAnthropic || r.Response.Model != "m" || r.CustomID != "a" {
		t.Errorf("fence: %+v", r)
	}
	r = collect(batchChatJSON, rawBatchResult{customID: "a", resp: Response{Text: "not json", InputTokens: 5, OutputTokens: 6}})[0]
	if !errors.Is(r.Err, ErrSchemaViolation) || r.Response.InputTokens != 5 || r.Response.OutputTokens != 6 {
		t.Errorf("bad json: %+v", r)
	}
	r = collect(batchChatJSON, rawBatchResult{customID: "a", resp: Response{Text: `{"a":1}`, StopReason: StopTruncated}})[0]
	if !errors.Is(r.Err, ErrSchemaViolation) {
		t.Errorf("truncated: %+v", r)
	}
	r = collect(batchChat, rawBatchResult{customID: "a", resp: Response{Text: "not json"}})[0]
	if r.Err != nil {
		t.Errorf("chat: %+v", r)
	}
	ref := &RefusalError{Provider: ProviderAnthropic, Model: "m"}
	r = collect(batchChatJSON, rawBatchResult{customID: "a", resp: Response{InputTokens: 9}, err: ref})[0]
	if r.Err != ref || r.Response.InputTokens != 9 {
		t.Errorf("refusal: %+v", r)
	}
	r = collect(batchEmbed, rawBatchResult{customID: "a", vector: []float32{3, 4}})[0]
	if r.Err != nil || len(r.Vector) != 2 || r.Vector[0] != 0.6 || r.Vector[1] != 0.8 {
		t.Errorf("embed: %+v", r)
	}
	r = collect(batchEmbed, rawBatchResult{customID: "a", vector: []float32{0, 0}})[0]
	if r.Err != nil || r.Vector[0] != 0 || r.Vector[1] != 0 {
		t.Errorf("zero: %+v", r)
	}
	r = collect(batchEmbed, rawBatchResult{customID: "a"})[0]
	if r.Err == nil {
		t.Errorf("empty vector: %+v", r)
	}
	// the input slice is copied, not changed
	in := []float32{3, 4}
	collect(batchEmbed, rawBatchResult{customID: "a", vector: in})
	if in[0] != 3 {
		t.Error("input vector changed")
	}
}

func TestBatchResultsEarlyBreak(t *testing.T) {
	f := &fakeBatch{raws: []rawBatchResult{{customID: "a"}, {customID: "b"}, {customID: "c"}}}
	b := newFakeBatcher(testCfg(), ProviderAnthropic, f)
	for r, err := range b.Results(context.Background(), makeBatchID(ProviderAnthropic, batchChat, "m", "p1")) {
		if err != nil || r.CustomID != "a" {
			t.Fatal(r, err)
		}
		break
	}
	if !f.closed || f.pulled != 1 {
		t.Fatalf("closed=%v pulled=%d", f.closed, f.pulled)
	}
}

func TestBatchResultsIterError(t *testing.T) {
	boom := errors.New("read failed")
	f := &fakeBatch{raws: []rawBatchResult{{customID: "a"}}, iterErr: boom}
	b := newFakeBatcher(testCfg(), ProviderAnthropic, f)
	var got []BatchResult
	var errs []error
	for r, err := range b.Results(context.Background(), makeBatchID(ProviderAnthropic, batchChat, "m", "p1")) {
		got = append(got, r)
		errs = append(errs, err)
	}
	if len(got) != 2 || errs[0] != nil || !errors.Is(errs[1], boom) || got[1].CustomID != "" {
		t.Fatalf("%+v %v", got, errs)
	}
}

func TestBatchDefaultDeadline(t *testing.T) {
	id := makeBatchID(ProviderAnthropic, batchChat, "m", "p1")
	check := func(cfg Config, want time.Duration) {
		f := &fakeBatch{}
		b := newFakeBatcher(cfg, ProviderAnthropic, f)
		if _, err := b.Status(context.Background(), id); err != nil {
			t.Fatal(err)
		}
		if err := b.Cancel(context.Background(), id); err != nil {
			t.Fatal(err)
		}
		for _, c := range f.ctxs {
			d, ok := c.Deadline()
			if !ok {
				t.Fatal("no deadline")
			}
			if left := time.Until(d); left > want || left < want-5*time.Second {
				t.Fatalf("left %v want ~%v", left, want)
			}
		}
	}
	check(Config{}, 60*time.Second)
	check(Config{RequestTimeout: 7 * time.Second}, 7*time.Second)
	check(Config{RequestTimeout: -1}, 60*time.Second)

	f := &fakeBatch{}
	b := newFakeBatcher(Config{}, ProviderAnthropic, f)
	ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
	defer cancel()
	want, _ := ctx.Deadline()
	if _, err := b.Status(ctx, id); err != nil {
		t.Fatal(err)
	}
	if got, _ := f.ctxs[0].Deadline(); !got.Equal(want) {
		t.Fatal("deadline changed")
	}
}

func TestBatchWait(t *testing.T) {
	ctx := context.Background()
	id := makeBatchID(ProviderAnthropic, batchChat, "m", "p1")
	f := &fakeBatch{statuses: []BatchStatus{{State: BatchRunning}, {State: BatchRunning}, {State: BatchDone, Total: 3}}}
	b := newFakeBatcher(testCfg(), ProviderAnthropic, f)
	st, err := Wait(ctx, b, id, time.Millisecond)
	if err != nil || st.State != BatchDone || f.statusN != 3 {
		t.Fatalf("%+v %v %d", st, err, f.statusN)
	}
	boom := errors.New("x")
	f = &fakeBatch{statuses: []BatchStatus{{State: BatchFailed, Err: boom}}}
	b = newFakeBatcher(testCfg(), ProviderAnthropic, f)
	st, err = Wait(ctx, b, id, time.Millisecond)
	if err != nil || st.State != BatchFailed {
		t.Fatalf("%+v %v", st, err)
	}
	f = &fakeBatch{statusErr: boom}
	b = newFakeBatcher(testCfg(), ProviderAnthropic, f)
	if _, err = Wait(ctx, b, id, time.Millisecond); !errors.Is(err, boom) {
		t.Fatal(err)
	}
	f = &fakeBatch{statuses: []BatchStatus{{State: BatchRunning}}}
	b = newFakeBatcher(testCfg(), ProviderAnthropic, f)
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	st, err = Wait(cctx, b, id, time.Hour)
	if !errors.Is(err, context.Canceled) || st.State != BatchRunning {
		t.Fatalf("%+v %v", st, err)
	}
}

func TestBatchCollectResults(t *testing.T) {
	boom := errors.New("x")
	seq := func(yield func(BatchResult, error) bool) {
		if !yield(BatchResult{CustomID: "a"}, nil) {
			return
		}
		if !yield(BatchResult{CustomID: "b"}, nil) {
			return
		}
		if !yield(BatchResult{}, boom) {
			return
		}
		t.Error("pulled past error")
		yield(BatchResult{CustomID: "c"}, nil)
	}
	got, err := CollectResults(seq)
	if !errors.Is(err, boom) || len(got) != 2 || got[1].CustomID != "b" {
		t.Fatalf("%v %v", got, err)
	}
	got, err = CollectResults(func(yield func(BatchResult, error) bool) { yield(BatchResult{CustomID: "a"}, nil) })
	if err != nil || len(got) != 1 {
		t.Fatal(got, err)
	}
}

func TestJsonlLines(t *testing.T) {
	collect := func(r io.Reader) ([]string, error) {
		var out []string
		for l, err := range jsonlLines(r) {
			if err != nil {
				return out, err
			}
			out = append(out, string(l))
		}
		return out, nil
	}
	got, err := collect(strings.NewReader("a\r\n\n  \nb\r\nc"))
	if err != nil || strings.Join(got, "|") != "a|b|c" {
		t.Fatalf("%q %v", got, err)
	}
	big := strings.Repeat("x", 1<<20)
	got, err = collect(strings.NewReader(big + "\nz\n"))
	if err != nil || len(got) != 2 || got[0] != big {
		t.Fatalf("big: %d %v", len(got), err)
	}
	boom := errors.New("net")
	got, err = collect(io.MultiReader(strings.NewReader("a\nb\n"), errReader{boom}))
	if !errors.Is(err, boom) || len(got) != 2 {
		t.Fatalf("%q %v", got, err)
	}
	// each line is a copy
	var keep [][]byte
	for l, _ := range jsonlLines(strings.NewReader("aa\nbb\n")) {
		keep = append(keep, l)
	}
	if string(keep[0]) != "aa" || string(keep[1]) != "bb" {
		t.Fatalf("%q", keep)
	}
}

type errReader struct{ err error }

func (e errReader) Read([]byte) (int, error) { return 0, e.err }

func TestErrCreateMayExist(t *testing.T) {
	inner := errors.New("503")
	err := errCreateMayExist("openai", inner)
	if !errors.Is(err, inner) || !strings.Contains(err.Error(), "may still have been created") || !strings.HasPrefix(err.Error(), "openai: ") {
		t.Fatal(err)
	}
}

func TestBatchEnabled(t *testing.T) {
	b := newFakeBatcher(testCfg(), ProviderAnthropic, &fakeBatch{})
	if !b.Enabled() {
		t.Fatal("not enabled")
	}
}
