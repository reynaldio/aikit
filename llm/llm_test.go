package llm

import (
	"context"
	"errors"
	"math"
	"testing"
)

// fakeProvider records which models it was asked for and returns a canned result
// per model ("" key = default). An entry in fail makes that model error.
type fakeProvider struct {
	calls []string
	fail  map[string]error
}

func (f *fakeProvider) complete(_ context.Context, model string, _ int, _ Request) (Response, error) {
	f.calls = append(f.calls, model)
	if err, ok := f.fail[model]; ok {
		return Response{}, err
	}
	return Response{Text: "ok:" + model}, nil
}

func newTestRouter(a, g *fakeProvider) *router {
	return &router{
		providers: map[Provider]provider{
			ProviderAnthropic: a,
			ProviderGoogle:    g,
		},
		profiles: map[Profile]ModelRef{
			ProfileFast:   {Provider: ProviderGoogle, Model: "flash-lite"},
			ProfileChat:   {Provider: ProviderGoogle, Model: "flash"},
			ProfileDeep:       {Provider: ProviderAnthropic, Model: "opus"},
			ProfileVision:     {Provider: ProviderGoogle, Model: "flash"},
			ProfileDocument:   {Provider: ProviderGoogle, Model: "flash"},
			ProfileTranscribe: {Provider: ProviderGoogle, Model: "flash"},
			ProfileSearch:     {Provider: ProviderAnthropic, Model: "haiku"},
		},
		fallbacks: map[Profile]ModelRef{
			ProfileChat: {Provider: ProviderAnthropic, Model: "haiku"},
		},
		maxTokens: 100,
	}
}

func TestResolveRoutesTaskToProfileModel(t *testing.T) {
	r := newTestRouter(&fakeProvider{}, &fakeProvider{})
	cases := []struct {
		task  Task
		model string
	}{
		{TaskChat, "flash"},
		{TaskNudge, "flash"},   // nudge rides the chat profile
		{TaskClassify, "flash-lite"},
		{TaskExtract, "flash-lite"},
		{TaskReason, "opus"},
		{TaskSearch, "haiku"},
		{TaskVision, "flash"},
		{TaskDocument, "flash"},   // documents have their own profile (PDF-native)
		{TaskTranscribe, "flash"}, // speech-to-text has its own profile
	}
	for _, c := range cases {
		ref, _ := r.resolve(Request{Task: c.task})
		if ref.Model != c.model {
			t.Errorf("task %s: got model %q, want %q", c.task, ref.Model, c.model)
		}
	}
}

func TestResolveLegacyTier(t *testing.T) {
	r := newTestRouter(&fakeProvider{}, &fakeProvider{})
	if ref, _ := r.resolve(Request{}); ref.Model != "flash-lite" {
		t.Errorf("default tier (cheap) should hit fast, got %q", ref.Model)
	}
	if ref, _ := r.resolve(Request{Tier: TierPremium}); ref.Model != "opus" {
		t.Errorf("premium tier should hit deep, got %q", ref.Model)
	}
}

func TestResolveDegradesWhenProviderUnconfigured(t *testing.T) {
	// Only Anthropic configured; chat profile points at Google → resolve must degrade
	// to a configured profile instead of failing.
	a := &fakeProvider{}
	r := newTestRouter(a, &fakeProvider{})
	delete(r.providers, ProviderGoogle)
	ref, _ := r.resolve(Request{Task: TaskChat})
	if ref.Provider != ProviderAnthropic {
		t.Fatalf("expected degradation to a configured provider, got %+v", ref)
	}
}

func TestCompleteFailsOverToConfiguredFallback(t *testing.T) {
	g := &fakeProvider{fail: map[string]error{"flash": errors.New("gemini: denied (status 403)")}}
	a := &fakeProvider{}
	r := newTestRouter(a, g)
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat})
	if err != nil {
		t.Fatalf("expected fallback to answer, got error: %v", err)
	}
	// The response must be stamped with the model that ACTUALLY answered — the
	// usage ledger attributes failover days from this.
	if resp.Model != "haiku" || resp.Provider != ProviderAnthropic {
		t.Fatalf("expected haiku/anthropic stamp, got %s/%s", resp.Model, resp.Provider)
	}
	if len(a.calls) != 1 || a.calls[0] != "haiku" {
		t.Fatalf("expected exactly one fallback call to haiku, got %v", a.calls)
	}
}

func TestCompleteExplicitModelNeverFailsOver(t *testing.T) {
	// The admin Compare tool measures THAT model — a silent substitution would lie.
	g := &fakeProvider{fail: map[string]error{"flash": errors.New("gemini: overloaded (status 503)")}}
	a := &fakeProvider{}
	r := newTestRouter(a, g)
	_, err := r.Complete(context.Background(), Request{Model: &ModelRef{Provider: ProviderGoogle, Model: "flash"}})
	if err == nil {
		t.Fatal("expected the explicit-model error to surface")
	}
	if len(a.calls) != 0 {
		t.Fatalf("no fallback may fire for explicit models, but got calls %v", a.calls)
	}
}

func TestCompleteNoFailoverOnContentError(t *testing.T) {
	// A content-shaped error (not throttle/auth/network) would likely fail anywhere —
	// surface it instead of burning a second call.
	g := &fakeProvider{fail: map[string]error{"flash": errors.New("gemini: request contains an invalid argument")}}
	a := &fakeProvider{}
	r := newTestRouter(a, g)
	_, err := r.Complete(context.Background(), Request{Task: TaskChat})
	if err == nil {
		t.Fatal("expected the content error to surface")
	}
	if len(a.calls) != 0 {
		t.Fatalf("content errors must not fail over, but got calls %v", a.calls)
	}
}

func TestRefusalFailsOverToConfiguredFallback(t *testing.T) {
	// A safety refusal is a 200, not a transport failure — but classifiers differ per
	// model, so the profile's fallback is exactly the model that may still answer.
	g := &fakeProvider{fail: map[string]error{
		"flash": &RefusalError{Provider: ProviderGoogle, Model: "flash", Category: "cyber"},
	}}
	a := &fakeProvider{}
	r := newTestRouter(a, g)
	resp, err := r.Complete(context.Background(), Request{Task: TaskChat})
	if err != nil {
		t.Fatalf("expected fallback to answer a refusal, got error: %v", err)
	}
	if resp.Model != "haiku" {
		t.Fatalf("expected the fallback model to answer, got %q", resp.Model)
	}
}

func TestRefusalSurfacesAsErrorNotEmptyString(t *testing.T) {
	// The whole point of ErrRefused: without it a refusal returns ("", nil) and the
	// caller silently ships an empty answer.
	refusal := &RefusalError{Provider: ProviderGoogle, Model: "flash", Category: "cyber", Explanation: "declined"}
	g := &fakeProvider{fail: map[string]error{"flash": refusal}}
	a := &fakeProvider{}
	r := newTestRouter(a, g)
	// NoFallback so the refusal is the final word rather than being rescued.
	_, err := r.Complete(context.Background(), Request{Task: TaskChat, NoFallback: true})
	if err == nil {
		t.Fatal("a refusal must surface as an error, never as an empty response")
	}
	if !errors.Is(err, ErrRefused) {
		t.Fatalf("expected errors.Is(err, ErrRefused), got %v", err)
	}
	var re *RefusalError
	if !errors.As(err, &re) || re.Category != "cyber" {
		t.Fatalf("expected the cyber category to survive unwrapping, got %v", err)
	}
	if len(a.calls) != 0 {
		t.Fatalf("NoFallback must suppress the retry, but got calls %v", a.calls)
	}
}

func TestOutputConfigOmittedWhenUnset(t *testing.T) {
	// Sending an empty output_config would override the model's own defaults; the
	// request must carry the field only when the caller asked for something.
	if _, ok := outputConfig(Request{}); ok {
		t.Fatal("expected no output_config when neither effort nor schema is set")
	}
	oc, ok := outputConfig(Request{Effort: EffortXHigh})
	if !ok || string(oc.Effort) != "xhigh" {
		t.Fatalf("expected effort xhigh to map through, got %q (ok=%v)", oc.Effort, ok)
	}
	oc, ok = outputConfig(Request{JSONSchema: map[string]any{"type": "object"}})
	if !ok || oc.Format.Schema == nil {
		t.Fatal("expected a JSON schema to map onto output_config.format")
	}
}

func TestShouldFailover(t *testing.T) {
	yes := []string{
		"gemini: quota exceeded (status 429)",
		"anthropic: overloaded (status 503)",
		"gemini: Your project has been denied access. Please contact support. (status 403)",
		"unauthorized (status 401)",
		"models/gemini-2.5-flash is no longer available to new users (status 404)",
		"Post \"https://x\": connection refused",
		"context deadline exceeded",
		"bad gateway (status 502)",
	}
	no := []string{
		"gemini: request contains an invalid argument",
		"prompt blocked by safety settings",
	}
	for _, s := range yes {
		if !shouldFailover(errors.New(s)) {
			t.Errorf("expected failover for %q", s)
		}
	}
	for _, s := range no {
		if shouldFailover(errors.New(s)) {
			t.Errorf("did NOT expect failover for %q", s)
		}
	}
	if shouldFailover(nil) {
		t.Error("nil error must not fail over")
	}
}

func TestPriceBookResolutionOrder(t *testing.T) {
	fetched := map[string]Rate{"m": {Input: 1, Output: 2, CachedRead: 0.1}}
	overrides := map[string]Rate{"m": {Input: 10, Output: 20, CachedRead: 1}}
	b := NewPriceBook(fetched, overrides)
	if got := b.Rate("m").Input; got != 10 {
		t.Errorf("override should win, got input rate %v", got)
	}
	b2 := NewPriceBook(fetched, nil)
	if got := b2.Rate("m").Input; got != 1 {
		t.Errorf("fetched should win over defaults, got %v", got)
	}
	if got := NewPriceBook(nil, nil).Rate("no-such-model"); got != (Rate{}) {
		t.Errorf("unknown model should resolve to the zero rate, got %+v", got)
	}
}

func TestRateCostMath(t *testing.T) {
	// Rates are USD per MILLION tokens.
	r := Rate{Input: 1, Output: 5, CachedRead: 0.1}
	got := r.Cost(1_000_000, 2_000_000, 500_000)
	want := 1.0 + 10.0 + 0.05
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("cost = %v, want %v", got, want)
	}
	if got := (Rate{}).Cost(1000, 1000, 1000); got != 0 {
		t.Errorf("zero rate must cost 0, got %v", got)
	}
}
