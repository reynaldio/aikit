# aikit

Shared AI-provider gateways for reynaldio projects. One public Go module; one package per
modality. Today it ships `aikit/llm`. `aikit/tts` (text-to-speech) and any future modality
are siblings added later — STT is not separate, it rides inside `llm` as a multimodal
completion.

## aikit/llm

A provider-agnostic LLM router: one `Client` interface over Anthropic, Google Gemini, and
OpenAI-compatible backends (DeepSeek, Moonshot, …), with per-profile failover and a
token-cost pricing catalog. Usage *persistence* is the consumer's job — the library emits
token counts on `Response`; wrap the `Client` to record them.

```go
import "github.com/reynaldio/aikit/llm"

c := llm.New(llm.Config{
    AnthropicAPIKey: os.Getenv("ANTHROPIC_API_KEY"),
    Profiles: map[llm.Profile]llm.ModelRef{
        llm.ProfileChat: {Provider: llm.ProviderAnthropic, Model: "claude-haiku-4-5"},
    },
    // Logger is optional; nil = the library logs nothing.
})
resp, err := c.Complete(ctx, llm.Request{
    Messages: []llm.Message{{Role: "user", Text: "halo"}},
    Profile:  llm.ProfileChat,
})
```

`Complete` returns `llm.ErrNotConfigured` when no provider key is set.

## Installing

```
go get github.com/reynaldio/aikit/llm
```

Public module — a plain `go get` works with no extra configuration (no `GOPRIVATE`, no
credentials). Versioned with SemVer tags; pin a release in your `go.mod` as usual. The
reference consumer is Nathan (this org's mobile assistant).

For co-development against a local checkout, you can temporarily add a `replace` in the
consumer's `go.mod` pointing at a local `aikit` clone — but the committed dependency is the
published tag, so containerized/CI builds resolve it straight from the module proxy.
