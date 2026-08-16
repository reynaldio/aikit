package llm

import "context"

// noopClient is the placeholder used until a real provider is wired.
// Every Complete call returns ErrNotConfigured.
type noopClient struct{}

// NewNoop returns an LLM client that is never enabled.
func NewNoop() Client { return noopClient{} }

func (noopClient) Complete(_ context.Context, _ Request) (Response, error) {
	return Response{}, ErrNotConfigured
}

func (noopClient) Enabled() bool { return false }

func (noopClient) CompareTargets() []ModelRef { return nil }
