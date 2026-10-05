// Package decide is a typed-decision seam: instead of generating text, a decision model
// evaluates one state against a map of typed questions and returns a calibrated answer
// per question — a yes/no probability, a choice from a closed list, or a score on a
// rubric. The backend today is TypeSafe's Jev (POST /v1/systemone).
//
// Use it for the cheap, high-volume decision points an LLM is overkill for — routing,
// triage, severity, gating — where a probability you can threshold beats prose you
// have to parse. It is a sibling of `llm`, not a provider behind it: the request and
// response shapes have nothing in common with a chat completion.
//
// This is the `decide` package of the shared aikit module (github.com/reynaldio/aikit).
package decide

import (
	"context"
	"encoding/json"
	"fmt"
)

// QuestionType is one of the three decision primitives.
type QuestionType string

const (
	// TypeNoul is a yes/no question answered with P(yes) in [0, 1].
	TypeNoul QuestionType = "noul"
	// TypeChoice picks one option from an explicit list (at most 255).
	TypeChoice QuestionType = "choice"
	// TypeScore places the state on an ordered rubric of 2–10 levels.
	TypeScore QuestionType = "score"
)

// Question is one typed question. Build it with Noul, Choice or Score rather than as a
// literal — each type uses a different criteria shape on the wire.
type Question struct {
	Type         QuestionType
	Instructions string
	// Options maps an option key to what it means. For TypeChoice the keys ARE the
	// answers the model picks between; for TypeNoul the keys are "true"/"false" and
	// refine what yes and no mean (optional).
	Options map[string]string
	// Levels is the ordered rubric for TypeScore, lowest first. A level's index is
	// its score value, so Levels[1] scores 1.
	Levels []string
}

// Noul asks a yes/no question. ifTrue/ifFalse define what yes and no mean; either may
// be empty, and with both empty the instructions alone decide.
func Noul(instructions, ifTrue, ifFalse string) Question {
	q := Question{Type: TypeNoul, Instructions: instructions}
	if ifTrue != "" || ifFalse != "" {
		q.Options = map[string]string{}
		if ifTrue != "" {
			q.Options["true"] = ifTrue
		}
		if ifFalse != "" {
			q.Options["false"] = ifFalse
		}
	}
	return q
}

// Choice asks the model to pick one of options (key → description).
func Choice(instructions string, options map[string]string) Question {
	return Question{Type: TypeChoice, Instructions: instructions, Options: options}
}

// Score asks the model to rate the state on levels, lowest first.
func Score(instructions string, levels ...string) Question {
	return Question{Type: TypeScore, Instructions: instructions, Levels: levels}
}

// MarshalJSON renders the wire shape: criteria is an object for noul/choice and an
// array for score.
func (q Question) MarshalJSON() ([]byte, error) {
	var criteria any
	switch q.Type {
	case TypeScore:
		if len(q.Levels) > 0 {
			criteria = q.Levels
		}
	default:
		if len(q.Options) > 0 {
			criteria = q.Options
		}
	}
	return json.Marshal(struct {
		Type         QuestionType `json:"type"`
		Instructions string       `json:"instructions"`
		Criteria     any          `json:"criteria,omitempty"`
	}{q.Type, q.Instructions, criteria})
}

// Request evaluates State against every question in one pass.
type Request struct {
	// Model overrides Config.Model for this call (e.g. pin "jev-1.13" for an eval).
	Model string
	// State is what is being judged: a string for text, or any JSON-marshalable value
	// for structured data.
	State any
	// Questions are keyed by a caller-chosen ID; Response.Answers uses the same keys.
	Questions map[string]Question
}

// Answer is the result for one question. Which fields are set depends on Type:
//   - TypeNoul:   Noul (P(yes)).
//   - TypeChoice: Choice, Probabilities (option key → p), Confidence.
//   - TypeScore:  Score, Probabilities and Legend (both keyed by level index as a
//     string, "0", "1", …), Confidence.
//
// Score is the probability-weighted level, so it can land BETWEEN levels (1.05 on a
// 0–2 rubric). Round it yourself if you need a discrete level, or read the most
// probable level off Probabilities.
type Answer struct {
	Type          QuestionType       `json:"type"`
	Noul          float64            `json:"noul"`
	Choice        string             `json:"choice"`
	Score         float64            `json:"score"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	Legend        map[string]string  `json:"legend"`
}

// Response carries one Answer per requested question. Model is the concrete version
// that served the call (e.g. "jev-1.13.0" for "jev-latest"); InputTokens/OutputTokens
// feed cost metering — price them with llm.PriceBook, which carries Jev's rates.
type Response struct {
	Model        string
	Answers      map[string]Answer
	InputTokens  int
	OutputTokens int
}

// Client is the decision interface used by every call site.
type Client interface {
	// Evaluate answers every question in req, or returns ErrNotConfigured when no
	// backend is wired. A nil error guarantees an Answer for every question.
	Evaluate(ctx context.Context, req Request) (Response, error)
	// Enabled reports whether a real backend is configured.
	Enabled() bool
}

// noopClient is used when no API key is set, so the app runs without decisions.
type noopClient struct{}

// NewNoop returns a decision client that is never enabled.
func NewNoop() Client { return noopClient{} }

func (noopClient) Evaluate(context.Context, Request) (Response, error) {
	return Response{}, ErrNotConfigured
}

func (noopClient) Enabled() bool { return false }

// validate catches a request that cannot succeed before it costs a round trip.
func (req Request) validate() error {
	if req.State == nil {
		return fmt.Errorf("decide: state is required")
	}
	if len(req.Questions) == 0 {
		return fmt.Errorf("decide: at least one question is required")
	}
	for id, q := range req.Questions {
		switch q.Type {
		case TypeNoul:
		case TypeChoice:
			if n := len(q.Options); n < 2 || n > 255 {
				return fmt.Errorf("decide: question %q: choice needs 2–255 options, got %d", id, n)
			}
		case TypeScore:
			if n := len(q.Levels); n < 2 || n > 10 {
				return fmt.Errorf("decide: question %q: score needs 2–10 levels, got %d", id, n)
			}
		default:
			return fmt.Errorf("decide: question %q: unknown type %q", id, q.Type)
		}
		if q.Instructions == "" {
			return fmt.Errorf("decide: question %q: instructions are required", id)
		}
	}
	return nil
}
