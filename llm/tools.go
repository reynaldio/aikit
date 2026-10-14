package llm

import (
	"context"
	"errors"
	"sync"
)

// ToolFunc runs one tool call. The string it returns is sent back to the model.
// An error is sent back as a failed result (IsError), and the loop goes on.
type ToolFunc func(ctx context.Context, call ToolCall) (string, error)

// ToolLoopOptions tunes RunTools.
type ToolLoopOptions struct {
	// MaxRounds is the most calls to Complete. 0 or less means 8.
	MaxRounds int
	// Parallel runs the calls of one round at the same time. The functions
	// must then be safe to run from several goroutines.
	Parallel bool
	// OnRound, when set, is called after each reply (round is 1-based). A
	// non-nil error stops the loop and is returned.
	OnRound func(round int, resp Response) error
}

// ToolRun is what RunTools returns, also when it returns an error.
type ToolRun struct {
	// Response is the last reply received.
	Response Response
	// Messages is req.Messages plus every assistant turn and tool-result turn
	// the loop added. On a normal end it does not hold the final reply. To
	// continue the chat, add that reply first:
	//
	//	run.Messages = append(run.Messages, llm.Message{Role: "assistant", Content: run.Response.Text})
	Messages []Message
	// Rounds is every reply received, in order.
	Rounds []Response
}

var (
	// ErrToolLoopLimit means the model still wanted tools after MaxRounds calls.
	ErrToolLoopLimit = errors.New("llm: tool loop reached MaxRounds")
	// ErrToolCallsTruncated means the reply with tool calls was cut off at the
	// token limit. The half-written calls are not run.
	ErrToolCallsTruncated = errors.New("llm: tool calls were cut off at the token limit")
)

const defaultMaxRounds = 8

// RunTools calls c.Complete in a loop. When a reply has tool calls, it runs
// them with tools, adds the assistant turn and one user turn with all the
// results to the history, and calls Complete again. It stops when a reply has
// no tool calls.
//
// A call with no entry in tools, or a function that returns an error, gives a
// result with IsError set; the loop goes on. A panic in a function is not
// caught. Every function gets ctx.
//
// Every field of req except Messages is sent unchanged each round. With
// OnEvent set, text from every round goes to the same callback, so the join
// rule holds per round, not for the whole run. With guard.Wrap, call
// guard.BeginTurn(ctx) before RunTools so the whole loop counts as one turn.
// With WithUsage, every round is reported on its own.
func RunTools(ctx context.Context, c Client, req Request, tools map[string]ToolFunc, opts ToolLoopOptions) (ToolRun, error) {
	maxRounds := opts.MaxRounds
	if maxRounds <= 0 {
		maxRounds = defaultMaxRounds
	}
	// Step 1: copy the history. The caller's slice is never changed.
	msgs := append([]Message(nil), req.Messages...)
	var run ToolRun
	for round := 1; ; round++ {
		run.Messages = msgs
		// Step 2.1: stop if ctx has ended.
		if err := ctx.Err(); err != nil {
			return run, err
		}
		// Step 2.2: ask the model.
		rreq := req
		rreq.Messages = msgs
		resp, err := c.Complete(ctx, rreq)
		// Step 2.3: keep the reply if it has tokens or there is no error.
		if err == nil || hasUsage(resp) {
			run.Rounds = append(run.Rounds, resp)
			run.Response = resp
		}
		// Step 2.4: return a Complete error unchanged.
		if err != nil {
			return run, err
		}
		// Step 2.5
		if opts.OnRound != nil {
			if err := opts.OnRound(round, resp); err != nil {
				return run, err
			}
		}
		// Step 2.6: no calls is the normal end.
		if len(resp.ToolCalls) == 0 {
			return run, nil
		}
		// Step 2.7: do not run half-written calls.
		if resp.StopReason == StopTruncated {
			return run, ErrToolCallsTruncated
		}
		// Step 2.8
		if round == maxRounds {
			return run, ErrToolLoopLimit
		}
		// Step 2.9: run the calls, then add the two turns.
		results := runToolCalls(ctx, resp.ToolCalls, tools, opts.Parallel)
		if err := ctx.Err(); err != nil {
			return run, err
		}
		msgs = append(msgs,
			Message{Role: "assistant", Content: resp.Text, ToolCalls: resp.ToolCalls},
			Message{Role: "user", ToolResults: results},
		)
	}
}

// runToolCalls runs every call and returns the results in call order. A failed
// call does not stop the others.
func runToolCalls(ctx context.Context, calls []ToolCall, tools map[string]ToolFunc, parallel bool) []ToolResult {
	results := make([]ToolResult, len(calls))
	// Each call writes only its own results[i], so no lock is needed.
	runOne := func(i int) {
		call := calls[i]
		results[i] = ToolResult{ToolCallID: call.ID}
		fn, ok := tools[call.Name]
		if !ok {
			results[i].Content = "unknown tool: " + call.Name
			results[i].IsError = true
			return
		}
		out, err := fn(ctx, call)
		if err != nil {
			results[i].Content = err.Error()
			results[i].IsError = true
			return
		}
		results[i].Content = out
	}
	if !parallel || len(calls) == 1 {
		for i := range calls {
			runOne(i)
		}
		return results
	}
	var wg sync.WaitGroup
	for i := range calls {
		wg.Add(1)
		go func() {
			defer wg.Done()
			runOne(i)
		}()
	}
	wg.Wait()
	return results
}
