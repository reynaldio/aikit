package llm_test

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/reynaldio/aikit/llm"
	"github.com/reynaldio/aikit/llm/llmtest"
)

func toolReply(text string, calls ...llm.ToolCall) llmtest.Reply {
	return llmtest.Reply{Response: llm.Response{Text: text, ToolCalls: calls, StopReason: llm.StopToolUse}}
}

func endReply(text string) llmtest.Reply {
	return llmtest.Reply{Response: llm.Response{Text: text, StopReason: llm.StopEndTurn}}
}

func call(id, name string) llm.ToolCall { return llm.ToolCall{ID: id, Name: name} }

func echoTool(out string) llm.ToolFunc {
	return func(ctx context.Context, c llm.ToolCall) (string, error) { return out, nil }
}

func TestRunTools_NoCalls(t *testing.T) {
	c := llmtest.New(endReply("hi"))
	in := []llm.Message{{Role: "user", Content: "q"}}
	run, err := llm.RunTools(context.Background(), c, llm.Request{Messages: in}, nil, llm.ToolLoopOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(run.Rounds) != 1 || run.Response.Text != "hi" {
		t.Fatalf("run = %+v", run)
	}
	if !reflect.DeepEqual(run.Messages, in) {
		t.Fatalf("messages = %+v", run.Messages)
	}
}

func TestRunTools_TwoRoundsOneUserTurn(t *testing.T) {
	c := llmtest.New(
		toolReply("thinking", call("1", "a"), call("2", "b")),
		endReply("done"),
	)
	tools := map[string]llm.ToolFunc{"a": echoTool("A"), "b": echoTool("B")}
	in := []llm.Message{{Role: "user", Content: "q"}}
	run, err := llm.RunTools(context.Background(), c, llm.Request{Messages: in}, tools, llm.ToolLoopOptions{})
	if err != nil {
		t.Fatal(err)
	}
	reqs := c.Requests()
	if len(reqs) != 2 {
		t.Fatalf("requests = %d", len(reqs))
	}
	want := []llm.Message{
		{Role: "user", Content: "q"},
		{Role: "assistant", Content: "thinking", ToolCalls: []llm.ToolCall{call("1", "a"), call("2", "b")}},
		{Role: "user", ToolResults: []llm.ToolResult{
			{ToolCallID: "1", Content: "A"},
			{ToolCallID: "2", Content: "B"},
		}},
	}
	if !reflect.DeepEqual(reqs[1].Messages, want) {
		t.Fatalf("second request messages = %+v", reqs[1].Messages)
	}
	if !reflect.DeepEqual(run.Messages, want) {
		t.Fatalf("run messages = %+v", run.Messages)
	}
	if len(run.Rounds) != 2 || run.Response.Text != "done" {
		t.Fatalf("run = %+v", run)
	}
}

func TestRunTools_ToolErrorGoesOn(t *testing.T) {
	c := llmtest.New(toolReply("", call("1", "bad")), endReply("ok"))
	tools := map[string]llm.ToolFunc{"bad": func(context.Context, llm.ToolCall) (string, error) {
		return "ignored", errors.New("boom")
	}}
	run, err := llm.RunTools(context.Background(), c, llm.Request{}, tools, llm.ToolLoopOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got := run.Messages[1].ToolResults
	want := []llm.ToolResult{{ToolCallID: "1", Content: "boom", IsError: true}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("results = %+v", got)
	}
}

func TestRunTools_UnknownTool(t *testing.T) {
	c := llmtest.New(toolReply("", call("1", "x"), call("2", "y")), endReply("ok"))
	run, err := llm.RunTools(context.Background(), c, llm.Request{}, nil, llm.ToolLoopOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want := []llm.ToolResult{
		{ToolCallID: "1", Content: "unknown tool: x", IsError: true},
		{ToolCallID: "2", Content: "unknown tool: y", IsError: true},
	}
	if !reflect.DeepEqual(run.Messages[1].ToolResults, want) {
		t.Fatalf("results = %+v", run.Messages[1].ToolResults)
	}
	if len(run.Rounds) != 2 {
		t.Fatalf("rounds = %d", len(run.Rounds))
	}
}

func TestRunTools_TruncatedCalls(t *testing.T) {
	c := llmtest.New(llmtest.Reply{Response: llm.Response{
		ToolCalls: []llm.ToolCall{call("1", "a")}, StopReason: llm.StopTruncated,
	}})
	ran := false
	tools := map[string]llm.ToolFunc{"a": func(context.Context, llm.ToolCall) (string, error) { ran = true; return "", nil }}
	run, err := llm.RunTools(context.Background(), c, llm.Request{}, tools, llm.ToolLoopOptions{})
	if !errors.Is(err, llm.ErrToolCallsTruncated) {
		t.Fatalf("err = %v", err)
	}
	if ran {
		t.Fatal("tool ran")
	}
	if len(run.Rounds) != 1 {
		t.Fatalf("rounds = %d", len(run.Rounds))
	}
}

func TestRunTools_MaxRounds(t *testing.T) {
	c := llmtest.New(toolReply("", call("1", "a")), toolReply("", call("2", "a")), toolReply("", call("3", "a")))
	n := 0
	tools := map[string]llm.ToolFunc{"a": func(context.Context, llm.ToolCall) (string, error) { n++; return "", nil }}
	run, err := llm.RunTools(context.Background(), c, llm.Request{}, tools, llm.ToolLoopOptions{MaxRounds: 2})
	if !errors.Is(err, llm.ErrToolLoopLimit) {
		t.Fatalf("err = %v", err)
	}
	if got := len(c.Requests()); got != 2 {
		t.Fatalf("Complete calls = %d", got)
	}
	if n != 1 {
		t.Fatalf("tool ran %d times; the last round's calls must not run", n)
	}
	if len(run.Rounds) != 2 {
		t.Fatalf("rounds = %d", len(run.Rounds))
	}
}

func TestRunTools_DefaultMaxRoundsIs8(t *testing.T) {
	var replies []llmtest.Reply
	for i := 0; i < 9; i++ {
		replies = append(replies, toolReply("", call("x", "a")))
	}
	c := llmtest.New(replies...)
	tools := map[string]llm.ToolFunc{"a": echoTool("")}
	_, err := llm.RunTools(context.Background(), c, llm.Request{}, tools, llm.ToolLoopOptions{})
	if !errors.Is(err, llm.ErrToolLoopLimit) {
		t.Fatalf("err = %v", err)
	}
	if got := len(c.Requests()); got != 8 {
		t.Fatalf("Complete calls = %d", got)
	}
}

func TestRunTools_OnRoundError(t *testing.T) {
	stop := errors.New("stop")
	c := llmtest.New(toolReply("", call("1", "a")), endReply("x"))
	var rounds []int
	opts := llm.ToolLoopOptions{OnRound: func(round int, resp llm.Response) error {
		rounds = append(rounds, round)
		return stop
	}}
	run, err := llm.RunTools(context.Background(), c, llm.Request{}, map[string]llm.ToolFunc{"a": echoTool("")}, opts)
	if err != stop {
		t.Fatalf("err = %v", err)
	}
	if !reflect.DeepEqual(rounds, []int{1}) || len(c.Requests()) != 1 || len(run.Rounds) != 1 {
		t.Fatalf("rounds=%v requests=%d run=%+v", rounds, len(c.Requests()), run)
	}
}

func TestRunTools_CompleteErrorRound2(t *testing.T) {
	boom := errors.New("provider down")
	c := llmtest.New(toolReply("", call("1", "a")), llmtest.Reply{Err: boom})
	run, err := llm.RunTools(context.Background(), c, llm.Request{}, map[string]llm.ToolFunc{"a": echoTool("res")}, llm.ToolLoopOptions{})
	if err != boom {
		t.Fatalf("err = %v", err)
	}
	if len(run.Rounds) != 1 {
		t.Fatalf("rounds = %d", len(run.Rounds))
	}
	if run.Rounds[0].Text != "" || !reflect.DeepEqual(run.Rounds[0].ToolCalls, []llm.ToolCall{call("1", "a")}) {
		t.Errorf("round 1 = %+v, want the reply with call 1", run.Rounds[0])
	}
	wantTail := llm.Message{Role: "user", ToolResults: []llm.ToolResult{{ToolCallID: "1", Content: "res"}}}
	if n := len(run.Messages); n != 2 || !reflect.DeepEqual(run.Messages[n-1], wantTail) {
		t.Errorf("messages = %+v, want the assistant turn then round 1's tool result", run.Messages)
	}
}

func TestRunTools_RequestFieldsReachEveryRound(t *testing.T) {
	c := llmtest.New(toolReply("one", call("1", "a")), endReply("two"))
	var got []string
	tools := []llm.ToolDef{{Name: "a"}}
	req := llm.Request{
		Tools: tools,
		OnEvent: func(ev llm.StreamEvent) {
			if ev.Kind == llm.StreamText {
				got = append(got, ev.Text)
			}
		},
	}
	if _, err := llm.RunTools(context.Background(), c, req, map[string]llm.ToolFunc{"a": echoTool("x")}, llm.ToolLoopOptions{}); err != nil {
		t.Fatal(err)
	}
	reqs := c.Requests()
	if len(reqs) != 2 {
		t.Fatalf("requests = %d", len(reqs))
	}
	for i, r := range reqs {
		if !reflect.DeepEqual(r.Tools, tools) {
			t.Errorf("round %d Tools = %+v, want %+v", i+1, r.Tools, tools)
		}
		if r.OnEvent == nil {
			t.Fatalf("round %d has no OnEvent", i+1)
		}
	}
	// Calling each round's OnEvent must reach the caller's recorder.
	got = nil
	for i, r := range reqs {
		r.OnEvent(llm.StreamEvent{Kind: llm.StreamText, Text: "ping"})
		if len(got) != i+1 || got[i] != "ping" {
			t.Errorf("round %d OnEvent did not reach the caller: %q", i+1, got)
		}
	}
}

func TestRunTools_ErrorWithUsageKeptInRounds(t *testing.T) {
	boom := errors.New("refused")
	c := llmtest.New(llmtest.Reply{Err: boom, Response: llm.Response{Text: "part", InputTokens: 5, OutputTokens: 2}})
	run, err := llm.RunTools(context.Background(), c, llm.Request{}, nil, llm.ToolLoopOptions{})
	if err != boom {
		t.Fatalf("err = %v", err)
	}
	if len(run.Rounds) != 1 || run.Response.Text != "part" {
		t.Fatalf("run = %+v", run)
	}
}

func TestRunTools_ParallelRunsAtOnce(t *testing.T) {
	c := llmtest.New(toolReply("", call("1", "a"), call("2", "b")), endReply("done"))
	started := make(chan string, 2)
	release := make(chan struct{})
	mk := func(out string) llm.ToolFunc {
		return func(ctx context.Context, tc llm.ToolCall) (string, error) {
			started <- tc.ID
			<-release
			return out, nil
		}
	}
	tools := map[string]llm.ToolFunc{"a": mk("A"), "b": mk("B")}
	done := make(chan struct{})
	var run llm.ToolRun
	var err error
	go func() {
		defer close(done)
		run, err = llm.RunTools(context.Background(), c, llm.Request{}, tools, llm.ToolLoopOptions{Parallel: true})
	}()
	for i := 0; i < 2; i++ {
		select {
		case <-started:
		case <-time.After(5 * time.Second):
			close(release)
			t.Fatal("tools did not run at the same time")
		}
	}
	close(release)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("RunTools did not return")
	}
	if err != nil {
		t.Fatal(err)
	}
	want := []llm.ToolResult{{ToolCallID: "1", Content: "A"}, {ToolCallID: "2", Content: "B"}}
	if !reflect.DeepEqual(run.Messages[1].ToolResults, want) {
		t.Fatalf("results = %+v", run.Messages[1].ToolResults)
	}
}

func TestRunTools_SequentialRunsOneByOne(t *testing.T) {
	c := llmtest.New(toolReply("", call("1", "a"), call("2", "b")), endReply("done"))
	started := make(chan string, 2)
	release := make(chan struct{})
	mk := func(out string) llm.ToolFunc {
		return func(ctx context.Context, tc llm.ToolCall) (string, error) {
			started <- tc.ID
			<-release
			return out, nil
		}
	}
	tools := map[string]llm.ToolFunc{"a": mk("A"), "b": mk("B")}
	done := make(chan struct{})
	var run llm.ToolRun
	var err error
	go func() {
		defer close(done)
		run, err = llm.RunTools(context.Background(), c, llm.Request{}, tools, llm.ToolLoopOptions{})
	}()
	select {
	case id := <-started:
		if id != "1" {
			close(release)
			t.Fatalf("first started = %s", id)
		}
	case <-time.After(5 * time.Second):
		close(release)
		t.Fatal("first tool did not start")
	}
	select {
	case id := <-started:
		close(release)
		t.Fatalf("tool %s started before the first one ended", id)
	default:
	}
	close(release)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("RunTools did not return")
	}
	if err != nil {
		t.Fatal(err)
	}
	want := []llm.ToolResult{{ToolCallID: "1", Content: "A"}, {ToolCallID: "2", Content: "B"}}
	if !reflect.DeepEqual(run.Messages[1].ToolResults, want) {
		t.Fatalf("results = %+v", run.Messages[1].ToolResults)
	}
}

func TestRunTools_DoesNotChangeCallerMessages(t *testing.T) {
	c := llmtest.New(toolReply("", call("1", "a")), endReply("done"))
	backing := make([]llm.Message, 1, 10)
	backing[0] = llm.Message{Role: "user", Content: "q"}
	_, err := llm.RunTools(context.Background(), c, llm.Request{Messages: backing},
		map[string]llm.ToolFunc{"a": echoTool("")}, llm.ToolLoopOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(backing) != 1 {
		t.Fatalf("len = %d", len(backing))
	}
	if extra := backing[:2][1]; !reflect.DeepEqual(extra, llm.Message{}) {
		t.Fatalf("caller's backing array was written: %+v", extra)
	}
}

func TestRunTools_ToolCancelsContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c := llmtest.New(toolReply("", call("1", "a")), endReply("never"))
	tools := map[string]llm.ToolFunc{"a": func(context.Context, llm.ToolCall) (string, error) {
		cancel()
		return "x", nil
	}}
	run, err := llm.RunTools(ctx, c, llm.Request{}, tools, llm.ToolLoopOptions{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if got := len(c.Requests()); got != 1 {
		t.Fatalf("Complete calls = %d", got)
	}
	if len(run.Messages) != 0 {
		t.Fatalf("results were appended: %+v", run.Messages)
	}
}
