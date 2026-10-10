package llm

import (
	"errors"
	"io"
	"strings"
	"testing"
)

// sseAll drains a reader, returning every payload and the error that stopped it.
func sseAll(r io.Reader) ([]string, error) {
	s := newSSEReader(r)
	var out []string
	for {
		p, err := s.next()
		if err != nil {
			return out, err
		}
		out = append(out, string(p))
	}
}

func wantPayloads(t *testing.T, body string, want ...string) {
	t.Helper()
	got, err := sseAll(strings.NewReader(body))
	if err != io.EOF {
		t.Fatalf("err = %v, want io.EOF", err)
	}
	if strings.Join(got, "|") != strings.Join(want, "|") || len(got) != len(want) {
		t.Errorf("payloads = %q, want %q", got, want)
	}
}

func TestSSETwoEvents(t *testing.T) {
	wantPayloads(t, "data: a\n\ndata: b\n\n", "a", "b")
}

func TestSSEMultiLineDataJoined(t *testing.T) {
	wantPayloads(t, "data: a\ndata: b\n\n", "a\nb")
}

func TestSSELeadingSpace(t *testing.T) {
	wantPayloads(t, "data:x\n\ndata: x\n\ndata:  x\n\n", "x", "x", " x")
}

func TestSSECRLF(t *testing.T) {
	wantPayloads(t, "data: a\r\n\r\ndata: b\r\n\r\n", "a", "b")
}

func TestSSEIgnoredFields(t *testing.T) {
	wantPayloads(t, ": keep-alive\nevent: ping\n\nevent: msg\nid: 7\nretry: 100\n: c\ndata: a\n\n", "a")
}

func TestSSEFinalEventWithoutBlankLine(t *testing.T) {
	wantPayloads(t, "data: a\n\ndata: b", "a", "b")
}

func TestSSELargeDataLine(t *testing.T) {
	big := strings.Repeat("x", 1<<20)
	wantPayloads(t, "data: "+big+"\n\n", big)
}

type failingReader struct {
	data string
	err  error
	done bool
}

func (f *failingReader) Read(p []byte) (int, error) {
	if f.done {
		return 0, f.err
	}
	f.done = true
	return copy(p, f.data), nil
}

func TestSSEReadErrorIsReturned(t *testing.T) {
	boom := errors.New("boom")
	got, err := sseAll(&failingReader{data: "data: a\n\ndata: b\n", err: boom})
	if !errors.Is(err, boom) || err == io.EOF {
		t.Fatalf("err = %v, want boom", err)
	}
	if len(got) != 1 || got[0] != "a" {
		t.Errorf("payloads = %q, want [a]", got)
	}
}
