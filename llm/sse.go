package llm

import (
	"bufio"
	"bytes"
	"io"
)

// sseMaxLine bounds one SSE line. A single chunk can carry a large piece of
// tool-call arguments, so the scanner's 64 KiB default is too small.
const sseMaxLine = 8 << 20

// sseReader reads server-sent events: the OpenAI-compatible and Gemini streams.
type sseReader struct {
	sc *bufio.Scanner
}

func newSSEReader(r io.Reader) *sseReader {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), sseMaxLine)
	return &sseReader{sc: sc}
}

// next returns the next event's data payload (several data lines joined with
// "\n"). It returns io.EOF at a clean end of the body, and any read error as is,
// so callers can pass it through readError. Comments and the event/id/retry
// fields are ignored, and an event with no data line yields nothing.
func (s *sseReader) next() ([]byte, error) {
	var data []byte
	has := false
	for s.sc.Scan() {
		line := bytes.TrimSuffix(s.sc.Bytes(), []byte("\r"))
		if len(line) == 0 {
			if has {
				return data, nil
			}
			continue
		}
		rest, ok := bytes.CutPrefix(line, []byte("data:"))
		if !ok {
			continue // comment, event:, id:, retry:
		}
		rest = bytes.TrimPrefix(rest, []byte(" "))
		if has {
			data = append(data, '\n')
		}
		data = append(data, rest...)
		has = true
	}
	if err := s.sc.Err(); err != nil {
		return nil, err
	}
	if has {
		return data, nil
	}
	return nil, io.EOF
}
