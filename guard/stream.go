package guard

import (
	"strings"
	"unicode/utf8"

	"github.com/reynaldio/aikit/llm"
)

// markerFilter removes the refusal marker (and the space after it) from a
// streamed conversational reply, so the app never shows it. It holds back the
// shortest tail that could still turn into the marker, so no event ever carries
// part of it. It runs on the calling goroutine only. It assumes the marker
// does not overlap itself.
type markerFilter struct {
	marker   string
	pat      string // marker + " ": what StripMarker removes
	next     func(llm.StreamEvent)
	buf      string // received, not yet sent
	blanks   string // leading blanks held until the first real text
	started  bool   // first non-blank text has been sent
	out      strings.Builder
	declined bool
}

func newMarkerFilter(marker string, next func(llm.StreamEvent)) *markerFilter {
	return &markerFilter{marker: marker, pat: marker + " ", next: next}
}

// text is everything sent so far, which is the reply's final text.
func (f *markerFilter) text() string { return f.out.String() }

// push takes one event from the inner client.
func (f *markerFilter) push(ev llm.StreamEvent) {
	if ev.Kind != llm.StreamText {
		f.next(ev)
		return
	}
	m := f.marker
	f.buf += ev.Text

	// Remove complete markers. One at the very end keeps its place: a space
	// may still follow it in the next piece.
	if strings.Contains(f.buf, f.pat) {
		f.buf = strings.ReplaceAll(f.buf, f.pat, "")
		f.declined = true
	}
	atEnd := strings.HasSuffix(f.buf, m)
	body := f.buf
	if atEnd {
		body = f.buf[:len(f.buf)-len(m)]
	}
	if strings.Contains(body, m) {
		body = strings.ReplaceAll(body, m, "")
		f.declined = true
	}
	if atEnd {
		f.buf = body + m
	} else {
		f.buf = body
	}

	// Hold back the longest tail that is a prefix of marker+" ".
	k := 0
	for n := min(len(f.buf), len(f.pat)); n > 0; n-- {
		if strings.HasPrefix(f.pat, f.buf[len(f.buf)-n:]) {
			k = n
			break
		}
	}
	ready := f.buf[:len(f.buf)-k]
	// Never end in the middle of a UTF-8 sequence.
	for j := len(ready) - 1; j >= 0 && j >= len(ready)-utf8.UTFMax; j-- {
		if utf8.RuneStart(ready[j]) {
			if !utf8.FullRuneInString(ready[j:]) {
				ready = ready[:j]
			}
			break
		}
	}
	f.buf = f.buf[len(ready):]

	f.emitReady(ready)
}

// emitReady sends text that can no longer change. Leading blanks wait for the
// first real text: StripMarker drops them after a marker, and keeps them when
// there is none.
func (f *markerFilter) emitReady(ready string) {
	if !f.started {
		trimmed := strings.TrimLeft(ready, " \n")
		f.blanks += ready[:len(ready)-len(trimmed)]
		ready = trimmed
		if ready == "" {
			return
		}
		if !f.declined {
			ready = f.blanks + ready
		}
		f.blanks = ""
		f.started = true
	}
	if ready != "" {
		f.send(ready)
	}
}

func (f *markerFilter) send(s string) {
	f.next(llm.StreamEvent{Kind: llm.StreamText, Text: s})
	f.out.WriteString(s)
}

// finish flushes what was held back. Call it only after the inner Complete
// succeeded.
func (f *markerFilter) finish() {
	if strings.Contains(f.buf, f.marker) {
		f.buf = strings.ReplaceAll(strings.ReplaceAll(f.buf, f.pat, ""), f.marker, "")
		f.declined = true
	}
	ready := f.buf
	f.buf = ""
	f.emitReady(ready)
	if !f.started && f.blanks != "" && !f.declined {
		f.send(f.blanks)
	}
	f.blanks = ""
}
