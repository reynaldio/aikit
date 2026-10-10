package guard

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/reynaldio/aikit/llm"
)

// run feeds pieces to a fresh filter, then finishes it.
func run(pieces ...string) (events []string, text string, declined bool) {
	f := newMarkerFilter(testMarker, func(ev llm.StreamEvent) {
		if ev.Kind == llm.StreamText {
			events = append(events, ev.Text)
		}
	})
	for _, p := range pieces {
		f.push(llm.StreamEvent{Kind: llm.StreamText, Text: p})
	}
	f.finish()
	return events, f.text(), f.declined
}

// checkSplit asserts the filter invariants for one delivery of full.
func checkSplit(t *testing.T, full string, pieces ...string) {
	t.Helper()
	events, text, declined := run(pieces...)
	for _, e := range events {
		if e == "" {
			t.Fatalf("%q: empty event", pieces)
		}
		if !utf8.ValidString(e) {
			t.Fatalf("%q: invalid UTF-8 event %q", pieces, e)
		}
		if strings.Contains(e, testMarker) {
			t.Fatalf("%q: event holds the marker: %q", pieces, e)
		}
		for n := 1; n < len(testMarker); n++ {
			if strings.HasSuffix(e, testMarker[:n]) {
				t.Fatalf("%q: event ends in a partial marker: %q", pieces, e)
			}
		}
	}
	joined := strings.Join(events, "")
	want, _ := StripMarker(full, testMarker)
	if joined != text || joined != want {
		t.Fatalf("%q: joined %q, text %q, want %q", pieces, joined, text, want)
	}
	if declined != strings.Contains(full, testMarker) {
		t.Fatalf("%q: declined = %v", pieces, declined)
	}
}

var filterReplies = []string{
	testMarker + " Maaf, saya hanya membantu.",
	"Jawaban: " + testMarker + " tidak bisa.",
	"Normal answer.",
	"Hello " + testMarker,
	testMarker + "\nHello",
	"é" + testMarker + " ü",
}

func TestFilterSplitAtEveryBytePosition(t *testing.T) {
	for _, full := range filterReplies {
		for i := 0; i <= len(full); i++ {
			checkSplit(t, full, full[:i], full[i:])
		}
	}
}

func TestFilterSplitAtEveryPairOfPositions(t *testing.T) {
	for _, full := range filterReplies[:2] {
		for i := 0; i <= len(full); i++ {
			for j := i; j <= len(full); j++ {
				checkSplit(t, full, full[:i], full[i:j], full[j:])
			}
		}
	}
}

func TestFilterBlanksThenMarker(t *testing.T) {
	events, text, declined := run("  ", testMarker+" Hi")
	if strings.Join(events, "") != "Hi" || text != "Hi" || !declined {
		t.Fatalf("got %q %q %v", events, text, declined)
	}
}

func TestFilterBlanksThenPlainText(t *testing.T) {
	events, text, declined := run("  ", "Hi")
	if strings.Join(events, "") != "  Hi" || text != "  Hi" || declined {
		t.Fatalf("got %q %q %v", events, text, declined)
	}
}

func TestFilterMarkerThenSpaceInNextPiece(t *testing.T) {
	events, text, _ := run(testMarker, " Hi")
	if strings.Join(events, "") != "Hi" || text != "Hi" {
		t.Fatalf("got %q %q", events, text)
	}
}

func TestFilterMarkerInTheMiddle(t *testing.T) {
	events, text, declined := run("Ini ", testMarker, " jawaban")
	if strings.Join(events, "") != "Ini jawaban" || text != "Ini jawaban" || !declined {
		t.Fatalf("got %q %q %v", events, text, declined)
	}
}

func TestFilterLooksLikeMarkerButIsNot(t *testing.T) {
	events, text, declined := run("⟦dossi:", "other⟧ text")
	if strings.Join(events, "") != "⟦dossi:other⟧ text" || text != "⟦dossi:other⟧ text" || declined {
		t.Fatalf("got %q %q %v", events, text, declined)
	}
}

func TestFilterOnlyBlanks(t *testing.T) {
	events, text, declined := run("  ")
	if strings.Join(events, "") != "  " || text != "  " || declined {
		t.Fatalf("got %q %q %v", events, text, declined)
	}
}

func TestFilterBlanksAndMarkerOnly(t *testing.T) {
	events, text, declined := run("  " + testMarker)
	if len(events) != 0 || text != "" || !declined {
		t.Fatalf("got %q %q %v", events, text, declined)
	}
}

// A known difference from StripMarker, which trims leading blanks of the whole
// reply even when text came before the marker.
func TestFilterKeepsBlanksBeforeEarlierText(t *testing.T) {
	events, text, _ := run("\n\nHello ", testMarker+" x")
	if strings.Join(events, "") != "\n\nHello x" || text != "\n\nHello x" {
		t.Fatalf("got %q %q", events, text)
	}
}

func TestFilterPassesOtherKindsThrough(t *testing.T) {
	var got []llm.StreamEvent
	f := newMarkerFilter(testMarker, func(ev llm.StreamEvent) { got = append(got, ev) })
	other := llm.StreamEvent{Kind: llm.StreamKind("other"), Text: "x"}
	f.push(other)
	f.push(llm.StreamEvent{Kind: llm.StreamText, Text: "hi"})
	f.finish()
	if len(got) != 2 || got[0] != other || got[1].Text != "hi" || f.text() != "hi" {
		t.Fatalf("got %+v, text %q", got, f.text())
	}
}
