package guard

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"strings"
)

// Untrusted-content sources (the fence's source attribute).
const (
	SourceFile       = "file"
	SourceTool       = "tool"
	SourceMemory     = "memory"
	SourceContext    = "context"
	SourceTranscript = "transcript"
	SourceSummary    = "summary"
	SourceDocument   = "document"
)

// FenceAttrs describe a fenced block. Ref is a related record id (e.g. the file
// id) or ""; Name is a display name (file name, tool name).
type FenceAttrs struct {
	Source string
	Ref    string
	Name   string
}

var (
	tagInText = regexp.MustCompile(`(?i)<(/?)untrusted-data`)
	attrBad   = strings.NewReplacer(`"`, " ", "<", " ", ">", " ", "\n", " ", "\r", " ")
	blockRe   = regexp.MustCompile(`(?s)<untrusted-data source="([^"]*)" ref="([^"]*)" name="([^"]*)" id="([0-9a-f]{16})">\n(.*?)\n</untrusted-data id="([0-9a-f]{16})">`)
)

// Fence wraps untrusted text so the model (told by DataRule) treats it as data.
// A fence tag inside the text is neutralised (non-breaking hyphen) so the text
// can't close its own block, and every block's id is a stable keyed hash of its
// content — same (source, ref, name, text) within this process always produces
// the same id. This is deliberate: a fresh random id on every call would make
// the fenced block (and the ~10KB+ system prompt it usually sits inside) churn
// on every single turn, defeating the LLM provider's prompt caching. A document
// author still can't predict or choose the id (it's keyed by a process-lifetime
// random secret), and the neutraliser above still stops content from closing
// its own fence early.
func Fence(a FenceAttrs, text string) string {
	id := fenceID(a, text)
	safe := tagInText.ReplaceAllString(text, "<${1}untrusted‑data")
	return fmt.Sprintf("<untrusted-data source=\"%s\" ref=\"%s\" name=\"%s\" id=\"%s\">\n%s\n</untrusted-data id=\"%s\">",
		attrBad.Replace(a.Source), attrBad.Replace(a.Ref), attrBad.Replace(a.Name), id, safe, id)
}

// Block is one fenced block found in a prompt.
type Block struct {
	Source, Ref, Name, Text string
}

// FindBlocks returns the well-formed fenced blocks in s (opening and closing ids
// must match).
func FindBlocks(s string) []Block {
	var out []Block
	for _, m := range blockRe.FindAllStringSubmatch(s, -1) {
		if m[4] != m[6] {
			continue
		}
		out = append(out, Block{Source: m[1], Ref: m[2], Name: m[3], Text: m[5]})
	}
	return out
}

// fenceKey is a 32-byte secret generated once per process (crypto/rand). It
// keys the id so a document's author can't predict or choose a fence's id
// (which would otherwise let them try to forge a closing tag with a known
// id), while still making the id a pure function of the block's content
// within this process — see Fence's doc comment on why that stability matters.
var fenceKey = newFenceKey()

func newFenceKey() []byte {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return b
}

// fenceID derives a block's id from its content: HMAC-SHA256(fenceKey,
// source|ref|name|text), truncated to the first 16 hex chars (64 bits — ample
// for this purpose; the id is a cache key and a self-closing guard, not a
// security boundary on its own). Same attrs + text within this process always
// yield the same id; any difference in source/ref/name/text changes it.
func fenceID(a FenceAttrs, text string) string {
	mac := hmac.New(sha256.New, fenceKey)
	mac.Write([]byte(a.Source + "|" + a.Ref + "|" + a.Name + "|" + text))
	return hex.EncodeToString(mac.Sum(nil))[:16]
}
