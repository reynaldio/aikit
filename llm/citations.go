package llm

// Citation is a web source a reply used.
//
// On Anthropic, URL is the page and CitedText is up to 150 characters quoted from it.
//
// On Gemini, URL is a vertexaisearch.cloud.google.com/grounding-api-redirect/ link
// that redirects to the page, Title is usually the site's domain, and CitedText is
// empty, because Gemini gives no text from the page.
type Citation struct {
	URL       string
	Title     string
	CitedText string
}

// dedupeCitations drops repeats and entries with no URL, keeping first-use order. Two
// different quotes from one page are both kept. It returns nil when nothing is left.
func dedupeCitations(in []Citation) []Citation {
	var out []Citation
	seen := make(map[Citation]bool, len(in))
	for _, c := range in {
		if c.URL == "" || seen[c] {
			continue
		}
		seen[c] = true
		out = append(out, c)
	}
	return out
}
