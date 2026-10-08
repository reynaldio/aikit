package guard

import "regexp"

// injectionPatterns flag text that tries to instruct the model. Chosen so that
// ordinary contract language ("shall disregard any prior agreements",
// "wajib mengabaikan perintah …") does not match.
//
// A negatable pattern doesn't count a match immediately preceded by a
// negation ("shall not ignore the instructions", "jangan abaikan aturan") —
// that is an ordinary obligation, not an instruction to the model. "DAN" (the
// jailbreak persona) is matched case-sensitively, so the Indonesian "dan"
// ("and") in "fesyen dan mode" never matches.
var injectionPatterns = []struct {
	name      string
	re        *regexp.Regexp
	negatable bool
}{
	{"ignore_instructions", regexp.MustCompile(`(?i)\b(ignore|disregard|forget)\s+(all\s+|any\s+)?(of\s+)?(the\s+|your\s+|my\s+|these\s+)?(previous\s+|prior\s+|above\s+|earlier\s+|preceding\s+|system\s+|original\s+)?(instructions?|rules|prompts?|guidelines|guardrails)\b`), true},
	{"abaikan_instruksi", regexp.MustCompile(`(?i)\b(abaikan|lupakan|hiraukan)\s+(semua\s+|any\s+)?(instruksi|aturan|prompt)\b`), true},
	{"role_override_you_are", regexp.MustCompile(`(?i:\byou\s+are\s+(now\s+)?(an?\s+)?)(?:(?i:ai|assistant|model|chatbot|unrestricted|jailbroken|free\s+of\s+rules)|DAN)\b`), false},
	{"role_override_you_act", regexp.MustCompile(`(?i)\byou\b[^.\n]{0,40}\bact\s+as\b[^.\n]{0,40}\b(no|without)\b[^.\n]{0,20}\b(restrictions?|rules)\b\s*(?:[.!,;:)]|$|\band\b|\bwhatsoever\b)`), false},
	{"role_override_act_ai", regexp.MustCompile(`(?i)\bact\s+as\s+(an?\s+)?(ai|assistant|chatbot|model|language\s+model|unrestricted\s+\w+|dan)\b[^.\n]{0,40}\b(no|without)\s+(any\s+)?(restrictions?|rules|filters|limits|guardrails)\b\s*(?:[.!,;:)]|$|\band\b|\bwhatsoever\b)`), false},
	{"role_override_pretend", regexp.MustCompile(`(?i)\b(pretend|imagine|assume|let'?s?\s+roleplay)\b[^.\n]{0,40}\byou\s+(have|had)\s+no\s+(rules|restrictions|limits|guidelines|filters)\b`), false},
	{"role_override_mode", regexp.MustCompile(`\b(?:(?i:developer|god)|DAN)\s+(?i:mode)\b|(?i:\b(jailbreak\s+(mode|prompt))\b|\byou\s+(are|'re)\s+(now\s+)?jailbroke?n?\b)`), false},
	{"reveal_prompt", regexp.MustCompile(`(?i)\b(reveal|print|show|repeat|output)\b[^.\n]{0,30}\b(system\s+prompt|your\s+instructions|hidden\s+instructions)\b|\btampilkan\b[^.\n]{0,30}\b(prompt\s+sistem|instruksi\s+(?:sistem|tersembunyi))\b`), false},
	{"fake_role_tag_xml", regexp.MustCompile(`<\s*/?\s*(system|assistant)\s*>|\[/?INST\]|<\|\s*im_(?:start|end)\s*\|>`), false},
	{"fake_role_tag_system_imperative", regexp.MustCompile(`(?im)^(system|assistant)\s*:\s+(?:you\b|ignore\b|from\s+now\s+on|new\s+instructions?|obey\b)`), false},
	{"fake_role_tag_markdown", regexp.MustCompile(`(?i)^\s*#{1,6}\s*system\b[\s\S]{0,30}(?:you|ignore|obey|from\s+now\s+on|new\s+instructions?)`), false},
	{"tool_call_json", regexp.MustCompile(`(?i)"(tool|function)(_call|_name)?\s*"?\s*:\s*"`), false},
}

// negationBefore matches a negation ending right before a match: jangan,
// tidak (boleh), not (shall/must/do not), never, don't.
var negationBefore = regexp.MustCompile(`(?i)(?:\b(?:jangan|tidak(?:\s+boleh)?|not|never)|\bdon['’]?t)\s+$`)

// ScanInjection returns the names of the injection patterns text matches (nil
// when clean). Detection only — it never blocks.
func ScanInjection(text string) []string {
	var hits []string
	for _, p := range injectionPatterns {
		if matches(p.re, text, p.negatable) {
			hits = append(hits, p.name)
		}
	}
	return hits
}

// matches reports a match of re in text; for a negatable pattern, a match
// immediately preceded by a negation doesn't count (any other match does).
func matches(re *regexp.Regexp, text string, negatable bool) bool {
	if !negatable {
		return re.MatchString(text)
	}
	for _, m := range re.FindAllStringIndex(text, -1) {
		if !negationBefore.MatchString(text[max(0, m[0]-32):m[0]]) {
			return true
		}
	}
	return false
}
