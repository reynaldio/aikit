// Package guard is the general-purpose half of an LLM guardrail layer, applied
// once at the llm.Client seam: fencing marks outside text (uploaded files, tool
// results, memory, context) as data the model must not obey; a scanner spots
// text that tries to instruct the model, in English and Indonesian; turns make
// one user message that runs several tool rounds count once; and Wrap adds the
// data rule, size caps and per-user rate limits to every call, audits suspected
// injections and strips a refusal marker.
//
// The scanner audits and never blocks. The protection is the fence plus
// DataRule; the audit is for awareness. Product-specific policy (what topics an
// assistant covers, how a decline is worded, where audits are stored) is passed
// in through Options, not built in.
//
// Ported from dossio's pkg/aiguard, where it has run in production since
// 2026-10-05.
package guard

// Class is how a call is guarded. Options.Classify decides it per call.
type Class int

const (
	// ClassOneShot: a user action with no free conversation (summaries,
	// extraction, checks). Counted per call; gets the data rule only.
	ClassOneShot Class = iota
	// ClassConversational: a person typed free text. Counted once per turn;
	// gets the data rule AND the topic policy; refusals are stripped + reported.
	ClassConversational
	// ClassBackground: system-driven work (map/reduce passes, memory merges).
	// Never rate limited or size-capped; gets the data rule.
	ClassBackground
)
