package main

import (
	"fmt"
	"regexp"
	"unicode"
)

// preTokenizer splits text into the pieces that byte-level BPE encodes
// separately, as GPT-2's regular expressions and their descendants do.
type preTokenizer func(text string) []string

// preTokenizerFor returns the pre-tokenizer that tokenizer.ggml.pre
// names, as llama.cpp maps the names, and whether a piece that is a token
// whole skips the merges, as Llama 3's tokenizer does.
func preTokenizerFor(name string) (pre preTokenizer, ignoreMerges bool, err error) {
	switch name {
	case "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2":
		return chain(splitLlama3), true, nil
	case "gpt-2", "phi-2", "jina-es", "jina-de", "gigachat", "jina-v2-es", "jina-v2-de", "mpt", "olmo", "jais", "trillion":
		return chain(splitGPT2), false, nil
	case "smollm", "starcoder", "refact", "command-r", "codeshell", "exaone", "minerva":
		return chain(isolate(digit), splitGPT2), false, nil
	case "", "default":
		return chain(isolate(punctuation), splitGPT2, isolate(digits), isolate(threeDigits)), false, nil
	}
	return nil, false, fmt.Errorf("%w: pre-tokenizer %q", errTokenizer, name)
}

var (
	digit       = regexp.MustCompile(`\p{N}`)
	digits      = regexp.MustCompile(`\p{N}+`)
	threeDigits = regexp.MustCompile(`[0-9][0-9][0-9]`)
	punctuation = regexp.MustCompile(`[\p{P}\$\+<=>\^~\|]+`)
)

// chain applies each split to every piece the previous one produced.
func chain(splits ...func(string) []string) preTokenizer {
	return func(text string) []string {
		pieces := []string{text}
		for _, split := range splits {
			var next []string
			for _, p := range pieces {
				next = append(next, split(p)...)
			}
			pieces = next
		}
		return pieces
	}
}

// isolate returns a split that makes each match of re, and each run of
// text between matches, a piece.
func isolate(re *regexp.Regexp) func(string) []string {
	return func(s string) []string {
		var out []string
		at := 0
		for _, m := range re.FindAllStringIndex(s, -1) {
			if m[0] > at {
				out = append(out, s[at:m[0]])
			}
			if m[1] > m[0] {
				out = append(out, s[m[0]:m[1]])
			}
			at = m[1]
		}
		if at < len(s) {
			out = append(out, s[at:])
		}
		return out
	}
}

// runes is a piece of text as code points, which reads as zero past its
// ends, as llama.cpp's splitters do.
type runes []rune

const outside = -1

func (r runes) at(i int) rune {
	if i < 0 || i >= len(r) {
		return outside
	}
	return r[i]
}

func (r runes) letter(i int) bool { c := r.at(i); return c != outside && unicode.IsLetter(c) }
func (r runes) number(i int) bool { c := r.at(i); return c != outside && unicode.IsNumber(c) }
func (r runes) space(i int) bool  { c := r.at(i); return c != outside && unicode.IsSpace(c) }

// assigned reports whether code point i is within the text and has a
// Unicode category.
func (r runes) assigned(i int) bool {
	c := r.at(i)
	return c != outside && unicode.In(c, unicode.L, unicode.M, unicode.N, unicode.P, unicode.S, unicode.Z, unicode.C)
}

// other reports whether code point i is assigned and neither a space, a
// letter nor a number: [^\s\p{L}\p{N}].
func (r runes) other(i int) bool { return r.assigned(i) && !r.space(i) && !r.letter(i) && !r.number(i) }

// contraction returns the length of an English contraction at i, such as
// 's or 'll, or 0, matching letters in either case if fold is set.
func (r runes) contraction(i int, fold bool) int {
	if r.at(i) != '\'' {
		return 0
	}
	lower := func(c rune) rune {
		if fold {
			return unicode.ToLower(c)
		}
		return c
	}
	switch a, b := lower(r.at(i+1)), lower(r.at(i+2)); {
	case a == 's' || a == 't' || a == 'm' || a == 'd':
		return 2
	case a == 'r' && b == 'e', a == 'v' && b == 'e', a == 'l' && b == 'l':
		return 3
	}
	return 0
}

// pieces collects the pieces of text ending at successive positions.
type pieces struct {
	text runes
	prev int
	out  []string
}

func (p *pieces) cut(end int) {
	if end > p.prev {
		p.out = append(p.out, string(p.text[p.prev:end]))
	}
	p.prev = end
}

// spaces returns the number of whitespace code points from i, and the
// position after the last line break among them, or 0.
func (r runes) spaces(i int) (n, afterBreak int) {
	for r.space(i + n) {
		if c := r.at(i + n); c == '\r' || c == '\n' {
			afterBreak = i + n + 1
		}
		n++
	}
	return n, afterBreak
}

// splitGPT2 splits text as GPT-2's expression does:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
func splitGPT2(text string) []string {
	r := runes(text)
	p := &pieces{text: r}
	for i := 0; i < len(r); {
		if n := r.contraction(i, false); n > 0 {
			i += n
			p.cut(i)
			continue
		}
		j := i
		if r.at(i) == ' ' {
			j++
		}
		switch {
		case r.letter(j):
			for r.letter(j) {
				j++
			}
		case r.number(j):
			for r.number(j) {
				j++
			}
		case r.other(j):
			for r.other(j) {
				j++
			}
		default:
			n, _ := r.spaces(i)
			switch {
			case n > 1 && r.at(i+n) != outside:
				j = i + n - 1 // \s+(?!\S): leave the last space to the next piece
			case n > 0:
				j = i + n
			default:
				j = i + 1
			}
		}
		i = j
		p.cut(i)
	}
	return p.out
}

// splitLlama3 splits text as Llama 3's expression does:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
func splitLlama3(text string) []string {
	r := runes(text)
	p := &pieces{text: r}
	for i := 0; i < len(r); {
		if n := r.contraction(i, true); n > 0 {
			i += n
			p.cut(i)
			continue
		}
		c := r.at(i)
		if c != '\r' && c != '\n' && !r.number(i) && (r.letter(i) || r.letter(i+1)) {
			i++
			for r.letter(i) {
				i++
			}
			p.cut(i)
			continue
		}
		if r.number(i) {
			start := i
			for r.number(i) {
				if i++; i-start >= 3 {
					p.cut(i)
					start = i
				}
			}
			p.cut(i)
			continue
		}
		j := i
		if c == ' ' {
			j++
		}
		if r.other(j) && r.assigned(i) {
			for r.other(j) {
				j++
			}
			for c := r.at(j); c == '\r' || c == '\n'; c = r.at(j) {
				j++
			}
			i = j
			p.cut(i)
			continue
		}
		n, afterBreak := r.spaces(i)
		switch {
		case afterBreak > 0:
			i = afterBreak // \s*[\r\n]+
		case n > 1 && r.at(i+n) != outside:
			i += n - 1
		case n > 0:
			i += n
		default:
			i++
		}
		p.cut(i)
	}
	return p.out
}
