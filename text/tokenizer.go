// Package text reads text with transformer encoders: a WordPiece
// tokenizer, as BERT's, and BERT itself, for classifying sentences or
// turning them into vectors.
//
//	tok := text.NewTokenizer(vocab)            // BERT's vocab.txt, one token a line
//	ids := tok.Encode("Where is my order?", 64) // [CLS] where is my order ? [SEP] [PAD]...
//	m := text.NewBERT(text.BERTTiny, tok.Size(), 3)
//	logits := m.Forward(tensor.FromSlice(ids, 1, 64))
//
// BuildVocab makes a vocabulary from texts, for a BERT taught from
// scratch; NewTokenizer reads a trained one, for a BERT whose weights
// came from elsewhere.
package text

import (
	"cmp"
	"maps"
	"slices"
	"strings"
	"unicode"
)

// The special tokens, as BERT names them.
const (
	Pad   = "[PAD]"
	Unk   = "[UNK]"
	Cls   = "[CLS]"
	Sep   = "[SEP]"
	Mask  = "[MASK]"
	piece = "##" // marks a word piece that continues a word
)

// Tokenizer splits text into WordPiece tokens and their ids, as BERT's
// uncased tokenizer does: lowercase, accents removed, punctuation and
// Chinese characters apart, then each word into the longest pieces the
// vocabulary has.
type Tokenizer struct {
	tokens []string
	ids    map[string]int32
	// Cased keeps case and accents, as a cased BERT's tokenizer does.
	Cased bool
	// MaxWord is the longest word, in runes, split into pieces; a longer
	// one is [UNK]. The default is 100.
	MaxWord int
}

// NewTokenizer returns a Tokenizer for vocab, each token's id its place.
// The special tokens it lacks are added at the end.
func NewTokenizer(vocab []string) *Tokenizer {
	t := &Tokenizer{ids: map[string]int32{}, MaxWord: 100}
	for _, v := range vocab {
		t.add(v)
	}
	for _, s := range []string{Pad, Unk, Cls, Sep, Mask} {
		t.add(s)
	}
	return t
}

func (t *Tokenizer) add(tok string) {
	if _, ok := t.ids[tok]; !ok {
		t.ids[tok] = int32(len(t.tokens))
		t.tokens = append(t.tokens, tok)
	}
}

// Size returns the number of tokens.
func (t *Tokenizer) Size() int { return len(t.tokens) }

// Vocab returns the tokens in id order, to save and read back with
// NewTokenizer.
func (t *Tokenizer) Vocab() []string { return slices.Clone(t.tokens) }

// ID returns tok's id, or [UNK]'s.
func (t *Tokenizer) ID(tok string) int32 {
	if id, ok := t.ids[tok]; ok {
		return id
	}
	return t.ids[Unk]
}

// Token returns the token of id, or [UNK] for one outside the
// vocabulary.
func (t *Tokenizer) Token(id int32) string {
	if id < 0 || int(id) >= len(t.tokens) {
		return Unk
	}
	return t.tokens[id]
}

// words splits s into words and punctuation, normalized as the tokenizer
// reads them.
func (t *Tokenizer) words(s string) []string {
	if !t.Cased {
		s = strings.ToLower(s)
		var b strings.Builder
		for _, r := range s {
			if unicode.Is(unicode.Mn, r) {
				continue // a combining accent
			}
			if base, ok := unaccented[r]; ok {
				r = base
			}
			b.WriteRune(r)
		}
		s = b.String()
	}
	var out []string
	var cur []rune
	flush := func() {
		if len(cur) > 0 {
			out = append(out, string(cur))
			cur = cur[:0]
		}
	}
	for _, r := range s {
		switch {
		case r == 0 || r == unicode.ReplacementChar || unicode.IsControl(r) && !unicode.IsSpace(r):
		case unicode.IsSpace(r):
			flush()
		case isPunct(r) || isCJK(r):
			flush()
			out = append(out, string(r))
		default:
			cur = append(cur, r)
		}
	}
	flush()
	return out
}

// isPunct reports whether r stands apart from words: Unicode punctuation,
// and the ASCII symbols BERT counts as punctuation too.
func isPunct(r rune) bool {
	return r >= 33 && r <= 47 || r >= 58 && r <= 64 || r >= 91 && r <= 96 || r >= 123 && r <= 126 || unicode.IsPunct(r)
}

func isCJK(r rune) bool {
	return r >= 0x4E00 && r <= 0x9FFF || r >= 0x3400 && r <= 0x4DBF || r >= 0x20000 && r <= 0x2A6DF ||
		r >= 0x2A700 && r <= 0x2CEAF || r >= 0xF900 && r <= 0xFAFF || r >= 0x2F800 && r <= 0x2FA1F
}

// Tokens returns s's WordPiece tokens.
func (t *Tokenizer) Tokens(s string) []string {
	var out []string
	for _, w := range t.words(s) {
		out = append(out, t.pieces(w)...)
	}
	return out
}

// pieces splits word into the longest pieces the vocabulary has, from
// the left, or returns [UNK] when it has none for some part.
func (t *Tokenizer) pieces(word string) []string {
	rs := []rune(word)
	if len(rs) > t.MaxWord {
		return []string{Unk}
	}
	var out []string
	for start := 0; start < len(rs); {
		end := len(rs)
		found := ""
		for ; end > start; end-- {
			sub := string(rs[start:end])
			if start > 0 {
				sub = piece + sub
			}
			if _, ok := t.ids[sub]; ok {
				found = sub
				break
			}
		}
		if found == "" {
			return []string{Unk}
		}
		out = append(out, found)
		start = end
	}
	return out
}

// Encode returns the ids of [CLS], s's tokens and [SEP], cut or padded
// with [PAD] to length, as one input row of a BERT.
func (t *Tokenizer) Encode(s string, length int) []int32 {
	ids := make([]int32, 0, length)
	ids = append(ids, t.ids[Cls])
	for _, tok := range t.Tokens(s) {
		if len(ids) >= length-1 {
			break
		}
		ids = append(ids, t.ids[tok])
	}
	ids = append(ids, t.ids[Sep])
	ids = ids[:min(len(ids), length)]
	for len(ids) < length {
		ids = append(ids, t.ids[Pad])
	}
	return ids
}

// PadID returns [PAD]'s id.
func (t *Tokenizer) PadID() int32 { return t.ids[Pad] }

// BuildVocab returns a vocabulary for texts of at most size tokens: the
// special tokens, every character the texts hold, alone and as a word
// piece, then the most common words and the most common word endings
// until size is reached. It makes a Tokenizer for a BERT taught from
// scratch on these texts, where every word is read whole or in pieces
// and none is [UNK].
func BuildVocab(texts []string, size int) []string {
	t := &Tokenizer{MaxWord: 100}
	words := map[string]int{}
	for _, s := range texts {
		for _, w := range t.words(s) {
			words[w]++
		}
	}
	vocab := []string{Pad, Unk, Cls, Sep, Mask}
	seen := map[string]bool{}
	for _, v := range vocab {
		seen[v] = true
	}
	add := func(v string) {
		if !seen[v] && len(vocab) < max(size, 5) {
			seen[v] = true
			vocab = append(vocab, v)
		}
	}
	// Every character, so no word is unreadable.
	chars := map[string]int{}
	for w, n := range words {
		for i, r := range []rune(w) {
			c := string(r)
			if i > 0 {
				c = piece + c
			}
			chars[c] += n
		}
	}
	for _, c := range byCount(chars) {
		add(c)
	}
	// Whole words and the endings of the words that aren't, most common
	// first, alternating so both kinds get room.
	endings := map[string]int{}
	for w, n := range words {
		rs := []rune(w)
		for i := 1; i < len(rs) && i <= 4; i++ {
			endings[piece+string(rs[len(rs)-i:])] += n
		}
	}
	ws, es := byCount(words), byCount(endings)
	for i := 0; len(vocab) < size && (i < len(ws) || i < len(es)); i++ {
		if i < len(ws) {
			add(ws[i])
		}
		if i < len(es) && i%2 == 0 {
			add(es[i/2])
		}
	}
	return vocab
}

// byCount returns count's keys, most common first.
func byCount(count map[string]int) []string {
	keys := slices.Collect(maps.Keys(count))
	slices.SortFunc(keys, func(a, b string) int { return cmp.Or(cmp.Compare(count[b], count[a]), cmp.Compare(a, b)) })
	return keys
}

// unaccented maps the accented Latin letters, lowercase, to their base
// letters, as BERT's uncased tokenizer strips accents.
var unaccented = map[rune]rune{
	'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'ç': 'c', 'è': 'e', 'é': 'e',
	'ê': 'e', 'ë': 'e', 'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i', 'ñ': 'n', 'ò': 'o', 'ó': 'o',
	'ô': 'o', 'õ': 'o', 'ö': 'o', 'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u', 'ý': 'y', 'ÿ': 'y',
	'ā': 'a', 'ă': 'a', 'ą': 'a', 'ć': 'c', 'ĉ': 'c', 'ċ': 'c', 'č': 'c', 'ď': 'd', 'ē': 'e',
	'ĕ': 'e', 'ė': 'e', 'ę': 'e', 'ě': 'e', 'ĝ': 'g', 'ğ': 'g', 'ġ': 'g', 'ģ': 'g', 'ĥ': 'h',
	'ĩ': 'i', 'ī': 'i', 'ĭ': 'i', 'į': 'i', 'ĵ': 'j', 'ķ': 'k', 'ĺ': 'l', 'ļ': 'l', 'ľ': 'l',
	'ń': 'n', 'ņ': 'n', 'ň': 'n', 'ō': 'o', 'ŏ': 'o', 'ő': 'o', 'ŕ': 'r', 'ŗ': 'r', 'ř': 'r',
	'ś': 's', 'ŝ': 's', 'ş': 's', 'š': 's', 'ţ': 't', 'ť': 't', 'ũ': 'u', 'ū': 'u', 'ŭ': 'u',
	'ů': 'u', 'ű': 'u', 'ų': 'u', 'ŵ': 'w', 'ŷ': 'y', 'ź': 'z', 'ż': 'z', 'ž': 'z', 'ơ': 'o',
	'ư': 'u', 'ǎ': 'a', 'ǐ': 'i', 'ǒ': 'o', 'ǔ': 'u', 'ǖ': 'u', 'ǘ': 'u', 'ǚ': 'u', 'ǜ': 'u',
	'ǟ': 'a', 'ǡ': 'a', 'ǧ': 'g', 'ǩ': 'k', 'ǫ': 'o', 'ǭ': 'o', 'ǰ': 'j', 'ǵ': 'g', 'ǹ': 'n',
	'ǻ': 'a', 'ȁ': 'a', 'ȃ': 'a', 'ȅ': 'e', 'ȇ': 'e', 'ȉ': 'i', 'ȋ': 'i', 'ȍ': 'o', 'ȏ': 'o',
	'ȑ': 'r', 'ȓ': 'r', 'ȕ': 'u', 'ȗ': 'u', 'ș': 's', 'ț': 't', 'ȟ': 'h', 'ȧ': 'a', 'ȩ': 'e',
	'ȫ': 'o', 'ȭ': 'o', 'ȯ': 'o', 'ȱ': 'o', 'ȳ': 'y',
}
