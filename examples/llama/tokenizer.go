package main

import (
	"container/heap"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	mimfile "github.com/gsoultan/mim"
)

// Token types, as tokenizer.ggml.token_type stores them.
const (
	typeNormal      = 1
	typeUnknown     = 2
	typeControl     = 3
	typeUserDefined = 4
	typeUnused      = 5
	typeByte        = 6
)

// Tokenizer turns text into tokens and back as llama.cpp does for the
// vocabulary that a GGUF file holds and its conversion to a Mimir Model
// file keeps in its metadata: SentencePiece's byte-fallback BPE for
// tokenizer model "llama", as Llama 2 and TinyLlama use, and GPT-2's
// byte-level BPE for "gpt2", as Llama 3 and SmolLM use, with the
// pre-tokenizer tokenizer.ggml.pre names.
type Tokenizer struct {
	spm    bool
	tokens []string
	types  []int32
	scores []float32 // SentencePiece's piece scores
	ids    map[string]int32
	ranks  map[string]int // BPE: "left right" → merge rank
	pre    preTokenizer
	// ignoreMerges takes a pre-token that is in the vocabulary whole as
	// one token, as Llama 3's tokenizer does.
	ignoreMerges bool

	specials       []int32 // control, user-defined and unknown tokens, longest first
	bytes          [256]int32
	unk            int32
	bos, eos       int32 // -1 if none
	eog            map[int32]bool
	addBOS, addEOS bool
	addSpacePrefix bool // SentencePiece: prefix text with a space
}

// maxTokenLen bounds the text of a token.
const maxTokenLen = 1 << 10

// errTokenizer reports a tokenizer this program does not read.
var errTokenizer = errors.New("llama: unsupported tokenizer")

// NewTokenizer returns the tokenizer of the vocabulary that the metadata
// md holds under the keys of GGUF's "tokenizer.ggml." namespace.
func NewTokenizer(md mimfile.Metadata) (*Tokenizer, error) {
	model, _ := md["tokenizer.ggml.model"].(string)
	tokens, ok := md["tokenizer.ggml.tokens"].([]string)
	if !ok || len(tokens) == 0 {
		return nil, fmt.Errorf("%w: no tokenizer.ggml.tokens", errTokenizer)
	}
	t := &Tokenizer{tokens: tokens, ids: make(map[string]int32, len(tokens)), bos: -1, eos: -1, unk: -1, eog: map[int32]bool{}}
	if t.types, ok = int32s(md["tokenizer.ggml.token_type"]); !ok {
		t.types = make([]int32, len(tokens))
		for i := range t.types {
			t.types[i] = typeNormal
		}
	}
	if len(t.types) != len(tokens) {
		return nil, fmt.Errorf("%w: %d token types for %d tokens", errTokenizer, len(t.types), len(tokens))
	}
	for i, s := range tokens {
		if len(s) > maxTokenLen {
			return nil, fmt.Errorf("%w: token %d of %d bytes", errTokenizer, i, len(s))
		}
		if _, dup := t.ids[s]; !dup {
			t.ids[s] = int32(i)
		}
	}
	switch model {
	case "llama":
		t.spm, t.addSpacePrefix, t.addBOS = true, true, true
		scores, _ := md["tokenizer.ggml.scores"].([]float64)
		if len(scores) != len(tokens) {
			return nil, fmt.Errorf("%w: SentencePiece vocabulary without a score for each token", errTokenizer)
		}
		t.scores = make([]float32, len(scores))
		for i, s := range scores {
			t.scores[i] = float32(s)
		}
		for b := range 256 {
			t.bytes[b] = t.lookup(fmt.Sprintf("<0x%02X>", b))
		}
		t.unk = 0
	case "gpt2":
		merges, _ := md["tokenizer.ggml.merges"].([]string)
		t.ranks = make(map[string]int, len(merges))
		for i, m := range merges {
			if _, dup := t.ranks[m]; !dup {
				t.ranks[m] = i
			}
		}
		pre, _ := md["tokenizer.ggml.pre"].(string)
		var err error
		if t.pre, t.ignoreMerges, err = preTokenizerFor(pre); err != nil {
			return nil, err
		}
		t.addBOS = t.ignoreMerges // Llama 3 adds BOS unless the file says otherwise
		for b := range 256 {
			t.bytes[b] = t.lookup(string(byteChars[b]))
		}
	default:
		return nil, fmt.Errorf("%w: tokenizer model %q", errTokenizer, model)
	}
	special := func(key string, dst *int32) error {
		v, ok := md[key]
		if !ok {
			return nil
		}
		id, ok := metaInt(md, key)
		if !ok || id < 0 || id >= int64(len(tokens)) {
			return fmt.Errorf("%w: %s = %v for %d tokens", errTokenizer, key, v, len(tokens))
		}
		*dst = int32(id)
		return nil
	}
	var eot, eom int32 = -1, -1
	if err := errors.Join(special("tokenizer.ggml.bos_token_id", &t.bos), special("tokenizer.ggml.eos_token_id", &t.eos),
		special("tokenizer.ggml.unknown_token_id", &t.unk), special("tokenizer.ggml.eot_token_id", &eot),
		special("tokenizer.ggml.eom_token_id", &eom)); err != nil {
		return nil, err
	}
	if b, ok := md["tokenizer.ggml.add_bos_token"].(bool); ok {
		t.addBOS = b
	}
	if b, ok := md["tokenizer.ggml.add_eos_token"].(bool); ok {
		t.addEOS = b
	}
	if b, ok := md["tokenizer.ggml.add_space_prefix"].(bool); ok && t.spm {
		t.addSpacePrefix = b
	}
	if t.addBOS && t.bos < 0 || t.addEOS && t.eos < 0 {
		return nil, fmt.Errorf("%w: adds a beginning or end token it does not name", errTokenizer)
	}
	for _, id := range []int32{t.eos, eot, eom} {
		if id >= 0 {
			t.eog[id] = true
		}
	}
	for i, s := range tokens {
		switch t.types[i] {
		case typeControl, typeUserDefined, typeUnknown:
			if s != "" {
				t.specials = append(t.specials, int32(i))
			}
		}
		if endsTurn[s] {
			t.eog[int32(i)] = true
		}
	}
	slices.SortStableFunc(t.specials, func(a, b int32) int { return len(tokens[b]) - len(tokens[a]) })
	return t, nil
}

// int32s returns v, an array of metadata integers, as int32s, and whether
// it is one whose every element fits.
func int32s(v any) ([]int32, bool) {
	ints, ok := v.([]int64)
	if !ok {
		return nil, false
	}
	out := make([]int32, len(ints))
	for i, x := range ints {
		if x != int64(int32(x)) {
			return nil, false
		}
		out[i] = int32(x)
	}
	return out, true
}

// endsTurn holds the texts of tokens that end a turn or a document in
// common vocabularies, which llama.cpp stops generating at.
var endsTurn = map[string]bool{
	"<|eot_id|>": true, "<|eom_id|>": true, "<|im_end|>": true, "<|end|>": true, "<|endoftext|>": true,
	"<|end_of_text|>": true, "<end_of_turn>": true, "<|end_of_turn|>": true, "<EOT>": true,
}

// lookup returns the token of text s, or -1.
func (t *Tokenizer) lookup(s string) int32 {
	if id, ok := t.ids[s]; ok {
		return id
	}
	return -1
}

// Len returns the number of tokens of the vocabulary.
func (t *Tokenizer) Len() int { return len(t.tokens) }

// BOS returns the token that begins a sequence, or -1.
func (t *Tokenizer) BOS() int32 { return t.bos }

// IsEOG reports whether id ends generation: the end of a sequence or of a
// turn.
func (t *Tokenizer) IsEOG(id int32) bool { return t.eog[id] }

// Encode returns the tokens of text, with the tokens the vocabulary adds
// at the start and end if addSpecial is set. The texts of control and
// user-defined tokens in text, such as "<|eot_id|>", become those tokens.
func (t *Tokenizer) Encode(text string, addSpecial bool) []int32 {
	var out []int32
	if addSpecial && t.addBOS {
		out = append(out, t.bos)
	}
	afterSpecial := true // SentencePiece prefixes the first text, and text after a special token, with a space
	for _, f := range t.partition(text) {
		if f.token >= 0 {
			out = append(out, f.token)
			afterSpecial = true
			continue
		}
		if t.spm {
			s := f.text
			if t.addSpacePrefix && afterSpecial {
				s = " " + s
			}
			out = t.encodeSPM(strings.ReplaceAll(s, " ", "▁"), out)
		} else {
			for _, w := range t.pre(f.text) {
				out = t.encodeBPE(w, out)
			}
		}
		afterSpecial = false
	}
	if addSpecial && t.addEOS {
		out = append(out, t.eos)
	}
	return out
}

// fragment is either text or a special token.
type fragment struct {
	text  string
	token int32 // -1 for text
}

// partition splits text at the texts of special tokens, longest first.
func (t *Tokenizer) partition(text string) []fragment {
	frags := []fragment{{text: text, token: -1}}
	if text == "" {
		return nil
	}
	for _, id := range t.specials {
		s := t.tokens[id]
		var next []fragment
		for _, f := range frags {
			if f.token >= 0 || !strings.Contains(f.text, s) {
				next = append(next, f)
				continue
			}
			rest := f.text
			for {
				i := strings.Index(rest, s)
				if i < 0 {
					break
				}
				if i > 0 {
					next = append(next, fragment{text: rest[:i], token: -1})
				}
				next = append(next, fragment{token: id})
				rest = rest[i+len(s):]
			}
			if rest != "" {
				next = append(next, fragment{text: rest, token: -1})
			}
		}
		frags = next
	}
	return frags
}

// symbol is a piece of the text being merged, linked to its neighbors.
type symbol struct {
	text       string
	prev, next int
}

// bigram is a candidate merge of two adjacent symbols.
type bigram struct {
	left, right int
	score       float64 // higher merges first
	text        string
}

// bigrams is a max-heap of bigrams by score, then leftmost first.
type bigrams []bigram

func (q bigrams) Len() int { return len(q) }
func (q bigrams) Less(i, j int) bool {
	if q[i].score != q[j].score {
		return q[i].score > q[j].score
	}
	return q[i].left < q[j].left
}
func (q bigrams) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *bigrams) Push(x any)   { *q = append(*q, x.(bigram)) }
func (q *bigrams) Pop() any {
	old := *q
	b := old[len(old)-1]
	*q = old[:len(old)-1]
	return b
}

// merger merges the characters of a word into symbols pair by pair, the
// best pair first, as both of llama.cpp's BPE tokenizers do.
type merger struct {
	syms  []symbol
	queue bigrams
	// score returns the priority of merging texts left and right, and
	// whether they merge at all.
	score func(left, right string) (float64, bool)
}

// merge returns the symbols of word once no pair merges any more.
func (m *merger) merge(word string) []symbol {
	m.syms, m.queue = m.syms[:0], m.queue[:0]
	for i, w := 0, 0; i < len(word); i += w {
		_, w = utf8.DecodeRuneInString(word[i:])
		m.syms = append(m.syms, symbol{text: word[i : i+w], prev: len(m.syms) - 1, next: len(m.syms) + 1})
	}
	if n := len(m.syms); n > 0 {
		m.syms[n-1].next = -1
	}
	for i := 1; i < len(m.syms); i++ {
		m.add(i-1, i)
	}
	for m.queue.Len() > 0 {
		b := heap.Pop(&m.queue).(bigram)
		l, r := &m.syms[b.left], &m.syms[b.right]
		if l.text == "" || r.text == "" || l.text+r.text != b.text {
			continue // a symbol of the pair merged since
		}
		l.text += r.text
		r.text = ""
		l.next = r.next
		if r.next >= 0 {
			m.syms[r.next].prev = b.left
		}
		m.add(l.prev, b.left)
		m.add(b.left, l.next)
	}
	return m.syms
}

func (m *merger) add(left, right int) {
	if left < 0 || right < 0 {
		return
	}
	l, r := m.syms[left].text, m.syms[right].text
	if s, ok := m.score(l, r); ok {
		heap.Push(&m.queue, bigram{left: left, right: right, score: s, text: l + r})
	}
}

// encodeSPM appends the SentencePiece tokens of text, whose spaces are
// already "▁", to out. Pairs merge if their text is a token, the token of
// the highest score first. Every merged symbol is a token, so a symbol
// that is not is a character outside the vocabulary, which becomes its
// bytes' tokens.
func (t *Tokenizer) encodeSPM(text string, out []int32) []int32 {
	m := &merger{score: func(l, r string) (float64, bool) {
		id, ok := t.ids[l+r]
		if !ok {
			return 0, false
		}
		return float64(t.scores[id]), true
	}}
	syms := m.merge(text)
	for i := 0; i >= 0 && i < len(syms); i = syms[i].next {
		s := syms[i].text
		if id, ok := t.ids[s]; ok {
			out = append(out, id)
			continue
		}
		for j := range len(s) {
			if id := t.bytes[s[j]]; id >= 0 {
				out = append(out, id)
			} else if t.unk >= 0 {
				out = append(out, t.unk)
			}
		}
	}
	return out
}

// encodeBPE appends the tokens of the pre-token word to out. Its bytes
// map to GPT-2's characters, and pairs merge by the merges' ranks, the
// lowest first.
func (t *Tokenizer) encodeBPE(word string, out []int32) []int32 {
	var b strings.Builder
	for i := range len(word) {
		b.WriteRune(byteChars[word[i]])
	}
	enc := b.String()
	if t.ignoreMerges {
		if id, ok := t.ids[enc]; ok {
			return append(out, id)
		}
	}
	m := &merger{score: func(l, r string) (float64, bool) {
		rank, ok := t.ranks[l+" "+r]
		return -float64(rank), ok
	}}
	syms := m.merge(enc)
	for i := 0; i >= 0 && i < len(syms); i = syms[i].next {
		s := syms[i].text
		if id, ok := t.ids[s]; ok {
			out = append(out, id)
			continue
		}
		for j := range len(s) {
			if id, ok := t.ids[s[j:j+1]]; ok {
				out = append(out, id)
			}
		}
	}
	return out
}

// Piece returns the bytes that token id stands for in text: nothing for a
// control token, and a byte token's byte.
func (t *Tokenizer) Piece(id int32) []byte {
	if id < 0 || int(id) >= len(t.tokens) {
		return nil
	}
	s := t.tokens[id]
	switch t.types[id] {
	case typeControl, typeUnused:
		return nil
	case typeUserDefined:
		return []byte(s)
	case typeByte:
		if v, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimPrefix(s, "<0x"), ">"), 16, 8); err == nil {
			return []byte{byte(v)}
		}
	}
	if t.spm {
		return []byte(strings.ReplaceAll(s, "▁", " "))
	}
	out := make([]byte, 0, len(s))
	for _, r := range s {
		if b, ok := charBytes[r]; ok {
			out = append(out, b)
		} else {
			out = utf8.AppendRune(out, r)
		}
	}
	return out
}

// Decode returns the text of tokens.
func (t *Tokenizer) Decode(tokens []int32) string {
	var b []byte
	for _, id := range tokens {
		b = append(b, t.Piece(id)...)
	}
	return string(b)
}

// byteChars maps each byte to the printable character GPT-2's byte-level
// BPE stands it for, and charBytes maps back.
var (
	byteChars [256]rune
	charBytes = map[rune]byte{}
)

func init() {
	n := 0
	for b := range 256 {
		switch {
		case b >= '!' && b <= '~', b >= 0xA1 && b <= 0xAC, b >= 0xAE:
			byteChars[b] = rune(b)
		default:
			byteChars[b] = rune(256 + n)
			n++
		}
		charBytes[byteChars[b]] = byte(b)
	}
}
