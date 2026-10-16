package main

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/io/gguf"
)

func openTokenizer(t *testing.T, path string) *Tokenizer {
	t.Helper()
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	tok, err := NewTokenizer(f)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// encodings are texts and the tokens llama.cpp encodes them to, with the
// beginning-of-text token and special tokens parsed, for the vocabularies
// of testdata/spm.gguf and testdata/bpe.gguf (Llama 3's pre-tokenizer).
var encodings = []struct {
	text     string
	spm, bpe []int32
}{
	{"Hello world", []int32{1, 305, 307}, []int32{285, 270, 266}},
	{" Hello world!", []int32{1, 259, 305, 307, 287}, []int32{285, 272, 266, 33}},
	{"Hello, world! What is this?", []int32{1, 305, 285, 307, 287, 324, 267, 313, 310, 312, 66}, []int32{285, 270, 44, 266, 33, 32, 87, 104, 283, 275, 277, 63}},
	{"the theater in the hall is what it is", []int32{1, 294, 294, 313, 296, 315, 294, 259, 267, 262, 301, 310, 322, 318, 310}, []int32{285, 116, 257, 258, 283, 261, 32, 260, 258, 32, 104, 97, 268, 275, 262, 104, 283, 32, 105, 116, 275}},
	{"  leading and trailing spaces  ", []int32{1, 259, 259, 259, 269, 260, 262, 270, 320, 297, 265, 270, 292, 268, 262, 264, 269, 320, 259, 266, 278, 262, 274, 319, 259, 259}, []int32{285, 32, 32, 108, 101, 97, 100, 260, 103, 259, 110, 100, 256, 114, 97, 105, 108, 260, 103, 32, 115, 112, 97, 99, 101, 115, 281}},
	{"line one\nline two\n\nthree", []int32{1, 259, 269, 295, 260, 259, 298, 260, 13, 269, 295, 260, 292, 272, 263, 13, 13, 261, 267, 268, 260, 260}, []int32{285, 108, 260, 101, 32, 284, 101, 10, 108, 260, 101, 256, 119, 111, 280, 116, 104, 114, 101, 101}},
	{"tabs\tand\nnewlines \n ", []int32{1, 292, 262, 279, 266, 12, 316, 270, 13, 265, 260, 272, 269, 295, 319, 259, 13, 259}, []int32{285, 116, 97, 98, 115, 9, 97, 110, 100, 10, 110, 101, 119, 108, 260, 101, 115, 32, 10, 32}},
	{"numbers 1 12 123 1234 12345", []int32{1, 259, 265, 271, 273, 279, 296, 266, 259, 289, 259, 289, 290, 259, 289, 290, 291, 259, 289, 290, 291, 55, 259, 289, 290, 291, 55, 56}, []int32{285, 110, 117, 109, 98, 261, 115, 32, 49, 32, 278, 32, 279, 32, 279, 52, 32, 279, 52, 53}},
	{"it's what they're doing, I'll see; we'd've", []int32{1, 318, 288, 266, 322, 294, 276, 288, 268, 260, 259, 270, 263, 320, 285, 259, 76, 288, 301, 259, 266, 260, 260, 62, 299, 260, 288, 270, 288, 281, 260}, []int32{285, 105, 116, 282, 262, 104, 283, 258, 121, 39, 114, 101, 32, 100, 111, 260, 103, 44, 32, 73, 39, 268, 32, 115, 101, 101, 59, 262, 101, 39, 100, 39, 118, 101}},
	{"CAPS'LL AND 'S 'Re", []int32{1, 259, 70, 68, 83, 86, 288, 79, 79, 259, 68, 81, 71, 259, 288, 86, 259, 288, 85, 260}, []int32{285, 67, 65, 80, 83, 39, 76, 76, 32, 65, 78, 68, 32, 39, 83, 32, 39, 82, 101}},
	{"Äpfel und Bäume — naïve café", []int32{1, 259, 198, 135, 278, 277, 260, 269, 259, 271, 265, 270, 259, 69, 198, 167, 271, 273, 260, 259, 229, 131, 151, 259, 265, 262, 198, 178, 281, 260, 259, 274, 262, 277, 198, 172}, []int32{285, 195, 132, 112, 102, 101, 108, 32, 117, 110, 100, 32, 66, 195, 164, 117, 109, 101, 32, 226, 128, 148, 32, 110, 97, 195, 175, 118, 101, 32, 99, 97, 102, 195, 169}},
	{"你好，世界", []int32{1, 259, 231, 192, 163, 232, 168, 192, 242, 191, 143, 231, 187, 153, 234, 152, 143}, []int32{285, 228, 189, 160, 229, 165, 189, 239, 188, 140, 228, 184, 150, 231, 149, 140}},
	{"emoji 🦙 llama", []int32{1, 259, 260, 273, 263, 109, 264, 259, 243, 162, 169, 156, 259, 301, 262, 273, 262}, []int32{285, 101, 109, 111, 106, 105, 32, 240, 159, 166, 153, 32, 268, 97, 109, 97}},
	{"<|user|>Hello<|end|> world", []int32{1, 327, 305, 328, 259, 307}, []int32{285, 60, 124, 117, 115, 261, 124, 62, 270, 60, 124, 101, 110, 100, 124, 62, 266}},
	{"<|eot_id|><|begin_of_text|>Hello", []int32{1, 259, 63, 127, 260, 263, 261, 98, 264, 270, 127, 65, 63, 127, 279, 260, 275, 295, 98, 263, 277, 98, 261, 260, 123, 261, 127, 65, 282, 260, 301, 263}, []int32{285, 287, 285, 270}},
	{"<s>the</s>", []int32{1, 1, 294, 2}, []int32{285, 60, 115, 62, 116, 257, 60, 47, 115, 62}},
	{"<unk>x", []int32{1, 0, 259, 123}, []int32{285, 60, 117, 110, 107, 62, 120}},
	{"x<|end|>", []int32{1, 259, 123, 328}, []int32{285, 120, 60, 124, 101, 110, 100, 124, 62}},
	{"   ", []int32{1, 259, 259, 259, 259}, []int32{285, 281, 32}},
	{"!!!...???", []int32{1, 259, 287, 287, 287, 286, 286, 286, 66, 66, 66}, []int32{285, 33, 33, 33, 46, 46, 46, 63, 63, 63}},
	{"a  b   c\r\n\r\nd", []int32{1, 297, 259, 259, 279, 259, 259, 259, 274, 16, 13, 16, 13, 270}, []int32{285, 97, 32, 32, 98, 281, 32, 99, 13, 10, 13, 10, 100}},
	{"", []int32{1}, []int32{285}},
}

// TestTokenizersMatchLlamaCpp checks the SentencePiece and byte-level BPE
// tokenizers against llama.cpp's encodings of the same texts with the
// same vocabularies, and that decoding gives the texts back.
func TestTokenizersMatchLlamaCpp(t *testing.T) {
	spm := openTokenizer(t, "testdata/spm.gguf")
	bpe := openTokenizer(t, "testdata/bpe.gguf")
	for _, e := range encodings {
		if got := spm.Encode(e.text, true); !slices.Equal(got, e.spm) {
			t.Errorf("spm %q:\n got %v\nwant %v", e.text, got, e.spm)
		}
		if got := bpe.Encode(e.text, true); !slices.Equal(got, e.bpe) {
			t.Errorf("bpe %q:\n got %v\nwant %v", e.text, got, e.bpe)
		}
		// Byte-level BPE decodes every text exactly, but for control
		// tokens, which decode to nothing; SentencePiece adds the prefix
		// space.
		if got := bpe.Decode(e.bpe[1:]); got != e.text && !strings.Contains(e.text, "<|eot_id|>") {
			t.Errorf("bpe decodes %v to %q, want %q", e.bpe[1:], got, e.text)
		}
		if !strings.Contains(e.text, "<") {
			if got := spm.Decode(e.spm[1:]); got != " "+e.text && e.text != "" {
				t.Errorf("spm decodes %v to %q, want %q", e.spm[1:], got, " "+e.text)
			}
		}
	}
	if !bpe.IsEOG(287) || !bpe.IsEOG(286) || bpe.IsEOG(285) || !spm.IsEOG(2) || !spm.IsEOG(328) {
		t.Error("end-of-generation tokens")
	}
}

// TestPreTokenizers checks the pre-tokenizers' pieces on texts that
// exercise each alternative of their expressions.
func TestPreTokenizers(t *testing.T) {
	for _, c := range []struct {
		name, text string
		want       []string
	}{
		{"llama-bpe", "Hello  world's 12345\n\n  x!!\n", []string{"Hello", " ", " world", "'s", " ", "123", "45", "\n\n", " ", " x", "!!\n"}},
		{"llama-bpe", "I'LL  ", []string{"I", "'LL", "  "}},
		{"gpt-2", "Hello  world's 12345\n\n  x!!", []string{"Hello", " ", " world", "'s", " 12345", "\n\n ", " x", "!!"}},
		{"gpt-2", "I'LL", []string{"I", "'", "LL"}},
		{"smollm", "a 123b", []string{"a", " ", "1", "2", "3", "b"}},
		{"default", "a+=1234", []string{"a", "+=", "123", "4"}},
	} {
		pre, _, err := preTokenizerFor(c.name)
		if err != nil {
			t.Fatal(err)
		}
		if got := pre(c.text); !slices.Equal(got, c.want) {
			t.Errorf("%s %q: %q, want %q", c.name, c.text, got, c.want)
		}
	}
	if _, _, err := preTokenizerFor("unknown"); err == nil {
		t.Error("an unknown pre-tokenizer is accepted")
	}
}

// TestLlamaCppVocabularies checks the tokenizers against the test
// vocabularies llama.cpp ships in its models directory and their
// expected encodings, ggml-vocab-<name>.gguf with .inp and .out files,
// when MIMIR_LLAMA_VOCABS names that directory.
func TestLlamaCppVocabularies(t *testing.T) {
	dir := os.Getenv("MIMIR_LLAMA_VOCABS")
	if dir == "" {
		t.Skip("MIMIR_LLAMA_VOCABS is not set")
	}
	for _, name := range []string{"llama-spm", "llama-bpe", "gpt-2", "phi-3", "starcoder", "refact", "mpt", "command-r"} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(dir, "ggml-vocab-"+name+".gguf")
			inp, err1 := os.ReadFile(path + ".inp")
			out, err2 := os.ReadFile(path + ".out")
			if err1 != nil || err2 != nil {
				t.Skip(err1, err2)
			}
			tok := openTokenizer(t, path)
			texts := strings.Split(strings.TrimSuffix(string(inp), "\n__ggml_vocab_test__\n"), "\n__ggml_vocab_test__\n")
			lines := strings.Split(strings.TrimSuffix(string(out), "\n"), "\n")
			if len(texts) != len(lines) {
				t.Fatalf("%d texts and %d encodings", len(texts), len(lines))
			}
			for i, text := range texts {
				var want []int32
				for _, f := range strings.Fields(lines[i]) {
					id, err := strconv.Atoi(f)
					if err != nil {
						t.Fatal(err)
					}
					want = append(want, int32(id))
				}
				// llama.cpp's test encodes without the special tokens.
				if got := tok.Encode(text, false); !slices.Equal(got, want) {
					t.Errorf("%q:\n got %v\nwant %v", text, got, want)
				}
			}
		})
	}
}
