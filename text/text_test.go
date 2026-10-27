package text_test

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/nn/loss"
	"github.com/gsoultan/mimir/optim"
	"github.com/gsoultan/mimir/tensor"
	"github.com/gsoultan/mimir/text"
)

func TestWordPiece(t *testing.T) {
	// The example of BERT's own tokenizer tests.
	tok := text.NewTokenizer([]string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing", ","})
	got := tok.Tokens("UNwantéd,running")
	if want := []string{"un", "##want", "##ed", ",", "runn", "##ing"}; !slices.Equal(got, want) {
		t.Errorf("tokens %q, want %q", got, want)
	}
	if got := tok.Tokens("unwantedX running"); !slices.Equal(got, []string{"[UNK]", "runn", "##ing"}) {
		t.Errorf("tokens %q", got)
	}
	ids := tok.Encode("want running", 8)
	want := []int32{2, 4, 9, 10, 3, 0, 0, 0}
	if !slices.Equal(ids, want) {
		t.Errorf("ids %v, want %v", ids, want)
	}
	if ids := tok.Encode("want want want want", 4); len(ids) != 4 || ids[0] != 2 || ids[3] != 3 {
		t.Errorf("cut ids %v", ids)
	}
	if got := tok.Tokens("hello, 世界!"); len(got) != 5 || got[1] != "," {
		t.Errorf("punctuation and CJK %q", got)
	}
}

func TestBuildVocabReadsEveryWord(t *testing.T) {
	texts := []string{"Where is my parcel?", "My parcel arrived broken.", "Refund my order please", "Café crème"}
	vocab := text.BuildVocab(texts, 60)
	if len(vocab) > 60 || vocab[0] != text.Pad {
		t.Fatalf("vocab of %d starting %q", len(vocab), vocab[0])
	}
	tok := text.NewTokenizer(vocab)
	for _, s := range append(texts, "parcels refunded") {
		if slices.Contains(tok.Tokens(s), text.Unk) {
			t.Errorf("%q reads as %q", s, tok.Tokens(s))
		}
	}
	if got := tok.Tokens("café"); len(got) == 0 || got[0][0] != 'c' || slices.Contains(got, "é") {
		t.Errorf("accents kept: %q", got)
	}
}

// BERT-mini has the parameters of Turc et al.'s, 11.17 million with
// BERT's vocabulary.
func TestBERTParameters(t *testing.T) {
	m, err := text.NewBERT(text.BERTMini, 30522, 2)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, p := range nn.Parameters(m) {
		n += p.Value().Size()
	}
	if want := 11_170_560 + 256*2 + 2; n != want {
		t.Errorf("%d parameters, want %d", n, want)
	}
}

func TestBERTLearnsAndIgnoresPadding(t *testing.T) {
	ctx := context.Background()
	var texts []string
	var labels []int32
	for i := range 32 {
		topic := []string{"refund", "shipping"}[i%2]
		texts = append(texts, fmt.Sprintf("hello the %s for order %d is %s", topic, i, []string{"late", "here", "wrong"}[i%3]))
		labels = append(labels, int32(i%2))
	}
	tok := text.NewTokenizer(text.BuildVocab(texts, 200))
	cfg := text.BERTTiny
	cfg.Dropout = 0
	m, err := text.NewBERT(cfg, tok.Size(), 2)
	if err != nil {
		t.Fatal(err)
	}
	const length = 16
	ids := make([]int32, 0, len(texts)*length)
	for _, s := range texts {
		ids = append(ids, tok.Encode(s, length)...)
	}
	x, y := tensor.FromSlice(ids, len(texts), length), tensor.FromSlice(labels, len(labels))
	opt := optim.AdamW(nn.Parameters(m), optim.LR(3e-3))
	var first, last float32
	for i := range 30 {
		l := loss.CrossEntropy(m.Forward(x), y)
		if err := l.Backward(ctx); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(ctx); err != nil {
			t.Fatal(err)
		}
		v, _ := l.Data[float32](ctx)
		if i == 0 {
			first = v[0]
		}
		last = v[0]
		for _, p := range nn.Parameters(m) {
			p.ZeroGrad()
		}
	}
	if !(last < first*0.2) {
		t.Errorf("loss %v after 30 steps from %v", last, first)
	}
	// More padding doesn't change the answer.
	nn.Eval(m)
	short, _ := m.Forward(tensor.FromSlice(tok.Encode(texts[0], length), 1, length)).Data[float32](ctx)
	long, _ := m.Forward(tensor.FromSlice(tok.Encode(texts[0], 40), 1, 40)).Data[float32](ctx)
	for i := range short {
		if math.Abs(float64(short[i]-long[i])) > 1e-4 {
			t.Errorf("logits %v padded to 16, %v padded to 40", short, long)
			break
		}
	}
	// A float id tensor reads as ids too.
	if err := m.Forward(tensor.FromSlice([]float32{2, 5, 3, 0}, 1, 4)).Err(); err != nil {
		t.Error(err)
	}
}
