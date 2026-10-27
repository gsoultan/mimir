package mim_test

import (
	"bytes"
	"errors"
	"testing"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/io/mim"
)

// adapter writes an adapter file with a pair of rank 1 for each target,
// of the shape [out, in] given with it, and opens it.
func adapter(t *testing.T, md mimfile.Metadata, targets map[string][2]uint64) *mim.Model {
	t.Helper()
	w := mimfile.NewWriter()
	for name, s := range targets {
		if err := w.AddTensor(name+".lora_a", mimfile.F32, []uint64{1, s[1]}, make([]byte, 4*s[1])); err != nil {
			t.Fatal(err)
		}
		if err := w.AddTensor(name+".lora_b", mimfile.F32, []uint64{s[0], 1}, make([]byte, 4*s[0])); err != nil {
			t.Fatal(err)
		}
	}
	mb, err := md.Encode()
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddSection(mimfile.SectionMetadata, false, mb); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := w.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	a, err := mim.Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { a.Close() })
	return a
}

// TestApplyLoRARejects checks the adapters ApplyLoRA refuses. Applying
// one, and its effect on the logits, is tested in package llm.
func TestApplyLoRARejects(t *testing.T) {
	m := llama(t, "exact")
	defer m.Close()
	q, err := m.Tensor("blk.0.attn_q.weight")
	if err != nil {
		t.Fatal(err)
	}
	qs := [2]uint64{uint64(q.Shape()[0]), uint64(q.Shape()[1])}
	e, err := m.Tensor("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	es := [2]uint64{uint64(e.Shape()[0]), uint64(e.Shape()[1])}
	lora := mimfile.LoRAMetadata(2, "")
	for _, tc := range []struct {
		name        string
		a           *mim.Model
		unsupported bool
	}{
		{"not an adapter", adapter(t, mimfile.Metadata{"general.name": "x"}, map[string][2]uint64{"blk.0.attn_q.weight": qs}), false},
		{"no such weight", adapter(t, lora, map[string][2]uint64{"blk.9.attn_q.weight": qs}), false},
		{"another shape", adapter(t, lora, map[string][2]uint64{"blk.0.attn_q.weight": {qs[0] + 1, qs[1]}}), false},
		{"read by embedding", adapter(t, lora, map[string][2]uint64{"token_embd.weight": es}), true},
	} {
		err := m.ApplyLoRA(tc.a, "")
		var u *mim.UnsupportedError
		switch {
		case tc.unsupported && !errors.As(err, &u):
			t.Errorf("%s: got %v, want an *UnsupportedError", tc.name, err)
		case !tc.unsupported && !errors.Is(err, mim.ErrFormat):
			t.Errorf("%s: got %v, want ErrFormat", tc.name, err)
		}
	}
	if err := m.ApplyLoRA(m, ""); err == nil {
		t.Error("model applied to itself")
	}
	// None of them changed the model.
	if err := m.ApplyLoRA(adapter(t, lora, map[string][2]uint64{"blk.0.attn_q.weight": qs}), ""); err != nil {
		t.Errorf("good adapter after the bad ones: %v", err)
	}
}
