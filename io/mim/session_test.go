package mim_test

import (
	"context"
	"encoding/base64"
	"path/filepath"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/tensor"
)

// Sessions of one model decode at once, each with its own key-value
// cache, and compute what one session alone does.
func TestConcurrentSessions(t *testing.T) {
	var e struct {
		Vocab        int      `json:"vocab"`
		PromptIDs    []int32  `json:"prompt_ids"`
		PromptLogits string   `json:"prompt_logits"`
		Greedy       []int32  `json:"greedy"`
		GreedyLogits []string `json:"greedy_logits"`
	}
	readJSON(t, filepath.Join("testdata", "llama", "expected.json"), &e)
	m := llama(t, "exact")
	defer m.Close()
	decode := func(s string) []float32 {
		b, err := base64.StdEncoding.DecodeString(s)
		if err != nil {
			t.Error(err)
		}
		return floats(b)
	}
	const sessions = 4
	var wg sync.WaitGroup
	for range sessions {
		wg.Go(func() {
			ctx := context.Background()
			s := m.NewSession()
			step := func(ids []int32, pos int) []float32 {
				p := make([]int32, len(ids))
				for i := range p {
					p[i] = int32(pos + i)
				}
				out, err := s.Run(ctx, map[string]*tensor.Tensor{
					"tokens":    tensor.FromSlice(ids, 1, len(ids)),
					"positions": tensor.FromSlice(p, 1, len(ids)),
				})
				if err != nil {
					t.Error(err)
					return nil
				}
				l, err := out["logits"].Data[float32](ctx)
				if err != nil {
					t.Error(err)
					return nil
				}
				return l
			}
			all := step(e.PromptIDs, 0)
			if all == nil {
				return
			}
			near(t, "prompt", all, decode(e.PromptLogits), 2e-5)
			logits := all[len(all)-e.Vocab:]
			pos := len(e.PromptIDs)
			for i, want := range e.Greedy {
				near(t, "step", logits, decode(e.GreedyLogits[i]), 2e-5)
				if got := argmax(logits); got != want {
					t.Errorf("step %d: token %d, want %d", i, got, want)
					return
				}
				if logits = step([]int32{want}, pos); logits == nil {
					return
				}
				pos++
			}
		})
	}
	wg.Wait()
}

// A run records no tape, even from inputs that require gradients.
func TestRunIsInference(t *testing.T) {
	var e struct {
		Models []struct {
			MIM  string    `json:"mim"`
			Runs []testRun `json:"runs"`
		} `json:"models"`
	}
	readJSON(t, filepath.Join("testdata", "pytorch", "expected.json"), &e)
	ctx := context.Background()
	tried := false
	for _, model := range e.Models {
		r := model.Runs[0]
		in := map[string]*tensor.Tensor{}
		float := false
		for k, a := range r.Inputs {
			x := a.tensor(t)
			if x.DType().IsFloat() {
				x, float = x.RequiresGrad(), true
			}
			in[k] = x
		}
		if !float {
			continue
		}
		m, err := mim.Open(filepath.Join("testdata", "pytorch", model.MIM))
		if err != nil {
			t.Fatal(err)
		}
		out, err := m.NewSession().Run(ctx, in)
		m.Close()
		if err != nil {
			t.Fatalf("%s: %v", model.MIM, err)
		}
		for k, y := range out {
			if y.Tracked() {
				t.Errorf("%s: output %s is tracked", model.MIM, k)
			}
		}
		tried = true
	}
	if !tried {
		t.Fatal("no model with a float input")
	}
}
