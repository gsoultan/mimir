package tensor_test

import (
	"context"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// productCounter counts the matrix products a backend runs.
type productCounter struct {
	*cpu.Backend
	mu       sync.Mutex
	products int
}

func (c *productCounter) Run(ctx context.Context, k *backend.Kernel, outs, ins []backend.Buffer) error {
	if k.Op == backend.OpMatMul {
		c.mu.Lock()
		c.products++
		c.mu.Unlock()
	}
	return c.Backend.Run(ctx, k, outs, ins)
}

// TestDecodingAStackComputesEachProductOnce decodes a token through a
// stack of blocks without computing each block's output before the next
// block reads it, as a model's decoding loop does. Each block must run
// its four products once: none for the blocks before it again.
func TestDecodingAStackComputesEachProductOnce(t *testing.T) {
	be := &productCounter{Backend: cpu.New()}
	defer tensor.SetDevice(be)()
	ctx := t.Context()
	const blocks = 3
	stack := make([]*nn.DecoderBlock, blocks)
	caches := make([]*nn.KVCache, blocks)
	for i := range stack {
		stack[i] = nn.NewGPT2Block(16, 4, nn.Seed(uint64(i+1)))
		caches[i] = nn.NewKVCache(1, 4, 8, 4, tensor.Float32)
	}
	x := tensor.Randn(1, 1, 3, 16)
	if err := x.Realize(ctx); err != nil {
		t.Fatal(err)
	}
	for _, b := range stack {
		if err := tensor.Realize(ctx, paramValues(b)...); err != nil {
			t.Fatal(err)
		}
	}
	be.products = 0
	for i, b := range stack {
		var err error
		if x, err = b.Decode(ctx, x, caches[i]); err != nil {
			t.Fatal(err)
		}
	}
	if err := x.Realize(ctx); err != nil {
		t.Fatal(err)
	}
	// The queries, keys and values come from one product; then the
	// attention's output projection and the MLP's two layers.
	if want := 4 * blocks; be.products != want {
		t.Errorf("decoding through %d blocks ran %d products, want %d", blocks, be.products, want)
	}
}

func paramValues(m nn.Module) []*tensor.Tensor {
	var vs []*tensor.Tensor
	for _, p := range nn.Parameters(m) {
		vs = append(vs, p.Value())
	}
	return vs
}
