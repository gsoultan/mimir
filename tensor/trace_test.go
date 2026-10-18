package tensor_test

import (
	"context"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/tensor"
)

func TestTraceRecordsTheOperations(t *testing.T) {
	w := tensor.NewParam(tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 3, 2))
	f := func(x ...*tensor.Tensor) []*tensor.Tensor {
		return []*tensor.Tensor{x[0].MatMul(w.Value()).ReLU(), x[1]}
	}
	args := []*tensor.Tensor{tensor.Zeros(tensor.Float32, 4, 3), tensor.Zeros(tensor.Int32, 5)}
	tg, err := tensor.Trace(f, args, map[string]*tensor.Tensor{"w": w.Value()})
	if err != nil {
		t.Fatal(err)
	}
	if len(tg.Outputs) != 2 {
		t.Fatalf("%d outputs, want 2", len(tg.Outputs))
	}
	ops := map[backend.Op]int{}
	for i, n := range tg.Nodes {
		ops[n.Op]++
		for _, in := range n.Inputs {
			if in >= i {
				t.Fatalf("node %d (%s) reads node %d, which comes after it", i, n.Op, in)
			}
		}
		switch {
		case n.Arg >= 0:
			if n.Op != backend.OpInvalid || len(n.Inputs) > 0 || n.Value != nil {
				t.Errorf("argument %d: %+v", n.Arg, n)
			}
		case n.Op == backend.OpLoad:
			if n.Name != "w" || n.Value == nil || !slices.Equal(n.Shape, []int{3, 2}) {
				t.Errorf("weight: %+v", n)
			}
		}
	}
	if ops[backend.OpMatMul] != 1 || ops[backend.OpMax] != 1 || ops[backend.OpInvalid] != 2 {
		t.Errorf("operations %v", ops)
	}
	out := tg.Nodes[tg.Outputs[0]]
	if out.Op != backend.OpMax || !slices.Equal(out.Shape, []int{4, 2}) || out.Value != nil {
		t.Errorf("first result: %+v", out)
	}
	if second := tg.Nodes[tg.Outputs[1]]; second.Arg != 1 || second.DType != tensor.Int32 {
		t.Errorf("second result: %+v", second)
	}
}

func TestTraceGivesTheValuesNoArgumentAffects(t *testing.T) {
	f := func(x ...*tensor.Tensor) []*tensor.Tensor {
		table := tensor.FromSlice([]float32{1, 2}, 2).Mul(tensor.Full(tensor.Float32, 3, 2))
		return []*tensor.Tensor{x[0].Add(table)}
	}
	tg, err := tensor.Trace(f, []*tensor.Tensor{tensor.Zeros(tensor.Float32, 2)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	add := tg.Nodes[tg.Outputs[0]]
	table := tg.Nodes[add.Inputs[1]]
	if table.Op != backend.OpMul || table.Value == nil {
		t.Fatalf("table: %+v", table)
	}
	got, err := table.Value.Data[float32](context.Background())
	if err != nil || !slices.Equal(got, []float32{3, 6}) {
		t.Errorf("table holds %v, %v; want [3 6]", got, err)
	}
}

func TestTraceFailsOnTheValuesOfArguments(t *testing.T) {
	f := func(x ...*tensor.Tensor) []*tensor.Tensor {
		if _, err := x[0].Data[float32](context.Background()); err == nil {
			t.Error("the values of an argument read inside Trace")
		}
		return x
	}
	if _, err := tensor.Trace(f, []*tensor.Tensor{tensor.Zeros(tensor.Float32, 2)}, nil); err != nil {
		t.Fatal(err)
	}
}

func TestTraceReportsFailedResults(t *testing.T) {
	f := func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{x[0].MatMul(x[0])} }
	if _, err := tensor.Trace(f, []*tensor.Tensor{tensor.Zeros(tensor.Float32, 2, 3)}, nil); err == nil {
		t.Fatal("trace of a product of mismatched shapes succeeded")
	}
}

// TestTraceOnAnotherDevice traces a function that moves what it computes
// from the arguments, as a factorization does to run on the CPU.
func TestTraceOnAnotherDevice(t *testing.T) {
	other := tensor.DeviceOn(cpu.New())
	f := func(x ...*tensor.Tensor) []*tensor.Tensor {
		return []*tensor.Tensor{x[0].MatMul(x[0].Transpose(0, 1)).Cholesky()}
	}
	tg, err := tensor.Trace(f, []*tensor.Tensor{tensor.Zeros(tensor.Float32, 2, 2).To(other)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if out := tg.Nodes[tg.Outputs[0]]; out.Op != backend.OpCholesky || tg.Nodes[out.Inputs[0]].Op != backend.OpMatMul {
		t.Errorf("result: %+v", out)
	}
}
