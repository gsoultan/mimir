package tensor_test

import (
	"slices"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// FuzzOps applies byte-driven chains of operations with arbitrary
// arguments. Every chain must either fail with an error or produce exactly
// Size elements; nothing may panic.
func FuzzOps(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})
	f.Add([]byte{3, 200, 7, 7, 1, 0, 255, 9, 2, 2})
	f.Add([]byte{17, 1, 18, 255, 14, 2, 17, 0, 18, 3})
	f.Fuzz(func(t *testing.T, ops []byte) {
		if len(ops) > 64 {
			return
		}
		x := tensor.Randn(1, 2, 3, 4)
		arg := func(i int) int {
			if i < len(ops) {
				return int(int8(ops[i])) % 6
			}
			return 0
		}
		dtypes := []tensor.DType{tensor.Float32, tensor.Int32, tensor.BFloat16, tensor.Float16}
		for i, op := range ops {
			a, b := arg(i+1), arg(i+2)
			switch op % 19 {
			case 0:
				x = x.Reshape(a, -1)
			case 1:
				x = x.Transpose(a, b)
			case 2:
				x = x.Sum(a)
			case 3:
				x = x.Softmax(a)
			case 4:
				x = x.Unsqueeze(a)
			case 5:
				x = x.Squeeze()
			case 6:
				x = x.Flip(a)
			case 7:
				x = x.Add(tensor.Ones(tensor.Float32, a, b))
			case 8:
				x = x.MatMul(tensor.Ones(tensor.Float32, max(a, 0), max(b, 0)))
			case 9:
				pads := make([][2]int, x.Rank())
				for d := range pads {
					pads[d] = [2]int{a, b}
				}
				x = x.Pad(pads, 1)
			case 10:
				ranges := make([][2]int, x.Rank())
				for d, s := range x.Shape() {
					ranges[d] = [2]int{min(max(a, 0), s), s}
				}
				x = x.Shrink(ranges)
			case 11:
				x = x.Expand(slices.Concat([]int{max(a, 1)}, x.Shape())...)
			case 12:
				x = x.Gather(0, tensor.Zeros(tensor.Int32, x.Shape()...))
			case 13:
				x = x.Max(a).Log()
			case 14:
				x = x.Cast(dtypes[(a+6)%4])
			case 15:
				x = x.Mul(tensor.Ones(tensor.BFloat16, a, b)).Sigmoid()
			case 16:
				x = x.MatMul(tensor.Ones(tensor.Float16, max(a, 0), max(b, 0)))
			case 17:
				x = x.CumSum(a).LogCumSumExp(b)
			case 18:
				v, i := x.CumMax(a)
				x = v.Add(i.Cast(v.DType()))
			}
			if x.Size() > 1<<16 {
				return
			}
		}
		v, err := x.Cast(tensor.Float32).Data[float32](t.Context())
		if err == nil && len(v) != x.Size() {
			t.Fatalf("%d elements for shape %v", len(v), x.Shape())
		}
	})
}
