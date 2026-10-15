package tensor_test

import (
	"fmt"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// realized returns a computed random tensor so benchmarks time only the
// operation under test.
func realized(b *testing.B, seed uint64, shape ...int) *tensor.Tensor {
	b.Helper()
	x := tensor.Randn(seed, shape...)
	if err := x.Realize(b.Context()); err != nil {
		b.Fatal(err)
	}
	return x
}

func run(b *testing.B, f func() *tensor.Tensor) {
	b.Helper()
	for b.Loop() {
		if err := f().Realize(b.Context()); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMul(b *testing.B) {
	for _, n := range []int{64, 256, 512} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			x, y := realized(b, 1, n, n), realized(b, 2, n, n)
			b.SetBytes(int64(3 * n * n * 4))
			run(b, func() *tensor.Tensor { return x.MatMul(y) })
			b.ReportMetric(2*float64(n*n*n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
	}
}

func BenchmarkAdd(b *testing.B) {
	const n = 1 << 22
	x, y := realized(b, 1, n), realized(b, 2, n)
	b.SetBytes(3 * n * 4)
	run(b, func() *tensor.Tensor { return x.Add(y) })
}

func BenchmarkAddBroadcast(b *testing.B) {
	x, y := realized(b, 1, 2048, 2048), realized(b, 2, 2048)
	b.SetBytes(2 * 2048 * 2048 * 4)
	run(b, func() *tensor.Tensor { return x.Add(y) })
}

func BenchmarkSum(b *testing.B) {
	x := realized(b, 1, 2048, 2048)
	b.SetBytes(2048 * 2048 * 4)
	run(b, func() *tensor.Tensor { return x.Sum(1) })
}

func BenchmarkSoftmax(b *testing.B) {
	x := realized(b, 1, 1024, 1024)
	b.SetBytes(1024 * 1024 * 4)
	run(b, func() *tensor.Tensor { return x.Softmax(-1) })
}

func BenchmarkRandn(b *testing.B) {
	const n = 1 << 22
	b.SetBytes(n * 4)
	run(b, func() *tensor.Tensor { return tensor.Randn(1, n) })
}

func BenchmarkMLPForward(b *testing.B) {
	x := realized(b, 1, 256, 784)
	w1, w2 := realized(b, 2, 784, 512), realized(b, 3, 512, 10)
	b1, b2 := realized(b, 4, 512), realized(b, 5, 10)
	run(b, func() *tensor.Tensor {
		return x.MatMul(w1).Add(b1).ReLU().MatMul(w2).Add(b2).LogSoftmax(-1)
	})
}
