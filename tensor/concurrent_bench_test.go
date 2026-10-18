package tensor_test

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/tensor"
)

// BenchmarkConcurrentMatMul measures the throughput of independent 128x128
// products, each goroutine with operands of its own, on 1 and 4
// goroutines. Each op is one product computed and downloaded; with
// independent graphs computing at once, 4 goroutines take less time per
// op than 1.
func BenchmarkConcurrentMatMul(b *testing.B) {
	const n = 128
	for _, g := range []int{1, 4} {
		b.Run(fmt.Sprintf("goroutines=%d", g), func(b *testing.B) {
			ctx := context.Background()
			as, bs := make([]*tensor.Tensor, g), make([]*tensor.Tensor, g)
			for i := range g {
				as[i], bs[i] = tensor.Randn(uint64(2*i+1), n, n), tensor.Randn(uint64(2*i+2), n, n)
				if err := tensor.Realize(ctx, as[i], bs[i]); err != nil {
					b.Fatal(err)
				}
			}
			b.ResetTimer()
			var wg sync.WaitGroup
			for i := range g {
				ops := b.N / g
				if i < b.N%g {
					ops++
				}
				wg.Go(func() {
					for range ops {
						if _, err := as[i].MatMul(bs[i]).Data[float32](ctx); err != nil {
							b.Error(err)
							return
						}
					}
				})
			}
			wg.Wait()
		})
	}
}
