package cpu

import "context"

// gemvRows is the most output rows of a product, over all its batches,
// that sgemv computes as dot products: the products of a decoding step,
// whose few rows make packing B for the GEMM cost as much as multiplying.
const gemvRows = 8

// sgemv computes the float32 product of matmul as dot products of the rows
// of the left operand with the columns of the right, when the product has
// at most gemvRows rows, sums no batches and both operands are read along
// contiguous rows of their memory, as a Linear layer reads its transposed
// weight: each worker streams the columns of a block once. It reports
// whether it computed the product. Each element is one dot product
// whatever the number of workers.
func (b *Backend) sgemv(ctx context.Context, out []float32, x operand, offX []int, y operand, offY []int, m, n, k, sums int, epi *evaluator) (bool, error) {
	rows := len(out) / n
	if rows > gemvRows || sums != 1 || x.cs != 1 || y.rs != 1 || k < 16 {
		return false, nil
	}
	grain := max(1, elementGrain/(k*rows))
	err := b.parallelFor(ctx, n, grain, func(lo, hi int) error {
		for o, ox := range offX {
			oy := offY[o]
			for j := lo; j < hi; j++ {
				col := y.data[oy+j*y.cs:][:k]
				for i := range m {
					out[(o*m+i)*n+j] = dot(x.data[ox+i*x.rs:][:k], col)
				}
			}
		}
		return nil
	})
	if err != nil {
		return true, err
	}
	return true, epi.apply(0, len(out))
}

// dot returns Σ a[i]·b[i] over the length of a, which b must have.
func dot(a, b []float32) float32 {
	if rowVector != nil && len(a) >= 16 {
		n := vectorPart(len(a))
		s := rowVector.dot(&a[0], &b[0], n)
		for i := n; i < len(a); i++ {
			s += a[i] * b[i]
		}
		return s
	}
	b = b[:len(a)]
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i+8 <= len(a); i += 8 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
		s4 += a[i+4] * b[i+4]
		s5 += a[i+5] * b[i+5]
		s6 += a[i+6] * b[i+6]
		s7 += a[i+7] * b[i+7]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
}
