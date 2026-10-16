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
	// Workers take groups of four columns, whose dot products stream the
	// four together, and each group runs every row of x while its columns
	// are in cache.
	groups := (n + 3) / 4
	grain := max(1, elementGrain/(4*k*rows))
	err := b.parallelFor(ctx, groups, grain, func(lo, hi int) error {
		var d [4]float32
		for o, ox := range offX {
			oy := offY[o]
			for g := lo; g < hi; g++ {
				j := 4 * g
				for i := range m {
					xi := x.data[ox+i*x.rs:][:k]
					row := out[(o*m+i)*n:][:n]
					if j+4 <= n {
						dots4(&d, xi, y.data[oy+j*y.cs:], y.cs)
						copy(row[j:], d[:])
						continue
					}
					for c := j; c < n; c++ {
						dots4(&d, xi, y.data[oy+c*y.cs:], 0)
						row[c] = d[0]
					}
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

// dots4 sets d[r] to the dot product of x with the row of y at r·stride,
// for r < 4: y holds len(x) elements from each. A row's sum is the same
// whatever the other rows are, so stride 0 computes one row.
func dots4(d *[4]float32, x, y []float32, stride int) {
	k := len(x)
	n := vectorPart(k)
	if n == 0 {
		for r := range d {
			d[r] = dot(x, y[r*stride:])
		}
		return
	}
	_ = y[3*stride+k-1]
	rowVector.dot4(&x[0], &y[0], stride, n, &d[0])
	for r := range d {
		row := y[r*stride:][:k]
		for i := n; i < k; i++ {
			d[r] += x[i] * row[i]
		}
	}
}

// dot returns Σ a[i]·b[i] over the length of a, which b must have.
func dot(a, b []float32) float32 {
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
