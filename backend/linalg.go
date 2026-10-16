package backend

import (
	"fmt"
	"slices"
)

// FactorParts returns the shapes of the parts that a factorization op,
// OpLU, OpQR, OpSVD or OpEigh, writes in turn into its one-dimensional
// output for an input of shape [B..., m, n] (see the ops), and their total
// number of elements. OpLU and OpEigh take square matrices.
func FactorParts(op Op, shape []int) (parts [][]int, size int, err error) {
	r := len(shape)
	if r < 2 {
		return nil, 0, fmt.Errorf("%s of %v: %w", op, shape, ErrShape)
	}
	batch, m, n := shape[:r-2], shape[r-2], shape[r-1]
	at := func(dims ...int) []int { return slices.Concat(batch, dims) }
	k := min(m, n)
	switch op {
	case OpLU:
		parts = [][]int{at(n, n), at(n), at()}
	case OpQR:
		parts = [][]int{at(m, k), at(k, n)}
	case OpSVD:
		parts = [][]int{at(m, k), at(k), at(k, n)}
	case OpEigh:
		parts = [][]int{at(n), at(n, n)}
	default:
		return nil, 0, fmt.Errorf("%s is not a factorization: %w", op, ErrInternal)
	}
	if (op == OpLU || op == OpEigh) && m != n {
		return nil, 0, fmt.Errorf("%s of %v: not square: %w", op, shape, ErrShape)
	}
	for _, p := range parts {
		s, ok := NumElements(p)
		if !ok || size > maxInt-s {
			return nil, 0, fmt.Errorf("%s of %v: %w", op, shape, ErrShape)
		}
		size += s
	}
	return parts, size, nil
}
