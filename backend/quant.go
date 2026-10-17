package backend

// IsQuantized reports whether d is a block-quantized type.
func (d DType) IsQuantized() bool { return d >= Q8_0 && d <= Q6_K }

// BlockSize returns the number of elements in a block of d: 32 for Q8_0,
// 256 for Q4_K and Q6_K, and 1 for the other types, whose blocks are
// single elements. A tensor of a block-quantized type holds whole blocks
// along its last dimension.
func (d DType) BlockSize() int {
	switch d {
	case Q8_0:
		return 32
	case Q4_K, Q6_K:
		return 256
	default:
		return 1
	}
}

// BlockBytes returns the size of a block of d in bytes: 34 for Q8_0, 144
// for Q4_K, 210 for Q6_K and Size for the other types, 0 for Invalid.
func (d DType) BlockBytes() int {
	switch d {
	case Q8_0:
		return 34
	case Q4_K:
		return 144
	case Q6_K:
		return 210
	default:
		return d.Size()
	}
}

// Bytes returns the size in bytes of n elements of d. It reports false if
// d has no size, n is negative or not a whole number of blocks, or the
// size overflows int.
func (d DType) Bytes(n int) (int, bool) {
	bs, bb := d.BlockSize(), d.BlockBytes()
	if bb == 0 || n < 0 || n%bs != 0 || n/bs > maxInt/bb {
		return 0, false
	}
	return n / bs * bb, true
}
