package metal

import (
	"fmt"
	"math"
	"strings"

	"github.com/gsoultan/mimir/backend"
)

// A float32 product runs on Metal 4 tensor operations, on GPUs that run
// them in hardware, as a kernel the backend generates around the matmul2d
// operation of Metal Performance Primitives. Each threadgroup of
// tensorSIMDGroups SIMD groups computes one tensorTile × tensorTile tile
// of one result matrix, reading its operands where planGEMM found them.
//
// Unlike Metal Performance Shaders, the kernel runs inside the open
// compute encoder, so it neither ends it nor makes the next kernel open
// another.
const (
	tensorTile       = 64
	tensorSIMDGroups = 4
	simdWidth        = 32 // threads in a SIMD group on Apple GPUs
)

// tensorMinWork is the number of multiply-adds from which a float32
// product runs on tensor operations instead of the generated kernel.
const tensorMinWork = 1 << 15

// Kernel parameters of a tensor product, in order.
const (
	tpM = iota
	tpN
	tpK
	tpAOffset
	tpALd
	tpAStride
	tpBOffset
	tpBLd
	tpBStride
	tpResultSize
	tpCount
)

// planTensorGEMM returns the kernel that computes the product g with
// tensor operations, from inputs 0 and 1 of ins into output 0. Its
// parameters follow the ins inputs, as for every kernel.
func planTensorGEMM(g gemm, ins int) (*launch, error) {
	vals := [tpCount]int{
		tpM: g.m, tpN: g.n, tpK: g.k,
		tpAOffset: g.a.offset, tpALd: g.a.ld, tpAStride: g.a.stride,
		tpBOffset: g.b.offset, tpBLd: g.b.ld, tpBStride: g.b.stride,
		tpResultSize: g.resultSize,
	}
	params := make([]int32, tpCount)
	for i, v := range vals {
		if v < 0 || v > math.MaxInt32 {
			return nil, fmt.Errorf("value %d does not fit a 32-bit kernel parameter: %w", v, backend.ErrShape)
		}
		params[i] = int32(v)
	}
	tiles := ceilDiv(g.m, tensorTile) * ceilDiv(g.n, tensorTile)
	if g.batch > math.MaxInt32/tiles {
		return nil, fmt.Errorf("%d tiles of %d matrices: %w", tiles, g.batch, backend.ErrShape)
	}
	return &launch{
		src:     tensorSource(g.a.trans, g.b.trans, ins),
		groups:  g.batch * tiles,
		threads: tensorSIMDGroups * simdWidth,
		params:  params,
		metal4:  true,
	}, nil
}

func ceilDiv(a, b int) int { return (a + b - 1) / b }

// tensorSource returns the source of the tensor product kernel of ins
// inputs whose operands are stored transposed as transLeft and transRight
// say.
//
// A tensor's first extent is its innermost. An operand stored as the
// transpose of the matrix it stands for has its extents swapped, and
// matmul2d reads it transposed.
func tensorSource(transLeft, transRight bool, ins int) string {
	left := operand{name: "a", buf: "i0", off: tpAOffset, ld: tpALd, stride: tpAStride,
		extents: "kk, m", slice: "0, row"}
	if transLeft {
		left.extents, left.slice = "m, kk", "row, 0"
	}
	right := operand{name: "b", buf: "i1", off: tpBOffset, ld: tpBLd, stride: tpBStride,
		extents: "n, kk", slice: "col, 0"}
	if transRight {
		right.extents, right.slice = "kk, n", "0, col"
	}
	var s strings.Builder
	fmt.Fprintf(&s, `#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
typedef tensor<device float, dextents<int32_t, 2>, tensor_inline> fmatrix;
kernel void k(device float* o0 [[buffer(0)]], device float* i0 [[buffer(1)]], device float* i1 [[buffer(2)]], constant int* p [[buffer(%d)]], uint bid [[threadgroup_position_in_grid]]) {
	int m = p[%d], n = p[%d], kk = p[%d];
	int tilesN = (n + %[5]d) / %[6]d, tiles = tilesN * ((m + %[5]d) / %[6]d);
	int batch = int(bid) / tiles, tile = int(bid) %% tiles;
	int row = tile / tilesN * %[6]d, col = tile %% tilesN * %[6]d;
`, 1+ins, tpM, tpN, tpK, tensorTile-1, tensorTile)
	left.declare(&s)
	right.declare(&s)
	fmt.Fprintf(&s, `	fmatrix c(o0 + batch * p[%d], dextents<int32_t, 2>(n, m), array<int32_t, 2>({1, n}));
	auto ta = a.slice(%s);
	auto tb = b.slice(%s);
	auto tc = c.slice(col, row);
	constexpr auto desc = matmul2d_descriptor(%d, %d, static_cast<int>(dynamic_extent), %t, %t, false);
	matmul2d<desc, execution_simdgroups<%d>> op;
	op.run(ta, tb, tc);
}
`, tpResultSize, left.slice, right.slice, tensorTile, tensorTile, transLeft, transRight, tensorSIMDGroups)
	return s.String()
}

// operand is how the tensor product kernel declares one input matrix.
type operand struct {
	name, buf       string
	off, ld, stride int
	extents, slice  string
}

func (o operand) declare(s *strings.Builder) {
	fmt.Fprintf(s, "\tfmatrix %s(%s + p[%d] + batch * p[%d], dextents<int32_t, 2>(%s), array<int32_t, 2>({1, p[%d]}));\n",
		o.name, o.buf, o.off, o.stride, o.extents, o.ld)
}
