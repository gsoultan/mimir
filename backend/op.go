package backend

import "fmt"

// Op is a primitive operation. Every higher-level operation in mimir is
// composed from these, so autograd, scheduling and every backend only need
// to handle this set.
type Op uint8

// Primitive operations. Unless noted, binary operations require inputs of
// identical shape and dtype; broadcasting is expressed with Expand.
const (
	OpInvalid Op = iota

	// Sources.
	OpLoad        // data uploaded from the host; never run as a kernel
	OpConst       // every element is Attrs.Value
	OpRandUniform // uniform [0, 1) from Philox4x32-10 keyed by Attrs.Seed; Float32
	OpRandNormal  // standard normal via Box-Muller over Philox4x32-10; Float32

	// Unary.
	OpNeg
	OpExp2  // Float32 only
	OpLog2  // Float32 only
	OpSin   // Float32 only
	OpSqrt  // Float32 only
	OpRecip // 1/x; Float32 only
	OpCast  // convert to Kernel.DType; a block-quantized input is dequantized

	// Binary.
	OpAdd
	OpMul
	OpMax   // elementwise maximum; NaN propagates
	OpCmpLT // a < b as 1 or 0 in the input dtype
	OpCmpEQ // a == b as 1 or 0 in the input dtype
	OpMod   // truncated remainder with the sign of a, like C fmod

	// Ternary.
	OpWhere // In[0] != 0 ? In[1] : In[2]

	// Reductions over Attrs.Axes; reduced axes keep size 1.
	OpReduceSum
	OpReduceMax // NaN propagates

	// Movement. All but Pad are views over the input buffer.
	OpReshape
	OpPermute // Attrs.Axes is the permutation
	OpExpand  // broadcast size-1 dimensions
	OpPad     // Attrs.Pads with Attrs.Value as fill
	OpShrink  // Attrs.Ranges
	OpFlip    // Attrs.Axes

	// Contraction: [..., M, K] x [..., K, N] -> [..., M, N] with equal batch
	// dims. A kernel sums the products over the batch axes Attrs.Axes, which
	// its output keeps with size 1.
	OpMatMul

	// Indexing along Attrs.Axis with Int32 indices. A gather from a
	// block-quantized In[0] dequantizes the elements it reads into a float
	// output.
	OpGather     // out[m] = In[0][m with m[axis] = In[1][m]]
	OpScatterAdd // out = In[0]; out[m with m[axis] = In[1][m]] += In[2][m]

	// Sliding windows over the trailing len(Attrs.Kernel) dimensions, D,
	// with Attrs.Stride, Attrs.Dilation and Attrs.Pads before and after
	// each. Window position o and kernel offset k read input position
	// i = o*stride - pad + k*dilation.
	OpUnfold // [B..., D...] -> [B..., K..., O...]: element [b, k, o] is In[0][b, i], or Attrs.Value in the padding
	OpFold   // [B..., K..., O...] -> [B..., D...]: sums element [b, k, o] into [b, i], dropping the padding

	// OpContiguous copies a strided view into a contiguous buffer. The
	// scheduler inserts it; it is not part of the public IR.
	OpContiguous

	// OpFused evaluates the elementwise program Attrs.Program over inputs
	// read through views of the output's shape. The scheduler inserts it
	// for chains of elementwise ops; it is not part of the public IR.
	OpFused

	// OpScan is the inclusive scan of In[0] along Attrs.Axis by
	// Attrs.Combine: out[m] combines the elements In[0][m'] with m' = m
	// outside axis and m'[axis] <= m[axis], in order along axis. Half
	// elements are combined in float32 and rounded once, into an output
	// of their dtype.
	OpScan

	// Joining. OpCat concatenates its inputs along Attrs.Axis, each after
	// the ones before it, like torch.cat: they share the output's dtype,
	// rank and every dimension but the axis. It takes at least one input
	// and at most MaxCatInputs.
	OpCat

	// OpQMatMul multiplies by the transpose of a block-quantized matrix,
	// like a linear layer: In[0] is Float32 [..., K], In[1] a
	// block-quantized [N, K] read contiguously from a block boundary, and
	// the Float32 output [..., N] holds x·Wᵀ. Backends may round x to the
	// precision of the weights first, as the CPU backend rounds each block
	// of a row of x to int8 with a float32 scale, so results differ from
	// the product of the dequantized weights by about the weights'
	// quantization error.
	OpQMatMul

	// Attention. Queries In[0] are [B..., H, Q, D], keys In[1] [B..., HK,
	// K, D] and values In[2] [B..., HK, K, DV], where HK divides H and
	// query head h reads key head h/(H/HK). See AttentionShape.
	//
	// OpAttention computes softmax(Attrs.Scale·q·kᵀ + mask)·v without
	// materializing the scores, adding the optional mask In[3], shaped
	// [B..., H, Q, K], and hiding key j from query i when Attrs.Causal is
	// set and j > i + Attrs.CausalOffset. Its Float32 output holds the
	// result, [B..., H, Q, DV], followed by the natural log of each
	// softmax's denominator, [B..., H, Q], whose maximum term is factored
	// out first: the logsumexp of the row's scores. A query that may
	// attend to no key gets zeros and a logsumexp of -Inf.
	OpAttention
	// OpAttentionGrad computes the gradients of an OpAttention from its
	// inputs, its result In[3] and logsumexp In[4] (views of its output),
	// and the gradient of its result In[5], with the forward kernel's
	// mask, if any, as In[6]. Its Float32 output holds the gradients of
	// the queries, keys and values in turn, each shaped like its input.
	OpAttentionGrad
	// OpSliceScatter copies In[0] with the block Attrs.Ranges replaced
	// by In[1], which has the block's shape.
	OpSliceScatter

	// Convolutions. The scheduler inserts them, in place of products
	// of the windows of OpUnfold, for a backend that runs them (see
	// Convolver); they are not part of the public IR. The windows never
	// take a buffer: a kernel reads them from the image, an input X
	// shaped [N, G*C, D...], as W[n, g, j, o] = X[n, g*C + c, i] with
	// j = (c, k...) and o = (o...) in row-major order and i = o*stride -
	// pad + k*dilation along each windowed dimension as for OpUnfold, or
	// 0 in the padding: OpUnfold of X reshaped to [N, G, C*K, O]. Float32
	// only.
	OpConv       // In[0] [N, G, M, C*K] times W, In[1]'s windows: [N, G, M, O], with an epilogue as OpMatMul has
	OpConvWeight // In[0] [N, G, M, O] times Wᵀ, In[1]'s windows, summed over the pairs (n, o): [S, G, M, C*K] (see ConvParts)
	OpConvInput  // In[0] [N, G, C*K, M] times In[1] [N, G, M, O], folded as OpFold folds windows: [N, G*C, D...]

	// OpArgsort holds Int32 positions along Attrs.Axis: element j along
	// the axis is the position of the j-th element of the stable sort of
	// In[0]'s line, ascending, or descending when Attrs.Descending is set.
	// NaN sorts as the largest value and ties keep their order. The
	// output has In[0]'s shape but along the axis, where it holds every
	// position or, for a top-k, the first K, with K at most MaxTopK.
	OpArgsort

	// OpFFT is the discrete Fourier transform along Attrs.Axis of the
	// complex Float32 In[0], [..., 2], whose last dimension holds the real
	// and imaginary parts and is not the axis: out[k] = Σ x[j]·e^(-2πijk/n)
	// over the n positions j, or e^(+2πijk/n) when Attrs.Inverse is set.
	// Neither direction scales its result.
	OpFFT

	// Linear algebra on Float32 matrices: the last two dimensions of
	// In[0], [B..., m, n], batched over the leading ones. Factorizations
	// other than OpCholesky write a one-dimensional output that holds
	// their parts in turn, each contiguous and shaped as FactorParts
	// returns.
	OpCholesky        // the lower factor L of a positive definite [B..., n, n], read from its lower triangle, with zeros above the diagonal; ErrSingular unless positive definite
	OpTriangularSolve // X of A·X = B for a triangular In[0] [B..., n, n], read from its lower triangle or, with Attrs.Upper, its upper, with ones on the diagonal if Attrs.Unit, and In[1] [B..., n, k]; ErrSingular on a zero diagonal
	OpLU              // P·A = L·U with partial pivoting: L and U packed in [B..., n, n] with L's unit diagonal implied, the permutation [B..., n], whose element i is the row of A that row i of P·A is, and its sign [B...]; a zero pivot is kept
	OpQR              // A = Q·R from Householder reflections, with min(m, n) columns of Q and rows of R: Q [B..., m, k], R [B..., k, n]; R's diagonal has LAPACK's signs
	OpSVD             // A = U·diag(S)·Vh with k = min(m, n) singular values in descending order: U [B..., m, k], S [B..., k], Vh [B..., k, n]
	OpEigh            // A = V·diag(W)·Vᵀ for a symmetric [B..., n, n], read from its lower triangle: W [B..., n] in ascending order, V [B..., n, n]

	// Special functions, unary and elementwise like OpExp2, on float types
	// only. Each is a pole or a limit where its mathematical function has
	// one, as in torch.special.
	OpLog1p    // log(1 + x)
	OpExpm1    // e^x - 1
	OpErf      // the error function
	OpErfinv   // the inverse of the error function on [-1, 1]: ±Inf at ±1, NaN outside
	OpLgamma   // log|Γ(x)|: +Inf at the non-positive integers
	OpDigamma  // ψ(x), the derivative of log Γ(x): -Inf at +0, +Inf at -0, NaN at the negative integers
	OpTrigamma // ψ'(x), the derivative of ψ: +Inf at the non-positive integers

	// Samplers whose elements each draw as many Philox4x32-10 blocks as
	// their rejection loop needs, keyed by Attrs.Seed: element i draws
	// blocks (i, j, 0, word) for j = 0, 1 and so on, where word tells the
	// ops apart. They map a Float32 parameter In[0] to Float32 samples of
	// its shape.
	OpRandGamma   // Gamma(In[0], 1) by Marsaglia and Tsang's method, with Gamma(α+1)·U^(1/α) for α < 1; at least the smallest normal float32
	OpRandPoisson // Poisson(In[0]) by multiplication of uniforms below 10 and Hörmann's PTRS above
	// OpGammaGrad is the derivative of a sample In[1] of Gamma(In[0], 1)
	// with respect to In[0] that keeps its CDF fixed, the implicit
	// reparameterization gradient: -(∂P(α, x)/∂α) / p(x; α).
	OpGammaGrad

	// OpSegment combines rows of In[0], [R, D...], in segments, as the
	// product of a sparse matrix in compressed rows: In[1], Int32 [S+1],
	// holds the offsets of S segments into In[2], Int32 [J], which lists
	// rows of In[0]. Row s of the output, [S, D...], combines the rows
	// In[2][j] for j from In[1][s] up to In[1][s+1], in that order, by
	// Attrs.Combine, CombineSum or CombineMax, and is zero for an empty
	// segment. Offsets that fall or leave [0, J], and rows outside
	// [0, R), fail with ErrIndex. Half rows are combined in float32 and
	// rounded once.
	OpSegment

	// OpScatter replaces elements of In[0] along Attrs.Axis at Int32
	// positions, like torch.Tensor.scatter: out = In[0]; out[m with
	// m[axis] = In[1][m]] = In[2][m]. In[1] and In[2] share a shape that
	// fits In[0] outside the axis. Of the updates of one element, the
	// last along the axis lands. Positions outside the axis fail with
	// ErrIndex.
	OpScatter

	// Pooling over the trailing len(Attrs.Kernel) dimensions of a Float32
	// In[0], [B..., D...], through windows placed as OpUnfold places them,
	// for a backend that runs them (see Pooler); other backends pool with
	// OpUnfold and a reduction. Padding holds no element: a window's
	// maximum is over the elements inside the input, and NaN propagates.
	OpMaxPool // [B..., D...] -> [B..., O...]: the largest element of each window
	// OpMaxPoolGrad routes the gradient In[1], [B..., O...], of an
	// OpMaxPool of In[0] back to the input: each window's gradient goes to
	// the first of its elements, in row-major order, that holds the
	// maximum, or NaN, as torch.nn.functional.max_pool2d's does, and
	// elements that several windows pick sum their gradients.
	OpMaxPoolGrad

	numOps
)

// Combine is the function a scan combines elements with. Each is
// associative, so a device may group the elements of a line as it likes,
// and the last bits of a float scan can differ between devices.
type Combine uint8

// Scan combines.
const (
	CombineSum       Combine = iota + 1 // a + b; Int32 wraps
	CombineMax                          // max(a, b), like OpMax: NaN propagates
	CombineLogSumExp                    // log(exp(a) + exp(b)), without overflow; float types only
)

func (c Combine) String() string {
	switch c {
	case CombineSum:
		return "sum"
	case CombineMax:
		return "max"
	case CombineLogSumExp:
		return "logsumexp"
	default:
		return fmt.Sprintf("Combine(%d)", uint8(c))
	}
}

// MaxTopK is the most positions an OpArgsort keeps along its axis when
// it keeps fewer than all, so that a GPU group can hold them while it
// scans the rest of a line.
const MaxTopK = 1024

// MaxCatInputs is the most inputs an OpCat kernel takes, so that the
// buffers of a GPU kernel fit its argument table. Longer joins are trees
// of OpCat.
const MaxCatInputs = 16

var opNames = [numOps]string{
	OpInvalid:     "invalid",
	OpLoad:        "load",
	OpConst:       "const",
	OpRandUniform: "rand_uniform",
	OpRandNormal:  "rand_normal",
	OpNeg:         "neg",
	OpExp2:        "exp2",
	OpLog2:        "log2",
	OpSin:         "sin",
	OpSqrt:        "sqrt",
	OpRecip:       "recip",
	OpCast:        "cast",
	OpAdd:         "add",
	OpMul:         "mul",
	OpMax:         "max",
	OpCmpLT:       "cmplt",
	OpCmpEQ:       "cmpeq",
	OpMod:         "mod",
	OpWhere:       "where",
	OpReduceSum:   "reduce_sum",
	OpReduceMax:   "reduce_max",
	OpReshape:     "reshape",
	OpPermute:     "permute",
	OpExpand:      "expand",
	OpPad:         "pad",
	OpShrink:      "shrink",
	OpFlip:        "flip",
	OpMatMul:      "matmul",
	OpGather:      "gather",
	OpScatterAdd:  "scatter_add",
	OpUnfold:      "unfold",
	OpFold:        "fold",
	OpContiguous:  "contiguous",
	OpFused:       "fused",
	OpScan:        "scan",
	OpCat:         "cat",
	OpQMatMul:     "qmatmul",

	OpAttention:     "attention",
	OpAttentionGrad: "attention_grad",
	OpSliceScatter:  "slice_scatter",

	OpConv:       "conv",
	OpConvWeight: "conv_weight",
	OpConvInput:  "conv_input",

	OpArgsort:         "argsort",
	OpFFT:             "fft",
	OpCholesky:        "cholesky",
	OpTriangularSolve: "triangular_solve",
	OpLU:              "lu",
	OpQR:              "qr",
	OpSVD:             "svd",
	OpEigh:            "eigh",

	OpLog1p:       "log1p",
	OpExpm1:       "expm1",
	OpErf:         "erf",
	OpErfinv:      "erfinv",
	OpLgamma:      "lgamma",
	OpDigamma:     "digamma",
	OpTrigamma:    "trigamma",
	OpRandGamma:   "rand_gamma",
	OpRandPoisson: "rand_poisson",
	OpGammaGrad:   "gamma_grad",
	OpSegment:     "segment",
	OpScatter:     "scatter",
	OpMaxPool:     "max_pool",
	OpMaxPoolGrad: "max_pool_grad",
}

func (o Op) String() string {
	if o < numOps {
		return opNames[o]
	}
	return fmt.Sprintf("Op(%d)", uint8(o))
}

// IsUnary reports whether o takes one input and applies elementwise.
func (o Op) IsUnary() bool { return o >= OpNeg && o <= OpCast || o.IsSpecial() }

// IsSpecial reports whether o is one of the special functions, unary ops
// defined on float types only, from OpLog1p to OpTrigamma.
func (o Op) IsSpecial() bool { return o >= OpLog1p && o <= OpTrigamma }

// IsBinary reports whether o takes two same-shaped inputs elementwise.
func (o Op) IsBinary() bool { return o >= OpAdd && o <= OpMod }

// IsElementwise reports whether o computes each output element from the
// input elements at the same index, so that it can join a fused program.
func (o Op) IsElementwise() bool { return o.IsUnary() || o.IsBinary() || o == OpWhere }

// IsReduce reports whether o reduces over Attrs.Axes.
func (o Op) IsReduce() bool { return o == OpReduceSum || o == OpReduceMax }

// IsView reports whether o can be expressed as a strided view of its input.
func (o Op) IsView() bool {
	switch o {
	case OpReshape, OpPermute, OpExpand, OpShrink, OpFlip:
		return true
	default:
		return false
	}
}
