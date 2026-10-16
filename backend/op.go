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
	OpCast  // convert to Kernel.DType

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

	// Indexing along Attrs.Axis with Int32 indices.
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
}

func (o Op) String() string {
	if o < numOps {
		return opNames[o]
	}
	return fmt.Sprintf("Op(%d)", uint8(o))
}

// IsUnary reports whether o takes one input and applies elementwise.
func (o Op) IsUnary() bool { return o >= OpNeg && o <= OpCast }

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
