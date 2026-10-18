package mim

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// ExportOptions configures Export and ExportFunc. The zero value is the
// default.
type ExportOptions struct {
	// Inputs and Outputs name the graph's inputs and outputs in order.
	// Those left unnamed are "input", "input1" and so on, and "output",
	// "output1" and so on.
	Inputs, Outputs []string
	// Names labels the tensors that the function reads, such as
	// parameters: each becomes a weight of the file under its name. The
	// other tensors it reads become constants named "const.N". Export
	// adds the parameters of the module, named by nn.NamedParams.
	Names map[string]*tensor.Tensor
	// Metadata is written as the file's metadata section.
	Metadata mimfile.Metadata
}

// Export writes m to w as a Mimir Model whose graph computes m.Forward
// for an input of x's dtype and shape. It is ExportFunc of m.Forward, with
// m's parameters named by their paths.
//
// The model must be in inference mode (see nn.Eval), so that dropout
// draws no random numbers.
func Export(ctx context.Context, w io.Writer, m nn.Module, x *tensor.Tensor, opts *ExportOptions) error {
	o := ExportOptions{}
	if opts != nil {
		o = *opts
	}
	names := map[string]*tensor.Tensor{}
	for name, p := range nn.NamedParams(m) {
		names[name] = p.Value()
	}
	for name, t := range o.Names {
		names[name] = t
	}
	o.Names = names
	f := func(x ...*tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{m.Forward(x[0])} }
	return ExportFunc(ctx, w, f, []*tensor.Tensor{x}, &o)
}

// ExportFunc writes f to w as a Mimir Model whose graph computes f's
// results from inputs of the dtypes and shapes of args. It traces f once
// (see tensor.Trace) and writes the operations it records, with the
// tensors f reads as the file's weights and constants, in the dtypes they
// have: a model whose linear layers hold block-quantized weights is
// written with them. Shapes are fixed at those of args.
//
// Most operations become primitives of the format. Products become calls
// of matmul.v1, products with block-quantized weights calls of
// dequant_matmul_a8.v2, whose tolerance admits the 8-bit rounding of the
// activations that mimir's kernels do, and attention calls of
// attention.v1, or of attention.v2 for keys or values of another dtype
// than the queries. An operation the format cannot express, such as a random
// draw or a factorization, fails with ErrUnsupported unless no argument
// affects it: its result is then computed and written as a constant.
func ExportFunc(ctx context.Context, w io.Writer, f func(x ...*tensor.Tensor) []*tensor.Tensor, args []*tensor.Tensor, opts *ExportOptions) error {
	o := ExportOptions{}
	if opts != nil {
		o = *opts
	}
	tg, err := tensor.Trace(f, args, o.Names)
	if err != nil {
		return fmt.Errorf("mim: export: %w", err)
	}
	e := &exporter{ctx: ctx, tg: tg, vals: make([]uint32, len(tg.Nodes)), bools: map[int]uint32{}, taken: map[string]bool{}}
	for name := range o.Names {
		e.taken[name] = true
	}
	if err := e.graph(args, o.Inputs, o.Outputs); err != nil {
		return fmt.Errorf("mim: export: %w", err)
	}
	return e.write(w, o.Metadata)
}

// exporter translates a traced graph into a mim graph.
type exporter struct {
	ctx     context.Context
	tg      *tensor.TracedGraph
	g       mimfile.Graph
	tensors []stored
	vals    []uint32       // the value of each node
	bools   map[int]uint32 // the bool value of nodes cast from a comparison
	taken   map[string]bool
	consts  int
}

// stored is a tensor of the weights section.
type stored struct {
	name  string
	dt    mimfile.DType
	shape []uint64
	data  []byte
}

func (e *exporter) graph(args []*tensor.Tensor, inNames, outNames []string) error {
	e.g = mimfile.Graph{
		PrimitiveVersion: mimfile.CurrentPrimitive,
		StdlibVersion:    mimfile.CurrentStdlib,
		Subgraphs:        []mimfile.Subgraph{{}},
	}
	live, folded := e.plan()
	// Every argument is an input, in order, whether f reads it or not.
	main := &e.g.Subgraphs[0]
	for a, x := range args {
		dt, err := mimDType(x.DType())
		if err != nil {
			return err
		}
		main.Inputs = append(main.Inputs, e.value(mimfile.Value{Name: nameAt(inNames, a, "input"), DType: dt, Shape: dims(x.Shape()), Kind: mimfile.KindInput}))
	}
	for i, n := range e.tg.Nodes {
		if n.Arg >= 0 {
			e.vals[i] = main.Inputs[n.Arg]
		}
	}
	for i, n := range e.tg.Nodes {
		if !live[i] || n.Arg >= 0 {
			continue
		}
		var err error
		if folded[i] {
			err = e.fold(i)
		} else {
			err = e.node(i)
		}
		if err != nil {
			return err
		}
	}
	for k, i := range e.tg.Outputs {
		v := e.vals[i]
		// An output is a value of its own, computed by an op: an argument
		// or a stored tensor returned as it is, or a value returned twice,
		// is copied by a reshape.
		if !e.computed(v) || slices.Contains(main.Outputs, v) {
			n := e.tg.Nodes[i]
			v = e.op(mimfile.OpReshape, e.g.Values[v].DType, n.Shape, []uint32{v}, ints("shape", n.Shape))
		}
		e.g.Values[v].Kind = mimfile.KindOutput
		e.g.Values[v].Name = nameAt(outNames, k, "output")
		main.Outputs = append(main.Outputs, v)
	}
	return e.g.Validate()
}

// computed reports whether an op of the main graph computes value v.
func (e *exporter) computed(v uint32) bool {
	k := e.g.Values[v].Kind
	return k == mimfile.KindIntermediate || k == mimfile.KindOutput
}

// plan marks the nodes that the outputs need, and the nodes written as
// constants: those that no argument affects and that the format cannot
// express, or that hold block-quantized values other than as stored.
func (e *exporter) plan() (live, folded []bool) {
	nodes := e.tg.Nodes
	live, folded = make([]bool, len(nodes)), make([]bool, len(nodes))
	for i, n := range nodes {
		if n.Value == nil || n.Op == backend.OpLoad {
			continue
		}
		folded[i] = (!translatable(n.Op) || n.DType.IsQuantized()) && !random(n.Op)
	}
	for _, i := range e.tg.Outputs {
		live[i] = true
	}
	for i := len(nodes) - 1; i >= 0; i-- {
		if !live[i] || folded[i] {
			continue
		}
		for _, in := range nodes[i].Inputs {
			live[in] = true
		}
	}
	return live, folded
}

// random reports whether op draws random numbers, which are never
// written as constants: a model draws them for dropout in training mode.
func random(op backend.Op) bool {
	switch op {
	case backend.OpRandUniform, backend.OpRandNormal, backend.OpRandGamma, backend.OpRandPoisson:
		return true
	}
	return false
}

// translatable reports whether node writes op.
func translatable(op backend.Op) bool {
	switch op {
	case backend.OpLoad, backend.OpConst,
		backend.OpNeg, backend.OpExp2, backend.OpLog2, backend.OpSin, backend.OpSqrt, backend.OpRecip, backend.OpCast,
		backend.OpAdd, backend.OpMul, backend.OpMax, backend.OpCmpLT, backend.OpCmpEQ, backend.OpMod, backend.OpWhere,
		backend.OpReduceSum, backend.OpReduceMax,
		backend.OpReshape, backend.OpPermute, backend.OpExpand, backend.OpPad, backend.OpShrink, backend.OpFlip,
		backend.OpMatMul, backend.OpQMatMul, backend.OpGather, backend.OpUnfold, backend.OpScan, backend.OpCat,
		backend.OpAttention, backend.OpArgsort, backend.OpErf:
		return true
	}
	return false
}

func nameAt(names []string, i int, base string) string {
	if i < len(names) && names[i] != "" {
		return names[i]
	}
	if i == 0 {
		return base
	}
	return base + strconv.Itoa(i)
}

// mimDType returns the mim dtype of mimir's dt.
func mimDType(dt tensor.DType) (mimfile.DType, error) {
	switch dt {
	case tensor.Float32:
		return mimfile.F32, nil
	case tensor.Float16:
		return mimfile.F16, nil
	case tensor.BFloat16:
		return mimfile.BF16, nil
	case tensor.Int32:
		return mimfile.I32, nil
	case tensor.Q8_0:
		return mimfile.Q8_0, nil
	case tensor.Q4_K:
		return mimfile.Q4_K, nil
	case tensor.Q6_K:
		return mimfile.Q6_K, nil
	}
	return 0, &UnsupportedError{Op: "dtype", Detail: dt.String()}
}

func dims(shape []int) []mimfile.Dim {
	d := make([]mimfile.Dim, len(shape))
	for i, s := range shape {
		d[i] = mimfile.Dim(s)
	}
	return d
}

func ints(key string, v []int) mimfile.Attr {
	a := mimfile.Attr{Key: key, Type: mimfile.AttrInts, Ints: make([]int64, len(v))}
	for i, x := range v {
		a.Ints[i] = int64(x)
	}
	return a
}

func intAttr(key string, v int) mimfile.Attr {
	return mimfile.Attr{Key: key, Type: mimfile.AttrInt, Int: int64(v)}
}

func boolAttr(key string, v bool) mimfile.Attr {
	if v {
		return intAttr(key, 1)
	}
	return intAttr(key, 0)
}

func floatAttr(key string, v float64) mimfile.Attr {
	return mimfile.Attr{Key: key, Type: mimfile.AttrFloat, Float: v}
}

func (e *exporter) value(v mimfile.Value) uint32 {
	e.g.Values = append(e.g.Values, v)
	return uint32(len(e.g.Values) - 1)
}

// ops adds an op of the main graph with outputs of the given dtypes and
// shapes.
func (e *exporter) ops(code mimfile.Opcode, dts []mimfile.DType, shapes [][]int, ins []uint32, attrs ...mimfile.Attr) []uint32 {
	outs := make([]uint32, len(dts))
	for i := range outs {
		outs[i] = e.value(mimfile.Value{DType: dts[i], Shape: dims(shapes[i]), Kind: mimfile.KindIntermediate})
	}
	main := &e.g.Subgraphs[0]
	main.Ops = append(main.Ops, mimfile.Op{Opcode: code, Inputs: ins, Outputs: outs, Attrs: attrs})
	return outs
}

// op adds an op with one output.
func (e *exporter) op(code mimfile.Opcode, dt mimfile.DType, shape []int, ins []uint32, attrs ...mimfile.Attr) uint32 {
	return e.ops(code, []mimfile.DType{dt}, [][]int{shape}, ins, attrs...)[0]
}

// call adds a call of the standard library with one output.
func (e *exporter) call(name string, version int, dt mimfile.DType, shape []int, ins []uint32, attrs ...mimfile.Attr) uint32 {
	attrs = append([]mimfile.Attr{{Key: "name", Type: mimfile.AttrString, String: name}, intAttr("version", version)}, attrs...)
	return e.op(mimfile.OpCall, dt, shape, ins, attrs...)
}

// constant adds a const op of dtype dt, shape and value v.
func (e *exporter) constant(dt mimfile.DType, shape []int, v float64) uint32 {
	a := floatAttr("value", v)
	if dt == mimfile.I32 || dt == mimfile.Bool {
		a = intAttr("value", int(v))
	}
	return e.op(mimfile.OpConst, dt, shape, nil, a)
}

// store adds a tensor of the weights section and a value of kind reading
// it. A name of "" makes the tensor a constant named const.N.
func (e *exporter) store(name string, kind mimfile.ValueKind, dt mimfile.DType, shape []int, data []byte) uint32 {
	if name == "" {
		for {
			name = "const." + strconv.Itoa(e.consts)
			e.consts++
			if !e.taken[name] {
				break
			}
		}
	}
	e.taken[name] = true
	us := make([]uint64, len(shape))
	for i, d := range shape {
		us[i] = uint64(d)
	}
	e.tensors = append(e.tensors, stored{name: name, dt: dt, shape: us, data: data})
	v := mimfile.Value{DType: dt, Shape: dims(shape), Kind: kind, Ref: uint32(len(e.tensors) - 1)}
	if kind == mimfile.KindWeight {
		v.Name = name
	}
	return e.value(v)
}

// int32s stores v as a constant of dtype i32 and shape.
func (e *exporter) int32s(shape []int, v []int32) uint32 {
	data := make([]byte, 4*len(v))
	for i, x := range v {
		le32(data[4*i:], uint32(x))
	}
	return e.store("", mimfile.KindConst, mimfile.I32, shape, data)
}

// float32s stores v as a constant of dtype f32 and shape.
func (e *exporter) float32s(shape []int, v []float32) uint32 {
	data := make([]byte, 4*len(v))
	for i, x := range v {
		le32(data[4*i:], math.Float32bits(x))
	}
	return e.store("", mimfile.KindConst, mimfile.F32, shape, data)
}

func le32(b []byte, x uint32) { b[0], b[1], b[2], b[3] = byte(x), byte(x>>8), byte(x>>16), byte(x>>24) }

// fold writes node i, which no argument affects, as a stored constant.
func (e *exporter) fold(i int) error {
	n := e.tg.Nodes[i]
	dt, err := mimDType(n.DType)
	if err != nil {
		return err
	}
	data, err := n.Value.Bytes(e.ctx)
	if err != nil {
		return fmt.Errorf("compute the constant %s%v: %w", n.Op, n.Shape, err)
	}
	e.vals[i] = e.store("", mimfile.KindConst, dt, n.Shape, data)
	return nil
}

// unsupported reports node i, which the format cannot express.
func (e *exporter) unsupported(i int, detail string) error {
	return &UnsupportedError{Op: e.tg.Nodes[i].Op.String(), Detail: detail}
}

// node writes node i, whose inputs are written.
func (e *exporter) node(i int) error {
	n := e.tg.Nodes[i]
	dt, err := mimDType(n.DType)
	if err != nil {
		return err
	}
	in := make([]uint32, len(n.Inputs))
	for k, j := range n.Inputs {
		in[k] = e.vals[j]
		if e.tg.Nodes[j].Op == backend.OpAttention && n.Op != backend.OpShrink {
			return e.unsupported(j, "use of the logsumexp")
		}
	}
	ins := func(k int) tensor.TracedNode { return e.tg.Nodes[n.Inputs[k]] }
	v, err := uint32(0), error(nil)
	switch n.Op {
	case backend.OpLoad:
		kind := mimfile.KindConst
		if n.Name != "" {
			kind = mimfile.KindWeight
		}
		data, err := n.Value.Bytes(e.ctx)
		if err != nil {
			return fmt.Errorf("read %q: %w", n.Name, err)
		}
		v = e.store(n.Name, kind, dt, n.Shape, data)
	case backend.OpConst:
		v = e.constant(dt, n.Shape, n.Attrs.Value)
	case backend.OpNeg, backend.OpExp2, backend.OpLog2, backend.OpSin, backend.OpSqrt, backend.OpRecip:
		v = e.op(unaryOps[n.Op], dt, n.Shape, in)
	case backend.OpErf:
		v = e.call("erf", 1, dt, n.Shape, in)
	case backend.OpCast:
		v = e.cast(in[0], ins(0).DType, dt, n.Shape)
	case backend.OpAdd, backend.OpMul, backend.OpMax:
		v = e.op(binaryOps[n.Op], dt, n.Shape, in)
	case backend.OpCmpLT, backend.OpCmpEQ:
		b := e.op(binaryOps[n.Op], mimfile.Bool, n.Shape, in)
		e.bools[i] = b
		v = e.op(mimfile.OpCast, dt, n.Shape, []uint32{b}, mimfile.Attr{Key: "to", Type: mimfile.AttrDType, DType: dt})
	case backend.OpMod:
		if dt != mimfile.I32 {
			return e.unsupported(i, "remainder of "+n.DType.String())
		}
		v = e.op(mimfile.OpImod, dt, n.Shape, in)
	case backend.OpWhere:
		cond, ok := e.bools[n.Inputs[0]]
		if !ok {
			cond = e.op(mimfile.OpCast, mimfile.Bool, ins(0).Shape, in[:1], mimfile.Attr{Key: "to", Type: mimfile.AttrDType, DType: mimfile.Bool})
		}
		v = e.op(mimfile.OpWhere, dt, n.Shape, []uint32{cond, in[1], in[2]})
	case backend.OpReduceSum, backend.OpReduceMax:
		mode := "add"
		if n.Op == backend.OpReduceMax {
			mode = "max"
		}
		v = e.op(mimfile.OpReduce, dt, n.Shape, in, mimfile.Attr{Key: "mode", Type: mimfile.AttrString, String: mode}, ints("axes", n.Attrs.Axes))
	case backend.OpReshape:
		v = e.op(mimfile.OpReshape, dt, n.Shape, in, ints("shape", n.Shape))
	case backend.OpPermute:
		v = e.op(mimfile.OpPermute, dt, n.Shape, in, ints("perm", n.Attrs.Axes))
	case backend.OpExpand:
		v = e.op(mimfile.OpExpand, dt, n.Shape, in)
	case backend.OpPad:
		v, err = e.pad(i, in[0], dt, n.Shape, n.Attrs.Pads, n.Attrs.Value)
	case backend.OpShrink:
		v, err = e.shrink(i, in[0], dt)
	case backend.OpFlip:
		v = in[0]
		for _, a := range n.Attrs.Axes {
			idx := make([]int32, n.Shape[a])
			for k := range idx {
				idx[k] = int32(len(idx) - 1 - k)
			}
			v = e.op(mimfile.OpGather, dt, n.Shape, []uint32{v, e.int32s([]int{len(idx)}, idx)}, intAttr("axis", a))
		}
	case backend.OpMatMul:
		if len(n.Attrs.Axes) > 0 {
			return e.unsupported(i, "products summed over batch axes")
		}
		v = e.call("matmul", 1, dt, n.Shape, in)
	case backend.OpQMatMul:
		v, err = e.qmatmul(i, in)
	case backend.OpGather:
		v, err = e.gather(i, in)
	case backend.OpUnfold:
		v, err = e.unfold(i, in[0])
	case backend.OpScan:
		if n.Attrs.Combine != backend.CombineSum {
			return e.unsupported(i, n.Attrs.Combine.String()+" scan")
		}
		v = e.call("cumsum", 1, dt, n.Shape, in, intAttr("axis", n.Attrs.Axis))
	case backend.OpCat:
		v = e.cat(i, in, dt)
	case backend.OpAttention:
		v, err = e.attention(i, in)
	case backend.OpArgsort:
		v = e.argsort(i, in[0])
	case backend.OpRandUniform, backend.OpRandNormal, backend.OpRandGamma, backend.OpRandPoisson:
		return e.unsupported(i, "random numbers, which a model in training mode draws for dropout; export it after nn.Eval, or realize a random tensor it reads")
	default:
		return e.unsupported(i, "")
	}
	if err != nil {
		return err
	}
	e.vals[i] = v
	return nil
}

var unaryOps = map[backend.Op]mimfile.Opcode{
	backend.OpNeg: mimfile.OpNeg, backend.OpExp2: mimfile.OpExp2, backend.OpLog2: mimfile.OpLog2,
	backend.OpSin: mimfile.OpSin, backend.OpSqrt: mimfile.OpSqrt, backend.OpRecip: mimfile.OpRecip,
}

var binaryOps = map[backend.Op]mimfile.Opcode{
	backend.OpAdd: mimfile.OpAdd, backend.OpMul: mimfile.OpMul, backend.OpMax: mimfile.OpMax,
	backend.OpCmpLT: mimfile.OpCmplt, backend.OpCmpEQ: mimfile.OpCmpeq,
}

// cast converts v of mimir's from to dt: a block-quantized v is decoded
// first, which a cast cannot do.
func (e *exporter) cast(v uint32, from tensor.DType, dt mimfile.DType, shape []int) uint32 {
	if from.IsQuantized() {
		v = e.op(mimfile.OpDequantize, mimfile.F32, shape, []uint32{v})
		if dt == mimfile.F32 {
			return v
		}
	}
	return e.op(mimfile.OpCast, dt, shape, []uint32{v}, mimfile.Attr{Key: "to", Type: mimfile.AttrDType, DType: dt})
}

func (e *exporter) pad(i int, v uint32, dt mimfile.DType, shape []int, pads [][2]int, value float64) (uint32, error) {
	flat := make([]int, 0, 2*len(pads))
	for _, p := range pads {
		if p[0] < 0 || p[1] < 0 {
			return 0, e.unsupported(i, fmt.Sprintf("negative padding %v", pads))
		}
		flat = append(flat, p[0], p[1])
	}
	return e.op(mimfile.OpPad, dt, shape, []uint32{v}, ints("pads", flat), floatAttr("value", value)), nil
}

// shrink writes node i, a shrink, which may read the result of an
// attention, whose node also holds the logsumexp after it.
func (e *exporter) shrink(i int, v uint32, dt mimfile.DType) (uint32, error) {
	n := e.tg.Nodes[i]
	begin, end := make([]int, len(n.Attrs.Ranges)), make([]int, len(n.Attrs.Ranges))
	for k, r := range n.Attrs.Ranges {
		begin[k], end[k] = r[0], r[1]
	}
	if src := e.tg.Nodes[n.Inputs[0]]; src.Op == backend.OpAttention {
		res := int(e.g.Values[v].Shape[0])
		if end[0] > res {
			return 0, e.unsupported(n.Inputs[0], "use of the logsumexp")
		}
		if begin[0] == 0 && end[0] == res {
			return v, nil
		}
	}
	return e.op(mimfile.OpShrink, dt, n.Shape, []uint32{v}, ints("begin", begin), ints("end", end)), nil
}

// qmatmul writes a product with a block-quantized matrix: x [..., K] by
// w [N, K] transposed.
func (e *exporter) qmatmul(i int, in []uint32) (uint32, error) {
	n := e.tg.Nodes[i]
	x, w := e.tg.Nodes[n.Inputs[0]], e.tg.Nodes[n.Inputs[1]]
	block := 32
	switch w.DType {
	case tensor.Q8_0:
	case tensor.Q4_K, tensor.Q6_K:
		block = 256
	default:
		return 0, e.unsupported(i, "weights of "+w.DType.String())
	}
	xv, shape := in[0], n.Shape
	if len(x.Shape) == 1 {
		xv = e.op(mimfile.OpReshape, mimfile.F32, []int{1, x.Shape[0]}, []uint32{xv}, ints("shape", []int{1, x.Shape[0]}))
		shape = []int{1, n.Shape[0]}
	}
	y := e.call("dequant_matmul_a8", 2, mimfile.F32, shape, []uint32{xv, in[1]}, intAttr("block", block))
	if len(x.Shape) == 1 {
		y = e.op(mimfile.OpReshape, mimfile.F32, n.Shape, []uint32{y}, ints("shape", n.Shape))
	}
	return y, nil
}

// gather writes mimir's gather, which takes an index of the output's
// shape for every element, as torch.gather does, with the format's,
// which takes whole slices, as numpy.take does.
func (e *exporter) gather(i int, in []uint32) (uint32, error) {
	n := e.tg.Nodes[i]
	x, idx := e.tg.Nodes[n.Inputs[0]], e.tg.Nodes[n.Inputs[1]]
	dt, _ := mimDType(n.DType)
	axis := n.Attrs.Axis
	// The rows of a table, as an embedding reads them: an index [n, 1]
	// expanded along the rows.
	if len(x.Shape) == 2 && axis == 0 && idx.Op == backend.OpExpand {
		src := e.tg.Nodes[idx.Inputs[0]]
		if len(src.Shape) == 2 && src.Shape[1] == 1 && idx.Shape[1] == x.Shape[1] {
			rows := e.op(mimfile.OpReshape, mimfile.I32, src.Shape[:1], []uint32{e.vals[idx.Inputs[0]]}, ints("shape", src.Shape[:1]))
			if x.DType.IsQuantized() {
				y := e.call("embedding", 1, mimfile.F32, n.Shape, []uint32{in[0], rows})
				if dt != mimfile.F32 {
					y = e.op(mimfile.OpCast, dt, n.Shape, []uint32{y}, mimfile.Attr{Key: "to", Type: mimfile.AttrDType, DType: dt})
				}
				return y, nil
			}
			return e.op(mimfile.OpGather, dt, n.Shape, []uint32{in[0], rows}, intAttr("axis", 0)), nil
		}
	}
	if x.DType.IsQuantized() {
		return 0, e.unsupported(i, "a gather of a block-quantized tensor other than its rows")
	}
	// Otherwise, the elements of x flattened, at the offsets the index
	// and its position give.
	size := 1
	strides := make([]int, len(x.Shape))
	for k := len(x.Shape) - 1; k >= 0; k-- {
		strides[k] = size
		size *= x.Shape[k]
	}
	if size > math.MaxInt32 {
		return 0, e.unsupported(i, fmt.Sprintf("a gather from %d elements", size))
	}
	base := make([]int32, 0, product(idx.Shape))
	for pos := range positions(idx.Shape) {
		off := 0
		for k, p := range pos {
			if k != axis {
				off += p * strides[k]
			}
		}
		base = append(base, int32(off))
	}
	ones := slices.Repeat([]int{1}, len(idx.Shape))
	off := e.op(mimfile.OpMul, mimfile.I32, idx.Shape, []uint32{in[1], e.constant(mimfile.I32, ones, float64(strides[axis]))})
	off = e.op(mimfile.OpAdd, mimfile.I32, idx.Shape, []uint32{off, e.int32s(idx.Shape, base)})
	flat := e.op(mimfile.OpReshape, dt, []int{size}, []uint32{in[0]}, ints("shape", []int{size}))
	return e.op(mimfile.OpGather, dt, n.Shape, []uint32{flat, off}, intAttr("axis", 0)), nil
}

// unfold writes the sliding windows of x over its trailing dimensions,
// [B..., D...] to [B..., K..., O...], as a gather from x padded and
// flattened over D at the offsets of each window's elements.
func (e *exporter) unfold(i int, v uint32) (uint32, error) {
	n := e.tg.Nodes[i]
	x := e.tg.Nodes[n.Inputs[0]]
	dt, _ := mimDType(n.DType)
	a := n.Attrs
	nb := len(x.Shape) - len(a.Kernel)
	batch := x.Shape[:nb]
	padded := slices.Clone(x.Shape)
	pads := make([][2]int, len(x.Shape))
	padding := false
	for d := range a.Kernel {
		pads[nb+d] = a.Pads[d]
		padded[nb+d] += a.Pads[d][0] + a.Pads[d][1]
		padding = padding || a.Pads[d] != [2]int{}
	}
	if padding {
		p, err := e.pad(i, v, dt, padded, pads, a.Value)
		if err != nil {
			return 0, err
		}
		v = p
	}
	space := product(padded[nb:])
	if space > math.MaxInt32 {
		return 0, e.unsupported(i, fmt.Sprintf("windows over %d elements", space))
	}
	flatShape := append(slices.Clone(batch), space)
	v = e.op(mimfile.OpReshape, dt, flatShape, []uint32{v}, ints("shape", flatShape))
	strides := make([]int, len(a.Kernel))
	size := 1
	for d := len(a.Kernel) - 1; d >= 0; d-- {
		strides[d] = size
		size *= padded[nb+d]
	}
	out := n.Shape[nb+len(a.Kernel):]
	idxShape := append(slices.Clone(a.Kernel), out...)
	idx := make([]int32, 0, product(idxShape))
	for pos := range positions(idxShape) {
		k, o := pos[:len(a.Kernel)], pos[len(a.Kernel):]
		off := 0
		for d := range k {
			off += (o[d]*a.Stride[d] + k[d]*a.Dilation[d]) * strides[d]
		}
		idx = append(idx, int32(off))
	}
	return e.op(mimfile.OpGather, dt, n.Shape, []uint32{v, e.int32s(idxShape, idx)}, intAttr("axis", nb)), nil
}

// cat writes a concatenation as the sum of the inputs, each padded with
// zeros to the output's length along the axis.
func (e *exporter) cat(i int, in []uint32, dt mimfile.DType) uint32 {
	n := e.tg.Nodes[i]
	if len(in) == 1 {
		return in[0]
	}
	axis, at := n.Attrs.Axis, 0
	var sum uint32
	for k, v := range in {
		shape := e.tg.Nodes[n.Inputs[k]].Shape
		pads := make([]int, 2*len(shape))
		pads[2*axis], pads[2*axis+1] = at, n.Shape[axis]-at-shape[axis]
		at += shape[axis]
		p := e.op(mimfile.OpPad, dt, n.Shape, []uint32{v}, ints("pads", pads))
		if k == 0 {
			sum = p
		} else {
			sum = e.op(mimfile.OpAdd, dt, n.Shape, []uint32{sum, p})
		}
	}
	return sum
}

// attention writes mimir's attention as a call of attention.v1, or v2
// when the keys or values have another dtype than the queries. Its node
// holds the result, [B..., H, Q, DV], flattened, and the logsumexp of
// each row after it, which the value written leaves out.
func (e *exporter) attention(i int, in []uint32) (uint32, error) {
	n := e.tg.Nodes[i]
	q, k, v := e.tg.Nodes[n.Inputs[0]], e.tg.Nodes[n.Inputs[1]], e.tg.Nodes[n.Inputs[2]]
	rq := len(q.Shape)
	if rq < 3 {
		return 0, e.unsupported(i, fmt.Sprintf("queries of shape %v", q.Shape))
	}
	// The format's attention takes one batch axis.
	four := func(shape []int) []int {
		return append([]int{product(shape[:len(shape)-3])}, shape[len(shape)-3:]...)
	}
	vals := slices.Clone(in)
	for j := range vals {
		src := e.tg.Nodes[n.Inputs[j]]
		if len(src.Shape) != 4 {
			sdt, _ := mimDType(src.DType)
			vals[j] = e.op(mimfile.OpReshape, sdt, four(src.Shape), []uint32{vals[j]}, ints("shape", four(src.Shape)))
		}
	}
	qs, ks, vs := four(q.Shape), four(k.Shape), four(v.Shape)
	tq, tk := qs[2], ks[2]
	attrs := []mimfile.Attr{floatAttr("scale", n.Attrs.Scale), intAttr("num_kv_heads", ks[1])}
	if n.Attrs.Causal {
		if off := n.Attrs.CausalOffset; off == tk-tq {
			attrs = append(attrs, boolAttr("causal", true))
		} else {
			// Key j is hidden from query i past i + off.
			m := make([]float32, tq*tk)
			for qi := range tq {
				for kj := range tk {
					if kj > qi+off {
						m[qi*tk+kj] = float32(math.Inf(-1))
					}
				}
			}
			causal := e.float32s([]int{1, 1, tq, tk}, m)
			if len(vals) > 3 {
				mask := e.tg.Nodes[n.Inputs[3]]
				mdt, _ := mimDType(mask.DType)
				if mdt != mimfile.F32 {
					vals[3] = e.op(mimfile.OpCast, mimfile.F32, four(mask.Shape), vals[3:4], mimfile.Attr{Key: "to", Type: mimfile.AttrDType, DType: mimfile.F32})
				}
				shape := four(mask.Shape)
				for d, s := range []int{1, 1, tq, tk} {
					shape[d] = max(shape[d], s)
				}
				vals[3] = e.op(mimfile.OpAdd, mimfile.F32, shape, []uint32{vals[3], causal})
			} else {
				vals = append(vals, causal)
			}
		}
	}
	qdt, _ := mimDType(q.DType)
	version := 1
	if k.DType != q.DType || v.DType != q.DType {
		version = 2
	}
	out := []int{qs[0], qs[1], tq, vs[3]}
	o := e.call("attention", version, qdt, out, vals, attrs...)
	if qdt != mimfile.F32 {
		o = e.op(mimfile.OpCast, mimfile.F32, out, []uint32{o}, mimfile.Attr{Key: "to", Type: mimfile.AttrDType, DType: mimfile.F32})
	}
	flat := []int{product(out)}
	return e.op(mimfile.OpReshape, mimfile.F32, flat, []uint32{o}, ints("shape", flat)), nil
}

// argsort writes the positions of a stable sort, or of a top-k when the
// output keeps fewer positions than the axis holds.
func (e *exporter) argsort(i int, v uint32) uint32 {
	n := e.tg.Nodes[i]
	x := e.tg.Nodes[n.Inputs[0]]
	axis := n.Attrs.Axis
	var idx uint32
	if n.Shape[axis] == x.Shape[axis] {
		idx = e.call("argsort", 1, mimfile.I64, n.Shape, []uint32{v}, intAttr("axis", axis), boolAttr("descending", n.Attrs.Descending))
	} else {
		xdt, _ := mimDType(x.DType)
		attrs := []mimfile.Attr{
			{Key: "name", Type: mimfile.AttrString, String: "top_k"}, intAttr("version", 1),
			intAttr("k", n.Shape[axis]), intAttr("axis", axis), boolAttr("largest", n.Attrs.Descending),
		}
		idx = e.ops(mimfile.OpCall, []mimfile.DType{xdt, mimfile.I64}, [][]int{n.Shape, n.Shape}, []uint32{v}, attrs...)[1]
	}
	return e.op(mimfile.OpCast, mimfile.I32, n.Shape, []uint32{idx}, mimfile.Attr{Key: "to", Type: mimfile.AttrDType, DType: mimfile.I32})
}

func product(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// positions yields every index of shape in row-major order. The slice is
// reused between iterations.
func positions(shape []int) func(yield func([]int) bool) {
	return func(yield func([]int) bool) {
		if product(shape) == 0 {
			return
		}
		pos := make([]int, len(shape))
		for {
			if !yield(pos) {
				return
			}
			d := len(shape) - 1
			for ; d >= 0; d-- {
				if pos[d]++; pos[d] < shape[d] {
					break
				}
				pos[d] = 0
			}
			if d < 0 {
				return
			}
		}
	}
}

// write writes the file: the tensors in name order, as the format's tools
// lay them out, the graph and the metadata.
func (e *exporter) write(w io.Writer, md mimfile.Metadata) error {
	order := make([]int, len(e.tensors))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int { return cmp.Compare(e.tensors[a].name, e.tensors[b].name) })
	fw := mimfile.NewWriter()
	pos := make([]uint32, len(order))
	for p, i := range order {
		t := e.tensors[i]
		if err := fw.AddTensor(t.name, t.dt, t.shape, t.data); err != nil {
			return fmt.Errorf("mim: export tensor %q: %w", t.name, err)
		}
		pos[i] = uint32(p)
	}
	for i, v := range e.g.Values {
		if v.Kind == mimfile.KindWeight || v.Kind == mimfile.KindConst {
			e.g.Values[i].Ref = pos[v.Ref]
		}
	}
	enc, err := e.g.Encode()
	if err != nil {
		return fmt.Errorf("mim: export: %w", err)
	}
	if err := fw.AddSection(mimfile.SectionGraph, true, enc); err != nil {
		return fmt.Errorf("mim: export: %w", err)
	}
	if len(md) > 0 {
		enc, err := md.Encode()
		if err != nil {
			return fmt.Errorf("mim: export metadata: %w", err)
		}
		if err := fw.AddSection(mimfile.SectionMetadata, false, enc); err != nil {
			return fmt.Errorf("mim: export: %w", err)
		}
	}
	if _, err := fw.WriteTo(w); err != nil {
		return fmt.Errorf("mim: export: %w", err)
	}
	return nil
}
