package onnx

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/tensor"
)

var (
	// ErrFormat reports a malformed file.
	ErrFormat = errors.New("onnx: invalid file")
	// ErrUnsupported reports a model that is valid but uses an operator,
	// an attribute value, a data type or an opset that mimir does not
	// run. An *UnsupportedError carries the details.
	ErrUnsupported = errors.New("onnx: unsupported")
)

// UnsupportedError reports what of a model mimir does not run. It matches
// ErrUnsupported.
type UnsupportedError struct {
	Node   string // the node's name, if it has one
	Domain string // the operator's domain, "" for the default ai.onnx
	Op     string // the operator, such as "Loop"
	Detail string // what is unsupported, when it is not the operator itself
}

func (e *UnsupportedError) Error() string {
	op := e.Op
	if e.Domain != "" {
		op = e.Domain + "." + op
	}
	s := "onnx: unsupported operator " + op
	if e.Node != "" {
		s += " (node " + strconv.Quote(e.Node) + ")"
	}
	if e.Detail != "" {
		s += ": " + e.Detail
	}
	return s
}

// Is reports whether target is ErrUnsupported.
func (e *UnsupportedError) Is(target error) bool { return target == ErrUnsupported }

// DataType is the element type of an ONNX tensor, TensorProto.DataType.
type DataType int32

// The data types of ONNX. mimir computes with the float types and the
// integer types narrower than 64 bits, and with Int64 and Bool as long as
// their values fit in an int32 (see ValueInfo.DType).
const (
	Undefined DataType = 0
	Float     DataType = 1
	Uint8     DataType = 2
	Int8      DataType = 3
	Uint16    DataType = 4
	Int16     DataType = 5
	Int32     DataType = 6
	Int64     DataType = 7
	String    DataType = 8
	Bool      DataType = 9
	Float16   DataType = 10
	Double    DataType = 11
	Uint32    DataType = 12
	Uint64    DataType = 13
	Complex64 DataType = 14
	BFloat16  DataType = 16
)

var dataTypeNames = map[DataType]string{
	Undefined: "undefined", Float: "float", Uint8: "uint8", Int8: "int8",
	Uint16: "uint16", Int16: "int16", Int32: "int32", Int64: "int64",
	String: "string", Bool: "bool", Float16: "float16", Double: "double",
	Uint32: "uint32", Uint64: "uint64", Complex64: "complex64", BFloat16: "bfloat16",
}

func (t DataType) String() string {
	if s, ok := dataTypeNames[t]; ok {
		return s
	}
	return "DataType(" + strconv.Itoa(int(t)) + ")"
}

// IsFloat reports whether t is a floating-point type.
func (t DataType) IsFloat() bool {
	return t == Float || t == Float16 || t == BFloat16 || t == Double
}

// DType returns the mimir dtype that holds values of t: Float32 for
// float and double, Float16 and BFloat16 for themselves, and Int32 for
// the integer types and bool. It reports false for the types mimir does
// not compute with: strings, complex numbers, 8-bit floats and 4-bit
// integers.
func (t DataType) DType() (tensor.DType, bool) {
	switch t {
	case Float, Double:
		return tensor.Float32, true
	case Float16:
		return tensor.Float16, true
	case BFloat16:
		return tensor.BFloat16, true
	case Uint8, Int8, Uint16, Int16, Int32, Int64, Uint32, Uint64, Bool:
		return tensor.Int32, true
	}
	return backend.Invalid, false
}

// size returns the bytes of an element of t in raw_data, or 0 for a type
// mimir does not read.
func (t DataType) size() int {
	switch t {
	case Uint8, Int8, Bool:
		return 1
	case Uint16, Int16, Float16, BFloat16:
		return 2
	case Float, Int32, Uint32:
		return 4
	case Int64, Uint64, Double:
		return 8
	}
	return 0
}

// Dim is one dimension of a declared shape: a size, or, when Size is -1,
// a symbolic dimension that Name may name, such as "batch".
type Dim struct {
	Size int
	Name string
}

func (d Dim) String() string {
	switch {
	case d.Size >= 0:
		return strconv.Itoa(d.Size)
	case d.Name != "":
		return d.Name
	}
	return "?"
}

// ValueInfo describes an input or output of a model.
type ValueInfo struct {
	Name string
	Type DataType
	// Shape is the declared shape, valid when Ranked: a model may leave
	// the rank of a value unknown.
	Shape  []Dim
	Ranked bool
}

// DType returns the mimir dtype of the value's tensors (see
// DataType.DType).
func (v ValueInfo) DType() tensor.DType {
	dt, _ := v.Type.DType()
	return dt
}

func (v ValueInfo) String() string {
	s := v.Name + ": " + v.Type.String()
	if v.Ranked {
		s += fmt.Sprint(v.Shape)
	}
	return s
}
