package gguf

import (
	"fmt"

	"github.com/gsoultan/mimir/backend"
)

// Type is the element type of a tensor, ggml's ggml_type.
type Type uint32

// The tensor types of ggml. The values are those GGUF files store.
const (
	F32     Type = 0
	F16     Type = 1
	Q4_0    Type = 2
	Q4_1    Type = 3
	Q5_0    Type = 6
	Q5_1    Type = 7
	Q8_0    Type = 8
	Q8_1    Type = 9
	Q2_K    Type = 10
	Q3_K    Type = 11
	Q4_K    Type = 12
	Q5_K    Type = 13
	Q6_K    Type = 14
	Q8_K    Type = 15
	IQ2_XXS Type = 16
	IQ2_XS  Type = 17
	IQ3_XXS Type = 18
	IQ1_S   Type = 19
	IQ4_NL  Type = 20
	IQ3_S   Type = 21
	IQ2_S   Type = 22
	IQ4_XS  Type = 23
	I8      Type = 24
	I16     Type = 25
	I32     Type = 26
	I64     Type = 27
	F64     Type = 28
	IQ1_M   Type = 29
	BF16    Type = 30
	TQ1_0   Type = 34
	TQ2_0   Type = 35
	MXFP4   Type = 39
	NVFP4   Type = 40
	Q1_0    Type = 41
)

// typeInfo is a type's name, the elements of one of its blocks and the
// bytes of a block, as ggml's type_traits give them.
type typeInfo struct {
	name        string
	size, bytes int
}

var types = map[Type]typeInfo{
	F32: {"f32", 1, 4}, F16: {"f16", 1, 2}, BF16: {"bf16", 1, 2}, F64: {"f64", 1, 8},
	I8: {"i8", 1, 1}, I16: {"i16", 1, 2}, I32: {"i32", 1, 4}, I64: {"i64", 1, 8},
	Q4_0: {"q4_0", 32, 18}, Q4_1: {"q4_1", 32, 20}, Q5_0: {"q5_0", 32, 22}, Q5_1: {"q5_1", 32, 24},
	Q8_0: {"q8_0", 32, 34}, Q8_1: {"q8_1", 32, 36},
	Q2_K: {"q2_K", 256, 84}, Q3_K: {"q3_K", 256, 110}, Q4_K: {"q4_K", 256, 144},
	Q5_K: {"q5_K", 256, 176}, Q6_K: {"q6_K", 256, 210}, Q8_K: {"q8_K", 256, 292},
	IQ2_XXS: {"iq2_xxs", 256, 66}, IQ2_XS: {"iq2_xs", 256, 74}, IQ3_XXS: {"iq3_xxs", 256, 98},
	IQ1_S: {"iq1_s", 256, 50}, IQ4_NL: {"iq4_nl", 32, 18}, IQ3_S: {"iq3_s", 256, 110},
	IQ2_S: {"iq2_s", 256, 82}, IQ4_XS: {"iq4_xs", 256, 136}, IQ1_M: {"iq1_m", 256, 56},
	TQ1_0: {"tq1_0", 256, 54}, TQ2_0: {"tq2_0", 256, 66}, MXFP4: {"mxfp4", 32, 17},
	NVFP4: {"nvfp4", 64, 36}, Q1_0: {"q1_0", 128, 18},
}

// String returns ggml's name of the type, such as "q4_K".
func (t Type) String() string {
	if info, ok := types[t]; ok {
		return info.name
	}
	return fmt.Sprintf("Type(%d)", uint32(t))
}

// BlockSize returns the number of elements in a block of t, 1 for the
// types that are not quantized in blocks, and 0 for a type ggml does not
// define.
func (t Type) BlockSize() int { return types[t].size }

// BlockBytes returns the size in bytes of a block of t, and 0 for a type
// ggml does not define.
func (t Type) BlockBytes() int { return types[t].bytes }

// dtypes are the mimir dtypes of the types that mimir reads.
var dtypes = map[Type]backend.DType{
	F32: backend.Float32, F16: backend.Float16, BF16: backend.BFloat16, I32: backend.Int32,
	Q8_0: backend.Q8_0, Q4_K: backend.Q4_K, Q6_K: backend.Q6_K,
}

// DType returns the mimir dtype whose elements are those of t, laid out
// alike, so that tensor.FromBytes reads a tensor of t in place. It
// reports false for the types mimir does not read.
func (t Type) DType() (backend.DType, bool) {
	dt, ok := dtypes[t]
	return dt, ok
}

// TypeError reports a tensor of a type that mimir does not read, such as
// a Q5_K tensor. It matches backend.ErrDType (tensor.ErrDType) with
// errors.Is.
type TypeError struct {
	Tensor string // the tensor's name
	Type   Type
}

func (e *TypeError) Error() string {
	return fmt.Sprintf("gguf: tensor %q has type %s, which mimir does not read", e.Tensor, e.Type)
}

// Unwrap returns backend.ErrDType.
func (e *TypeError) Unwrap() error { return backend.ErrDType }
