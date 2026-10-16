package gguf

import (
	"fmt"
	"math"
)

// ValueType is the type of a metadata value, GGUF's gguf_type.
type ValueType uint32

// The metadata value types. The values are those GGUF files store.
const (
	Uint8   ValueType = 0
	Int8    ValueType = 1
	Uint16  ValueType = 2
	Int16   ValueType = 3
	Uint32  ValueType = 4
	Int32   ValueType = 5
	Float32 ValueType = 6
	Bool    ValueType = 7
	String  ValueType = 8
	Array   ValueType = 9
	Uint64  ValueType = 10
	Int64   ValueType = 11
	Float64 ValueType = 12

	invalidValue ValueType = 0xffffffff
)

var valueTypeNames = [...]string{"uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "bool", "string", "array", "uint64", "int64", "float64"}

func (t ValueType) String() string {
	if int(t) < len(valueTypeNames) {
		return valueTypeNames[t]
	}
	return fmt.Sprintf("ValueType(%d)", uint32(t))
}

// size returns the bytes of a value of the fixed-size type t, or 0 for a
// string or an array, whose size varies, and for an unknown type.
func (t ValueType) size() int {
	switch t {
	case Uint8, Int8, Bool:
		return 1
	case Uint16, Int16:
		return 2
	case Uint32, Int32, Float32:
		return 4
	case Uint64, Int64, Float64:
		return 8
	}
	return 0
}

// Value is a metadata value: a number, a bool, a string or an array of
// values of one type. It holds the Go value of its type: uint8, int8,
// uint16, int16, uint32, int32, float32, bool, string, uint64, int64 or
// float64, a slice of one of them for an array, or []Value for an array
// of arrays.
type Value struct {
	v any
}

// ValueOf returns a value holding v, which should have the Go type of a
// GGUF value; Write reports any other.
func ValueOf(v any) Value { return Value{v} }

// Interface returns the Go value v holds.
func (v Value) Interface() any { return v.v }

// Type returns the type of v: Array for an array.
func (v Value) Type() ValueType {
	t, _ := typeOf(v.v)
	return t
}

// ElemType returns the type of the elements of an array, Array for an
// array of arrays.
func (v Value) ElemType() ValueType {
	_, e := typeOf(v.v)
	return e
}

// typeOf returns the GGUF type of the Go value x and, for an array, the
// type of its elements; invalidValue for a Go type that has none.
func typeOf(x any) (t, elem ValueType) {
	switch x.(type) {
	case uint8:
		return Uint8, invalidValue
	case int8:
		return Int8, invalidValue
	case uint16:
		return Uint16, invalidValue
	case int16:
		return Int16, invalidValue
	case uint32:
		return Uint32, invalidValue
	case int32:
		return Int32, invalidValue
	case float32:
		return Float32, invalidValue
	case bool:
		return Bool, invalidValue
	case string:
		return String, invalidValue
	case uint64:
		return Uint64, invalidValue
	case int64:
		return Int64, invalidValue
	case float64:
		return Float64, invalidValue
	case []uint8:
		return Array, Uint8
	case []int8:
		return Array, Int8
	case []uint16:
		return Array, Uint16
	case []int16:
		return Array, Int16
	case []uint32:
		return Array, Uint32
	case []int32:
		return Array, Int32
	case []float32:
		return Array, Float32
	case []bool:
		return Array, Bool
	case []string:
		return Array, String
	case []uint64:
		return Array, Uint64
	case []int64:
		return Array, Int64
	case []float64:
		return Array, Float64
	case []Value:
		return Array, Array
	}
	return invalidValue, invalidValue
}

// Len returns the number of elements of an array, and 0 for any other
// value.
func (v Value) Len() int {
	switch x := v.v.(type) {
	case []uint8:
		return len(x)
	case []int8:
		return len(x)
	case []uint16:
		return len(x)
	case []int16:
		return len(x)
	case []uint32:
		return len(x)
	case []int32:
		return len(x)
	case []float32:
		return len(x)
	case []bool:
		return len(x)
	case []string:
		return len(x)
	case []uint64:
		return len(x)
	case []int64:
		return len(x)
	case []float64:
		return len(x)
	case []Value:
		return len(x)
	}
	return 0
}

// Int returns an integer value of any type, and false for any other
// value or a uint64 beyond the range of int64.
func (v Value) Int() (int64, bool) {
	switch x := v.v.(type) {
	case uint8:
		return int64(x), true
	case int8:
		return int64(x), true
	case uint16:
		return int64(x), true
	case int16:
		return int64(x), true
	case uint32:
		return int64(x), true
	case int32:
		return int64(x), true
	case int64:
		return x, true
	case uint64:
		return int64(x), x <= math.MaxInt64
	}
	return 0, false
}

// Uint returns a non-negative integer value of any type, and false for
// any other value.
func (v Value) Uint() (uint64, bool) {
	if x, ok := v.v.(uint64); ok {
		return x, true
	}
	i, ok := v.Int()
	return uint64(i), ok && i >= 0
}

// Float returns a float32 or float64 value, or an integer value converted
// to float64, and false for any other value.
func (v Value) Float() (float64, bool) {
	switch x := v.v.(type) {
	case float32:
		return float64(x), true
	case float64:
		return x, true
	case uint64:
		return float64(x), true
	}
	i, ok := v.Int()
	return float64(i), ok
}

// Bool returns a bool value, and false for any other value.
func (v Value) Bool() (value, ok bool) {
	b, ok := v.v.(bool)
	return b, ok
}

// Text returns a string value, and false for any other value.
func (v Value) Text() (string, bool) {
	s, ok := v.v.(string)
	return s, ok
}

// Strings returns an array of strings, such as a tokenizer's vocabulary,
// and false for any other value. The slice is v's own.
func (v Value) Strings() ([]string, bool) {
	s, ok := v.v.([]string)
	return s, ok
}

// Int32s returns an array of integers of any type of 32 bits or fewer,
// such as a tokenizer's token types, as int32s, and false for any other
// value. The slice is v's own for an array of int32.
func (v Value) Int32s() ([]int32, bool) {
	switch x := v.v.(type) {
	case []int32:
		return x, true
	case []int8:
		return widen(x), true
	case []uint8:
		return widen(x), true
	case []int16:
		return widen(x), true
	case []uint16:
		return widen(x), true
	}
	return nil, false
}

func widen[T int8 | uint8 | int16 | uint16](s []T) []int32 {
	out := make([]int32, len(s))
	for i, x := range s {
		out[i] = int32(x)
	}
	return out
}

// Float32s returns an array of float32, such as a tokenizer's scores, and
// false for any other value. The slice is v's own.
func (v Value) Float32s() ([]float32, bool) {
	s, ok := v.v.([]float32)
	return s, ok
}

// String formats v for printing, an array by its type and length.
func (v Value) String() string {
	if t, e := typeOf(v.v); t == Array {
		return fmt.Sprintf("[%d]%s", v.Len(), e)
	}
	return fmt.Sprint(v.v)
}
