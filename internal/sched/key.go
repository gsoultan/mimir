package sched

import (
	"fmt"
	"math/bits"
	"reflect"
	"sync/atomic"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/internal/graph"
)

// A schedule is planned from the structure of its graph alone: the ops,
// dtypes, shapes and attributes of the nodes to compute and how they
// connect, and of the realized nodes they read, their views, buffer
// lengths, devices and which of them share memory, together with the
// targets and the destinations they may be written over. walk writes all
// of that as a key, a sequence of words, while it numbers the nodes, so
// that a schedule of a graph with the same key can replay the program
// compiled for the first (see program.go and cache.go) instead of
// planning again.
//
// The key leaves out Attrs.Value and Attrs.Seed, the values of constants
// and the seeds of random numbers. They change no plan: a program takes
// them from the graph each time it runs, as kernel arguments, so a
// training step whose learning rate or dropout seed changes still replays
// its program. Every other field of Attrs is part of the key, found by
// reflection once, so a field added for a new op shapes the key without
// changes here.

// walks counts the walks, so that a node marked by an earlier walk reads
// as unmarked.
var walks atomic.Uint64

// walker walks the graph of one schedule. Its slices are reused from walk
// to walk, so that a schedule replayed from the cache allocates nothing
// to find its program. It is guarded by mu.
type walker struct {
	walk   uint64
	be     backend.Backend
	order  []*graph.Node // the unrealized nodes, each after its inputs
	leaves []*graph.Node // the realized nodes the schedule reads or writes over, by number
	mems   int32         // storages numbered
	key    []uint64
	stack  []frame

	gifts []gift // the donated leaves whose memory the schedule takes
}

// gift is a donated leaf whose memory a schedule takes (see donate.go):
// its number, and the index of the key word that says so.
type gift struct{ leaf, at int32 }

type frame struct {
	n    *graph.Node
	next int
}

// Key words start with a tag that says what follows.
const (
	tagNode uint64 = iota + 1
	tagLeaf
	tagTargets
	tagNone
)

// push appends w to the key.
func (w *walker) push(v uint64) { w.key = append(w.key, v) }

// hashKey returns a hash of key, mixing four words at a time in
// independent lanes.
func hashKey(key []uint64) uint64 {
	const m = 0x9e3779b97f4a7c15
	a, b, c, d := uint64(1), uint64(2), uint64(3), uint64(len(key))
	i := 0
	for ; i+4 <= len(key); i += 4 {
		a = bits.RotateLeft64(a^key[i], 29) * m
		b = bits.RotateLeft64(b^key[i+1], 29) * m
		c = bits.RotateLeft64(c^key[i+2], 29) * m
		d = bits.RotateLeft64(d^key[i+3], 29) * m
	}
	for ; i < len(key); i++ {
		a = bits.RotateLeft64(a^key[i], 29) * m
	}
	h := (a ^ bits.RotateLeft64(b, 16) ^ bits.RotateLeft64(c, 32) ^ bits.RotateLeft64(d, 48)) * m
	return h ^ h>>32
}

// pushInts appends the length of s and its elements to the key.
func (w *walker) pushInts(s []int) {
	w.push(uint64(len(s)))
	for _, v := range s {
		w.push(uint64(v))
	}
}

// run numbers the unrealized nodes that targets need, each after its
// inputs, and the realized nodes they read, and
// writes the key of the schedule that computes targets on be and may
// write them over dsts, which is nil or holds a destination, or nil, per
// target. It returns ErrOverwritten if a node it needs is stale. It
// walks iteratively, so deep graphs cannot overflow the stack.
func (w *walker) run(be backend.Backend, targets, dsts []*graph.Node) error {
	w.walk, w.be = walks.Add(1), be
	w.gifts = w.gifts[:0]
	w.order, w.leaves, w.mems = w.order[:0], w.leaves[:0], 0
	w.key = w.key[:0]
	for _, t := range targets {
		if t.Stale() {
			return stale(t)
		}
		if t.Realized() || t.Mark.Walk == w.walk {
			continue
		}
		t.Mark = graph.Mark{Walk: w.walk}
		w.stack = append(w.stack[:0], frame{n: t})
		for len(w.stack) > 0 {
			f := &w.stack[len(w.stack)-1]
			if f.next < len(f.n.Inputs) {
				in := f.n.Inputs[f.next]
				f.next++
				switch {
				case in.Stale():
					return stale(in)
				case in.Mark.Walk == w.walk:
				case in.Realized():
					w.leaf(in)
				default:
					in.Mark = graph.Mark{Walk: w.walk}
					w.stack = append(w.stack, frame{n: in})
				}
				continue
			}
			w.node(f.n)
			w.stack = w.stack[:len(w.stack)-1]
		}
	}
	w.push(tagTargets<<56 | uint64(len(targets)))
	for _, t := range targets {
		w.push(w.ref(t))
	}
	for i := range targets {
		switch d := destination(dsts, i); {
		case d == nil || !d.Realized() || d.Stale():
			w.push(tagNone << 56)
		default:
			w.push(w.ref(d))
		}
	}
	return nil
}

// forget drops the nodes of the last walk, so that the walker keeps no
// graph alive.
func (w *walker) forget() {
	clear(w.order)
	clear(w.leaves)
	clear(w.stack[:cap(w.stack)])
	w.order, w.leaves, w.stack, w.be = w.order[:0], w.leaves[:0], w.stack[:0], nil
}

// destination returns the destination of target i, or nil.
func destination(dsts []*graph.Node, i int) *graph.Node {
	if dsts == nil {
		return nil
	}
	return dsts[i]
}

func stale(n *graph.Node) error {
	if n.Mem.Donated {
		return fmt.Errorf("sched: %s%v: %w: %w", n.DType, n.Shape, graph.ErrDonated, graph.ErrOverwritten)
	}
	return fmt.Errorf("sched: %s%v: %w", n.DType, n.Shape, graph.ErrOverwritten)
}

// ref returns the word that names n, which the walk has numbered unless
// it is realized: its number in order, or, with the top bit set, as a
// leaf.
func (w *walker) ref(n *graph.Node) uint64 {
	if n.Realized() {
		if n.Mark.Walk != w.walk {
			w.leaf(n)
		}
		return 1<<63 | uint64(n.Mark.Num)
	}
	return uint64(n.Mark.Num)
}

// node numbers the unrealized node n, whose inputs are numbered, and
// writes it to the key.
func (w *walker) node(n *graph.Node) {
	n.Mark.Num = int32(len(w.order))
	w.order = append(w.order, n)
	w.push(tagNode<<56 | uint64(n.Op)<<40 | uint64(n.DType)<<32 | uint64(len(n.Inputs))<<16 | uint64(len(n.Shape)))
	for _, d := range n.Shape {
		w.push(uint64(d))
	}
	for _, in := range n.Inputs {
		w.push(w.ref(in))
	}
	if keyAttrs != nil {
		keyAttrs(w, unsafe.Pointer(&n.Attrs))
	}
}

// leaf numbers the realized node n and writes it to the key: what a plan
// reads of it, which is its dtype, shape and view, the length of its
// buffer, whether it lives on the schedule's device, whether the
// schedule may take its memory (see donate.go), and which other leaves
// share its storage.
func (w *walker) leaf(n *graph.Node) {
	n.Mark = graph.Mark{Walk: w.walk, Num: int32(len(w.leaves))}
	w.leaves = append(w.leaves, n)
	m := n.Mem
	if m.Mark.Walk != w.walk {
		m.Mark = graph.Mark{Walk: w.walk, Num: w.mems}
		w.mems++
	}
	var here, donated uint64
	if m.Dev == w.be {
		here = 1
	}
	if donates(n, w.be) {
		donated = 1
		w.gifts = append(w.gifts, gift{n.Mark.Num, int32(len(w.key))})
	}
	w.push(tagLeaf<<56 | donated<<49 | here<<48 | uint64(n.DType)<<32 | uint64(len(n.Shape)))
	for _, d := range n.Shape {
		w.push(uint64(d))
	}
	w.pushInts(n.View.Strides)
	w.push(uint64(n.View.Offset))
	w.push(uint64(m.Buf.Len()))
	w.push(uint64(m.Mark.Num))
}

// argFields are the fields of Attrs that a program takes from the graph
// each time it runs, as kernel arguments, and so leaves out of the key.
var argFields = map[string]bool{"Value": true, "Seed": true}

// keyAttrs writes the fields of an Attrs at p that shape a schedule. It
// is nil, which turns the cache off, if Attrs has a field of a kind the
// key cannot hold, such as a pointer.
var keyAttrs = func() encoder {
	enc, err := encoderOf(reflect.TypeFor[backend.Attrs](), argFields)
	if err != nil {
		return nil
	}
	return enc
}()

// encoder writes the value at p to a key.
type encoder func(w *walker, p unsafe.Pointer)

// encoderOf returns the encoder of values of type t, leaving out the
// fields of a struct named in skip. Values of the same type have the
// same words exactly when they are equal, floats bit for bit.
func encoderOf(t reflect.Type, skip map[string]bool) (encoder, error) {
	switch t.Kind() {
	case reflect.Bool, reflect.Uint8, reflect.Int8:
		return func(w *walker, p unsafe.Pointer) { w.push(uint64(*(*uint8)(p))) }, nil
	case reflect.Int16, reflect.Uint16:
		return func(w *walker, p unsafe.Pointer) { w.push(uint64(*(*uint16)(p))) }, nil
	case reflect.Int32, reflect.Uint32, reflect.Float32:
		return func(w *walker, p unsafe.Pointer) { w.push(uint64(*(*uint32)(p))) }, nil
	case reflect.Int, reflect.Uint, reflect.Int64, reflect.Uint64, reflect.Uintptr, reflect.Float64:
		if t.Size() != 8 {
			break
		}
		return func(w *walker, p unsafe.Pointer) { w.push(*(*uint64)(p)) }, nil
	case reflect.String:
		return func(w *walker, p unsafe.Pointer) {
			s := *(*string)(p)
			w.push(uint64(len(s)))
			for i := 0; i < len(s); i += 8 {
				var v uint64
				for j := i; j < min(i+8, len(s)); j++ {
					v = v<<8 | uint64(s[j])
				}
				w.push(v)
			}
		}, nil
	case reflect.Slice:
		if t.Elem().Kind() == reflect.Int && t.Elem().Size() == 8 {
			return func(w *walker, p unsafe.Pointer) { w.pushInts(*(*[]int)(p)) }, nil
		}
		elem, err := encoderOf(t.Elem(), nil)
		if err != nil {
			return nil, err
		}
		size := t.Elem().Size()
		return func(w *walker, p unsafe.Pointer) {
			s := (*sliceHeader)(p)
			w.push(uint64(s.len))
			for i := range s.len {
				elem(w, unsafe.Add(s.data, uintptr(i)*size))
			}
		}, nil
	case reflect.Array:
		elem, err := encoderOf(t.Elem(), nil)
		if err != nil {
			return nil, err
		}
		n, size := t.Len(), t.Elem().Size()
		return func(w *walker, p unsafe.Pointer) {
			for i := range n {
				elem(w, unsafe.Add(p, uintptr(i)*size))
			}
		}, nil
	case reflect.Struct:
		// A word with a bit per field says which fields are not zero,
		// and only those follow, so that a node whose attributes are
		// mostly empty, as most are, costs a word or two.
		type field struct {
			off  uintptr
			enc  encoder
			zero func(unsafe.Pointer) bool
		}
		var fields []field
		for f := range t.Fields() {
			if skip[f.Name] {
				continue
			}
			enc, err := encoderOf(f.Type, nil)
			if err != nil {
				return nil, fmt.Errorf("field %s: %w", f.Name, err)
			}
			fields = append(fields, field{f.Offset, enc, zeroOf(f.Type)})
		}
		if len(fields) > 64 {
			return nil, fmt.Errorf("sched: no key for %s: %d fields", t, len(fields))
		}
		size := t.Size()
		return func(w *walker, p unsafe.Pointer) {
			// Most nodes' attributes are all zero bytes, which says every
			// field is zero at a glance, without asking each.
			if allZero(unsafe.Slice((*byte)(p), size)) {
				w.push(0)
				return
			}
			at := len(w.key)
			w.push(0)
			var present uint64
			for i, f := range fields {
				if q := unsafe.Add(p, f.off); !f.zero(q) {
					present |= 1 << i
					f.enc(w, q)
				}
			}
			w.key[at] = present
		}, nil
	}
	return nil, fmt.Errorf("sched: no key for %s", t)
}

// allZero reports whether every byte of b is zero.
func allZero(b []byte) bool {
	for len(b) >= 8 {
		if *(*uint64)(unsafe.Pointer(&b[0])) != 0 {
			return false
		}
		b = b[8:]
	}
	for _, c := range b {
		if c != 0 {
			return false
		}
	}
	return true
}

// zeroOf returns a function that reports whether the value of type t at
// p is zero, or, for an array or struct, always false, so that it is
// written whole.
func zeroOf(t reflect.Type) func(unsafe.Pointer) bool {
	switch t.Kind() {
	case reflect.Slice:
		return func(p unsafe.Pointer) bool { return (*sliceHeader)(p).len == 0 }
	case reflect.String:
		return func(p unsafe.Pointer) bool { return len(*(*string)(p)) == 0 }
	case reflect.Array, reflect.Struct:
		return func(unsafe.Pointer) bool { return false }
	}
	switch t.Size() {
	case 1:
		return func(p unsafe.Pointer) bool { return *(*uint8)(p) == 0 }
	case 2:
		return func(p unsafe.Pointer) bool { return *(*uint16)(p) == 0 }
	case 4:
		return func(p unsafe.Pointer) bool { return *(*uint32)(p) == 0 }
	default:
		return func(p unsafe.Pointer) bool { return *(*uint64)(p) == 0 }
	}
}

// sliceHeader is the layout of a slice.
type sliceHeader struct {
	data     unsafe.Pointer
	len, cap int
}

// AppendAttrs appends to key the words that a schedule's key holds of
// a: those of every field but the ones a program takes as arguments,
// Value and Seed. It reports false, appending nothing, if the schedule
// cache is off for want of an encoding of Attrs.
func AppendAttrs(key []uint64, a *backend.Attrs) ([]uint64, bool) {
	if keyAttrs == nil {
		return key, false
	}
	w := walker{key: key}
	keyAttrs(&w, unsafe.Pointer(a))
	return w.key, true
}

// HashKey returns a hash of key, as the schedule cache hashes its keys.
func HashKey(key []uint64) uint64 { return hashKey(key) }
