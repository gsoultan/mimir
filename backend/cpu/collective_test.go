package cpu

import (
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// groupOf returns n devices: Streams, but the CPU itself at rank 0 when
// withBackend is set.
func groupOf(n int, withBackend bool) []backend.Backend {
	devs := make([]backend.Backend, n)
	for r := range devs {
		devs[r] = NewStream(1 + r%2)
	}
	if withBackend {
		devs[0] = New()
	}
	return devs
}

func TestCollectivesOfOddSizes(t *testing.T) {
	ctx := t.Context()
	for _, ranks := range []int{1, 2, 3, 5} {
		for _, k := range []int{1, 3, 4099, 3 * collectiveGrain} {
			for _, withBackend := range []bool{false, true} {
				t.Run(fmt.Sprintf("%d/%d/%v", ranks, k, withBackend), func(t *testing.T) {
					devs := groupOf(ranks, withBackend)
					comm, err := devs[ranks-1].(backend.Grouper).Group(devs)
					if err != nil {
						t.Fatal(err)
					}
					in := make([]backend.Buffer, ranks)
					vals := make([][]float32, ranks)
					for r := range ranks {
						vals[r] = make([]float32, k*ranks)
						for i := range vals[r] {
							vals[r][i] = float32((i+3*r)%11) - 5
						}
						in[r] = f32buf(vals[r]...)
					}
					bufs := func(n int) []backend.Buffer {
						out := make([]backend.Buffer, ranks)
						for r := range out {
							out[r], _ = devs[r].Alloc(backend.Float32, n)
						}
						return out
					}
					root := ranks / 2
					reduced, scattered, gathered, cast := bufs(k*ranks), bufs(k), bufs(k*ranks*ranks), bufs(k*ranks)
					reduceOut := make([]backend.Buffer, ranks)
					reduceOut[root] = reduced[root]
					castIn := make([]backend.Buffer, ranks)
					castIn[root] = in[root]
					// AllReduce in place over copies of the inputs, with
					// the other collectives in the same group.
					inPlace := make([]backend.Buffer, ranks)
					for r := range inPlace {
						inPlace[r] = f32buf(vals[r]...)
					}
					cs := []backend.Collective{
						{Kind: backend.AllReduce, Reduce: backend.ReduceMax, In: inPlace, Out: inPlace},
						{Kind: backend.Reduce, Reduce: backend.ReduceSum, Root: root, In: in, Out: reduceOut},
						{Kind: backend.ReduceScatter, Reduce: backend.ReduceMean, In: in, Out: scattered},
						{Kind: backend.AllGather, In: in, Out: gathered},
						{Kind: backend.Broadcast, Root: root, In: castIn, Out: cast},
					}
					if err := comm.Run(ctx, cs); err != nil {
						t.Fatal(err)
					}
					for _, d := range devs {
						if err := d.Sync(ctx); err != nil {
							t.Fatal(err)
						}
					}
					sum, top := slices.Clone(vals[0]), slices.Clone(vals[0])
					for _, v := range vals[1:] {
						for i, x := range v {
							sum[i] += x
							top[i] = max(top[i], x)
						}
					}
					all := slices.Concat(vals...)
					for r := range ranks {
						check := func(what string, b backend.Buffer, want []float32) {
							t.Helper()
							if got := b.(*buffer[float32]).data; !slices.Equal(got, want) {
								t.Errorf("%s of rank %d: %v..., want %v...", what, r, got[:min(4, len(got))], want[:min(4, len(want))])
							}
						}
						check("max", inPlace[r], top)
						mean := make([]float32, k)
						for i := range mean {
							mean[i] = sum[r*k+i] * (1 / float32(ranks))
						}
						check("reduce-scatter", scattered[r], mean)
						check("all-gather", gathered[r], all)
						check("broadcast", cast[r], vals[root])
						if r == root {
							check("reduce", reduced[r], sum)
						}
					}
				})
			}
		}
	}
}

func TestCollectivesOfHalves(t *testing.T) {
	devs := groupOf(2, false)
	comm, err := New().Group(devs)
	if err != nil {
		t.Fatal(err)
	}
	for _, dt := range []backend.DType{backend.BFloat16, backend.Float16} {
		in := make([]backend.Buffer, 2)
		for r := range in {
			in[r], _ = devs[r].Alloc(dt, 5)
			b, _ := bytesOf(in[r])
			for i := range 5 {
				h := f32ToBF16(float32(i) + 0.5)
				if dt == backend.Float16 {
					h = f32ToF16(float32(i) + 0.5)
				}
				b[2*i], b[2*i+1] = byte(h), byte(h>>8)
			}
		}
		if err := comm.Run(t.Context(), []backend.Collective{{Kind: backend.AllReduce, Reduce: backend.ReduceSum, In: in, Out: in}}); err != nil {
			t.Fatal(err)
		}
		if err := devs[1].Sync(t.Context()); err != nil {
			t.Fatal(err)
		}
		out := make([]float32, 5)
		switch b := in[1].(type) {
		case *buffer[bf16]:
			loadHalf(dt, out, bitsOf(b.data), 0, 1)
		case *buffer[f16]:
			loadHalf(dt, out, bitsOf(b.data), 0, 1)
		}
		if !slices.Equal(out, []float32{1, 3, 5, 7, 9}) {
			t.Errorf("%s sum: %v", dt, out)
		}
	}
}

func TestCollectiveErrors(t *testing.T) {
	devs := groupOf(2, false)
	comm, err := devs[0].(backend.Grouper).Group(devs)
	if err != nil {
		t.Fatal(err)
	}
	a, b, c := f32buf(1, 2, 3), f32buf(1, 2, 3), f32buf(1, 2)
	ints := i32buf(1, 2, 3)
	for name, tc := range map[string]struct {
		c    backend.Collective
		want error
	}{
		"one rank":            {backend.Collective{In: []backend.Buffer{a}, Out: []backend.Buffer{a}}, backend.ErrShape},
		"lengths":             {backend.Collective{In: []backend.Buffer{a, c}, Out: []backend.Buffer{a, c}}, backend.ErrShape},
		"dtypes":              {backend.Collective{In: []backend.Buffer{a, ints}, Out: []backend.Buffer{a, ints}}, backend.ErrDType},
		"int mean":            {backend.Collective{Reduce: backend.ReduceMean, In: []backend.Buffer{ints, i32buf(1, 2, 3)}, Out: []backend.Buffer{i32buf(0, 0, 0), i32buf(0, 0, 0)}}, backend.ErrDType},
		"missing output":      {backend.Collective{In: []backend.Buffer{a, b}, Out: []backend.Buffer{a, nil}}, backend.ErrInternal},
		"shared output":       {backend.Collective{In: []backend.Buffer{a, b}, Out: []backend.Buffer{b, a}}, backend.ErrInternal},
		"root out of range":   {backend.Collective{Kind: backend.Broadcast, Root: 2, In: []backend.Buffer{a, nil}, Out: []backend.Buffer{nil, b}}, backend.ErrShape},
		"gather too short":    {backend.Collective{Kind: backend.AllGather, In: []backend.Buffer{a, b}, Out: []backend.Buffer{f32buf(make([]float32, 6)...), c}}, backend.ErrShape},
		"scatter of 3 over 2": {backend.Collective{Kind: backend.ReduceScatter, In: []backend.Buffer{a, b}, Out: []backend.Buffer{c, f32buf(1)}}, backend.ErrShape},
	} {
		if err := comm.Run(t.Context(), []backend.Collective{tc.c}); !errors.Is(err, tc.want) {
			t.Errorf("%s: %v, want %v", name, err, tc.want)
		}
	}
	if _, err := New().Group([]backend.Backend{devs[0], devs[0]}); !errors.Is(err, backend.ErrDevice) {
		t.Errorf("repeated device: %v, want ErrDevice", err)
	}
}

func TestStreamRunsInOrderAndReportsErrorsAtSync(t *testing.T) {
	ctx := t.Context()
	s := NewStream(2)
	x := f32buf(1, 2, 3, 4)
	// Doubling x in place twice must see the first doubling.
	k := &backend.Kernel{Op: backend.OpFused, DType: backend.Float32, Shape: []int{4}, In: []backend.View{backend.Contiguous([]int{4})},
		Attrs: backend.Attrs{Program: []backend.Instr{
			{Op: backend.OpLoad, DType: backend.Float32, Args: []int{0}},
			{Op: backend.OpConst, DType: backend.Float32, Value: 2},
			{Op: backend.OpMul, DType: backend.Float32, Args: []int{0, 1}},
		}}}
	for range 2 {
		k.Attrs.Program[1].Value = 2
		if err := s.Run(ctx, k, []backend.Buffer{x}, []backend.Buffer{x}); err != nil {
			t.Fatal(err)
		}
		// The queued kernel keeps the constant it was queued with.
		k.Attrs.Program[1].Value = 100
	}
	k.Attrs.Program[1].Value = 2
	got := make([]byte, 16)
	if err := s.Download(ctx, got, x); err != nil {
		t.Fatal(err)
	}
	if v := x.(*buffer[float32]).data; !slices.Equal(v, []float32{4, 8, 12, 16}) {
		t.Errorf("got %v", v)
	}
	// An index out of range surfaces at the next Sync, once.
	gather := &backend.Kernel{Op: backend.OpGather, DType: backend.Float32, Shape: []int{1},
		In: []backend.View{backend.Contiguous([]int{4}), backend.Contiguous([]int{1})}}
	out := f32buf(0)
	if err := s.Run(ctx, gather, []backend.Buffer{out}, []backend.Buffer{x, i32buf(9)}); err != nil {
		t.Fatalf("Run: %v, want the error at Sync", err)
	}
	if err := s.Sync(ctx); !errors.Is(err, backend.ErrIndex) {
		t.Errorf("Sync: %v, want ErrIndex", err)
	}
	if err := s.Sync(ctx); err != nil {
		t.Errorf("second Sync: %v", err)
	}
	// Many goroutines queue on one Stream at once.
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			buf := f32buf(1, 1, 1, 1)
			if err := s.Run(ctx, k, []backend.Buffer{buf}, []backend.Buffer{buf}); err != nil {
				t.Error(err)
			}
			s.Free(buf)
		})
	}
	wg.Wait()
	if err := s.Sync(ctx); err != nil {
		t.Error(err)
	}
}
