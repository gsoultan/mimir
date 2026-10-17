//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"math"
	"slices"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// TestBatchesMatchTheirKernels runs a batch of a fused kernel with a
// constant and a product again and again, with a new constant or buffers
// between runs, and checks each run's results against the host's.
func TestBatchesMatchTheirKernels(t *testing.T) {
	if newGPU(t, -1, -1) == nil {
		t.Skip("no Metal device")
	}
	gpuMu.Lock()
	defer gpuMu.Unlock()
	b := gpu
	upload := func(vs []float32) backend.Buffer {
		t.Helper()
		buf, err := b.Alloc(backend.Float32, len(vs))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { b.Free(buf) })
		var data []byte
		for _, v := range vs {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}
		if err := b.Upload(t.Context(), buf, data); err != nil {
			t.Fatal(err)
		}
		return buf
	}
	download := func(buf backend.Buffer) []float32 {
		t.Helper()
		data := make([]byte, 4*buf.Len())
		if err := b.Download(t.Context(), data, buf); err != nil {
			t.Fatal(err)
		}
		vs := make([]float32, buf.Len())
		for i := range vs {
			vs[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		}
		return vs
	}

	xs := []float32{1, -2, 3, -4, 5, -6, 7, -8}
	as, bs := make([]float32, 16), make([]float32, 16)
	for i := range as {
		as[i], bs[i] = float32(i)/4, float32(15-i)/8
	}
	fused := &backend.Kernel{Op: backend.OpFused, DType: backend.Float32, Shape: []int{8},
		In: []backend.View{backend.Contiguous([]int{8})},
		Attrs: backend.Attrs{Program: []backend.Instr{
			{Op: backend.OpLoad, DType: backend.Float32, Args: []int{0}},
			{Op: backend.OpConst, DType: backend.Float32, Value: 2},
			{Op: backend.OpMul, DType: backend.Float32, Args: []int{0, 1}},
		}}}
	product := &backend.Kernel{Op: backend.OpMatMul, DType: backend.Float32, Shape: []int{4, 4},
		In: []backend.View{backend.Contiguous([]int{4, 4}), backend.Contiguous([]int{4, 4})}}
	bt := &backend.Batch{
		Kernels: []*backend.Kernel{fused, product},
		Outs:    [][]backend.Buffer{{upload(make([]float32, 8))}, {upload(make([]float32, 16))}},
		Ins:     [][]backend.Buffer{{upload(xs)}, {upload(as), upload(bs)}},
	}
	for run, c := range []float64{2, 2, 3, 3, 5} {
		fused.Attrs.Program[1].Value = c
		switch run {
		case 3:
			bt.Outs[0][0] = upload(make([]float32, 8))
		case 4:
			bt.Ins[1][0] = upload(as)
		}
		if err := b.RunBatch(t.Context(), bt); err != nil {
			t.Fatal(err)
		}
		if err := b.Sync(t.Context()); err != nil {
			t.Fatal(err)
		}
		for i, v := range download(bt.Outs[0][0]) {
			if want := xs[i] * float32(c); v != want {
				t.Fatalf("run %d: element %d is %v, want %v", run, i, v, want)
			}
		}
		got := download(bt.Outs[1][0])
		for i := range 4 {
			for j := range 4 {
				var want float32
				for k := range 4 {
					want += as[i*4+k] * bs[k*4+j]
				}
				if d := got[i*4+j] - want; d > 1e-4 || d < -1e-4 {
					t.Fatalf("run %d: product element %d,%d is %v, want %v", run, i, j, got[i*4+j], want)
				}
			}
		}
	}
	if st := bt.State.(*batchState); len(st.calls[0].launches) != 1 || !slices.Contains(st.calls[0].args, math.Float64bits(5)) {
		t.Error("the batch does not keep its kernels' plans")
	}
}
