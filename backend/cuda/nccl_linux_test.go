//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// TestNCCLLoads checks the bindings against libnccl where it is
// installed, which needs no GPU.
func TestNCCLLoads(t *testing.T) {
	if err := loadNCCL(); err != nil {
		t.Skipf("no NCCL: %v", err)
	}
	if msg := ncclGetErrorString(ncclSuccess); msg == "" {
		t.Error("ncclGetErrorString returned nothing")
	}
	if err := checkNCCL(2, "test"); !errors.Is(err, backend.ErrInternal) || !strings.Contains(err.Error(), "test") {
		t.Errorf("checkNCCL: %v", err)
	}
}

// ncclDevices returns a backend for each CUDA device, skipping t unless
// there are two or more and NCCL loads.
func ncclDevices(t *testing.T) []*Backend {
	t.Helper()
	n := DeviceCount()
	if n < 2 {
		t.Skipf("%d CUDA devices; NCCL needs 2 or more", n)
	}
	if err := loadNCCL(); err != nil {
		t.Skipf("no NCCL: %v", err)
	}
	devs := make([]*Backend, n)
	for i := range devs {
		var err error
		if devs[i], err = NewDevice(i); err != nil {
			t.Fatal(err)
		}
	}
	return devs
}

func floatBytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func TestNCCLCollectives(t *testing.T) {
	devs := ncclDevices(t)
	ctx := t.Context()
	ranks := len(devs)
	group := make([]backend.Backend, ranks)
	for i, d := range devs {
		group[i] = d
	}
	comm, err := devs[0].Group(group)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range devs {
		if !d.Asynchronous() {
			t.Error("device of an open group is synchronous")
		}
	}
	for _, k := range []int{1, 7, 100003} {
		t.Run(fmt.Sprint(k), func(t *testing.T) {
			alloc := func(r, n int) backend.Buffer {
				b, err := devs[r].Alloc(backend.Float32, n)
				if err != nil {
					t.Fatal(err)
				}
				return b
			}
			vals := make([][]float32, ranks)
			in, inPlace, reduced, scattered, gathered, cast := make([]backend.Buffer, ranks), make([]backend.Buffer, ranks),
				make([]backend.Buffer, ranks), make([]backend.Buffer, ranks), make([]backend.Buffer, ranks), make([]backend.Buffer, ranks)
			root := ranks - 1
			for r := range ranks {
				vals[r] = make([]float32, k*ranks)
				for i := range vals[r] {
					vals[r][i] = float32((i+3*r)%11) - 5
				}
				in[r], inPlace[r] = alloc(r, k*ranks), alloc(r, k*ranks)
				for _, b := range []backend.Buffer{in[r], inPlace[r]} {
					if err := devs[r].Upload(ctx, b, floatBytes(vals[r])); err != nil {
						t.Fatal(err)
					}
				}
				scattered[r], gathered[r], cast[r] = alloc(r, k), alloc(r, k*ranks*ranks), alloc(r, k*ranks)
			}
			reduced[root] = alloc(root, k*ranks)
			castIn := make([]backend.Buffer, ranks)
			castIn[root] = in[root]
			cs := []backend.Collective{
				{Kind: backend.AllReduce, Reduce: backend.ReduceMax, In: inPlace, Out: inPlace},
				{Kind: backend.Reduce, Reduce: backend.ReduceSum, Root: root, In: in, Out: reduced},
				{Kind: backend.ReduceScatter, Reduce: backend.ReduceMean, In: in, Out: scattered},
				{Kind: backend.AllGather, In: in, Out: gathered},
				{Kind: backend.Broadcast, Root: root, In: castIn, Out: cast},
			}
			if err := comm.Run(ctx, cs); err != nil {
				t.Fatal(err)
			}
			sum, top := slices.Clone(vals[0]), slices.Clone(vals[0])
			for _, v := range vals[1:] {
				for i, x := range v {
					sum[i] += x
					top[i] = max(top[i], x)
				}
			}
			check := func(what string, r int, b backend.Buffer, want []float32) {
				t.Helper()
				raw := make([]byte, 4*b.Len())
				if err := devs[r].Download(ctx, raw, b); err != nil {
					t.Fatal(err)
				}
				for i, w := range want {
					if got := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])); math.Abs(float64(got-w)) > 1e-6 {
						t.Errorf("%s of rank %d, element %d: %v, want %v", what, r, i, got, w)
						return
					}
				}
			}
			for r := range ranks {
				check("all-reduce max", r, inPlace[r], top)
				mean := make([]float32, k)
				for i := range mean {
					mean[i] = sum[r*k+i] / float32(ranks)
				}
				check("reduce-scatter mean", r, scattered[r], mean)
				check("all-gather", r, gathered[r], slices.Concat(vals...))
				check("broadcast", r, cast[r], vals[root])
			}
			check("reduce", root, reduced[root], sum)
		})
	}
	if err := comm.Close(); err != nil {
		t.Fatal(err)
	}
	for _, d := range devs {
		if d.Asynchronous() {
			t.Error("device stays asynchronous after its group closed")
		}
	}
}
