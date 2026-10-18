//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gsoultan/mimir/backend"
)

// TestCopyPeerReadsTheSourceBeforeItIsWritten copies a large buffer to
// another device and at once queues a kernel that writes over the
// source, which must wait for the copy, though the source's stream does
// not wait for it otherwise.
func TestCopyPeerReadsTheSourceBeforeItIsWritten(t *testing.T) {
	if DeviceCount() < 2 {
		t.Skipf("%d CUDA devices; a copy between devices needs 2 or more", DeviceCount())
	}
	ctx := t.Context()
	from, err := NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	to, err := NewDevice(1)
	if err != nil {
		t.Fatal(err)
	}
	const n = 8 << 20 // long enough to copy that a write would overtake it
	src, err := from.Alloc(backend.Float32, n)
	if err != nil {
		t.Fatal(err)
	}
	dst, err := to.Alloc(backend.Float32, n)
	if err != nil {
		t.Fatal(err)
	}
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	if err := from.Upload(ctx, src, floatBytes(ones)); err != nil {
		t.Fatal(err)
	}
	for round := range 3 {
		if err := to.CopyPeer(ctx, dst, from, src); err != nil {
			t.Fatal(err)
		}
		fill := &backend.Kernel{Op: backend.OpConst, DType: backend.Float32, Shape: []int{n}, Attrs: backend.Attrs{Value: float64(round + 2)}}
		if err := from.Run(ctx, fill, []backend.Buffer{src}, nil); err != nil {
			t.Fatal(err)
		}
		got := make([]byte, 4*n)
		if err := to.Download(ctx, got, dst); err != nil {
			t.Fatal(err)
		}
		want := float32(round + 1)
		for i := 0; i < n; i += 4093 {
			if v := math.Float32frombits(binary.LittleEndian.Uint32(got[4*i:])); v != want {
				t.Fatalf("round %d: copy holds %v at %d, want %v: the write overtook the copy", round, v, i, want)
			}
		}
		if err := from.Sync(ctx); err != nil {
			t.Fatal(err)
		}
	}
	if len(from.reads) != 0 {
		t.Errorf("%d buffers still wait for copies after their writes", len(from.reads))
	}
}
