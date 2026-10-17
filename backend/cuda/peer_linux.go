//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"context"
	"fmt"
	"runtime"

	"github.com/gsoultan/mimir/backend"
)

var _ backend.PeerCopier = (*Backend)(nil)

// eventDisableTiming is CU_EVENT_DISABLE_TIMING: the events order streams
// and time nothing.
const eventDisableTiming = 2

// CopiesFrom reports whether from is another CUDA device of the process.
func (b *Backend) CopiesFrom(from backend.Backend) bool {
	f, ok := from.(*Backend)
	return ok && f != b
}

// CopyPeer queues a copy of src, a buffer of the CUDA device from, into
// dst on b's stream (see backend.PeerCopier). b's stream waits for an
// event recorded on from's after the work queued there so far, and
// from's stream waits for one recorded on b's after the copy, so from's
// later work, such as writing over src or freeing it, runs after the copy.
// The copy goes between the devices directly where they can reach each
// other's memory, and through the host otherwise, but the host never waits.
func (b *Backend) CopyPeer(ctx context.Context, dst backend.Buffer, from backend.Backend, src backend.Buffer) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !b.CopiesFrom(from) {
		return fmt.Errorf("cuda: copy from a %s device: %w", from.Name(), backend.ErrDevice)
	}
	f := from.(*Backend)
	cd, err := b.own(dst)
	if err != nil {
		return err
	}
	cs, err := f.own(src)
	if err != nil {
		return err
	}
	if dst.DType() != src.DType() || cd.size() != cs.size() {
		return fmt.Errorf("cuda: copy %d x %s into %d x %s: %w", src.Len(), src.DType(), dst.Len(), dst.DType(), backend.ErrShape)
	}
	// The devices' locks are taken in the order of the devices, as a
	// group's collectives take them.
	first, second := b, f
	if f.dev < b.dev {
		first, second = f, b
	}
	first.mu.Lock()
	defer first.mu.Unlock()
	second.mu.Lock()
	defer second.mu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	f.share(b)
	b.share(f)
	cuCtxSetCurrent(f.ctx)
	ready, err := f.record()
	if err != nil {
		return err
	}
	cuCtxSetCurrent(b.ctx)
	r := cuStreamWaitEvent(b.stream, ready, 0)
	cuEventDestroy(ready)
	if r != cudaSuccess {
		return check(r, "wait for the source device")
	}
	if r := cuMemcpyPeerAsync(cd.ptr, b.ctx, cs.ptr, f.ctx, uintptr(cd.size()), b.stream); r != cudaSuccess {
		return check(r, "copy between devices")
	}
	done, err := b.record()
	if err != nil {
		return err
	}
	cuCtxSetCurrent(f.ctx)
	r = cuStreamWaitEvent(f.stream, done, 0)
	cuEventDestroy(done)
	return check(r, "wait for the copy")
}

// record returns an event recorded on b's stream after the work queued
// so far. b's context is current; the caller destroys the event, which
// it may do as soon as a stream waits for it.
func (b *Backend) record() (uintptr, error) {
	var ev uintptr
	if r := cuEventCreate(&ev, eventDisableTiming); r != cudaSuccess {
		return 0, check(r, "create event")
	}
	if r := cuEventRecord(ev, b.stream); r != cudaSuccess {
		cuEventDestroy(ev)
		return 0, check(r, "record event")
	}
	return ev, nil
}

// share lets peer read and write the allocations of b's memory pool, if
// it can reach b's memory, so that copies between them go directly
// rather than through the host; pool allocations are reachable from
// their own device alone until then. It settles each peer once. Both
// devices' locks are held and the OS thread is locked.
func (b *Backend) share(peer *Backend) {
	if !b.pools || b.peers[peer.dev] {
		return
	}
	if b.peers == nil {
		b.peers = map[int32]bool{}
	}
	b.peers[peer.dev] = true
	var can int32
	if cuDeviceCanAccessPeer(&can, peer.dev, b.dev) != cudaSuccess || can == 0 {
		return
	}
	var pool uintptr
	if cuDeviceGetDefaultMemPool(&pool, b.dev) != cudaSuccess {
		return
	}
	cuCtxSetCurrent(b.ctx)
	desc := memAccessDesc{locationType: memLocationDevice, device: peer.dev, flags: memAccessReadWrite}
	// Without access the copy goes through the host, as between devices
	// that cannot reach each other.
	_ = cuMemPoolSetAccess(pool, &desc, 1)
}
