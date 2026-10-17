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
// event recorded on from's after the work queued there so far. from's
// stream goes on with its own work while the copy runs: only its next
// work that writes over src or frees it waits for the copy (see settle).
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
	if err := b.settle(cd.ptr); err != nil {
		return err
	}
	if r := cuMemcpyPeerAsync(cd.ptr, b.ctx, cs.ptr, f.ctx, uintptr(cd.size()), b.stream); r != cudaSuccess {
		return check(r, "copy between devices")
	}
	done, err := b.record()
	if err != nil {
		return err
	}
	f.readBy(cs.ptr, b.dev, done)
	return nil
}

// peerRead is a copy from a buffer of one device to the device dev, whose
// event done is recorded on dev's stream after the copy.
type peerRead struct {
	dev  int32
	done uintptr
}

// readBy notes that a copy to peer reads the memory at ptr, until the
// event done. b.mu is held.
func (b *Backend) readBy(ptr uint64, peer int32, done uintptr) {
	if b.reads == nil {
		b.reads = map[uint64][]peerRead{}
	}
	rs := b.reads[ptr]
	for i := range rs {
		if rs[i].dev == peer {
			// peer's stream runs its copies in order, so the new copy's
			// event follows the old one's.
			cuEventDestroy(rs[i].done)
			rs[i].done = done
			return
		}
	}
	b.reads[ptr] = append(rs, peerRead{peer, done})
}

// settle makes b's stream wait for the copies to other devices that read
// its memory at ptr, before the work queued next, which writes over it or
// frees it. Copies that read from b leave b's stream free to compute
// meanwhile: the source of the shares of a data-parallel batch computes
// its own share while the others' are copied. b.mu is held and b's
// context is current.
func (b *Backend) settle(ptr uint64) error {
	rs, ok := b.reads[ptr]
	if !ok {
		return nil
	}
	delete(b.reads, ptr)
	var err error
	for _, r := range rs {
		if res := cuStreamWaitEvent(b.stream, r.done, 0); res != cudaSuccess && err == nil {
			err = check(res, "wait for a copy to another device")
		}
		cuEventDestroy(r.done)
	}
	return err
}

// settleAll makes b's stream wait for every copy to another device that
// reads its memory, as settle does for one buffer.
func (b *Backend) settleAll() error {
	var err error
	for ptr := range b.reads {
		if serr := b.settle(ptr); err == nil {
			err = serr
		}
	}
	return err
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
