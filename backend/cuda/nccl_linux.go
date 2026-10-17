//go:build linux && (amd64 || arm64) && !nocuda

package cuda

import (
	"cmp"
	"context"
	"fmt"
	"runtime"
	"slices"
	"sync"

	"github.com/ebitengine/purego"

	"github.com/gsoultan/mimir/backend"
)

// Collectives over several GPUs of one process run on NCCL, loaded with
// purego like the driver. A group's communicators come from
// ncclCommInitAll, one per device, and each collective runs on the
// devices' own streams, after the kernels queued on them, all of a Run
// launched between ncclGroupStart and ncclGroupEnd. RCCL, AMD's port of
// NCCL, has the same API, so a ROCm backend can bind it the same way.

var (
	_ backend.Grouper      = (*Backend)(nil)
	_ backend.Asynchronous = (*Backend)(nil)
	_ backend.Communicator = (*communicator)(nil)
)

// DeviceCount returns the number of CUDA devices, or 0 if the driver
// cannot be loaded.
func DeviceCount() int {
	if loadDriver() != nil {
		return 0
	}
	var n int32
	if cuDeviceGetCount(&n) != cudaSuccess {
		return 0
	}
	return int(n)
}

// Asynchronous reports whether a group holds the device (see Group):
// then a schedule returns once its kernels are queued, and their errors
// surface at the next Sync, Upload or Download, so that the host can
// queue work for the group's other devices meanwhile.
func (b *Backend) Asynchronous() bool { return b.groups.Load() > 0 }

// ncclResult is an ncclResult_t.
type ncclResult int32

// NCCL results, data types and reduction operations, as nccl.h numbers
// them.
const (
	ncclSuccess ncclResult = 0

	ncclInt32    = 2
	ncclFloat16  = 6
	ncclFloat32  = 7
	ncclBfloat16 = 9

	ncclSum = 0
	ncclMax = 2
	ncclMin = 3
	ncclAvg = 4
)

// Loaded functions of NCCL, named as in nccl.h.
var (
	ncclOnce sync.Once
	ncclErr  error

	ncclCommInitAll       func(comms *uintptr, n int32, devices *int32) ncclResult
	ncclCommDestroy       func(comm uintptr) ncclResult
	ncclCommGetAsyncError func(comm uintptr, r *ncclResult) ncclResult
	ncclGetErrorString    func(r ncclResult) string
	ncclGroupStart        func() ncclResult
	ncclGroupEnd          func() ncclResult
	ncclAllReduce         func(send, recv uint64, count uintptr, dt, op int32, comm, stream uintptr) ncclResult
	ncclReduce            func(send, recv uint64, count uintptr, dt, op, root int32, comm, stream uintptr) ncclResult
	ncclBroadcast         func(send, recv uint64, count uintptr, dt, root int32, comm, stream uintptr) ncclResult
	ncclAllGather         func(send, recv uint64, count uintptr, dt int32, comm, stream uintptr) ncclResult
	ncclReduceScatter     func(send, recv uint64, count uintptr, dt, op int32, comm, stream uintptr) ncclResult
)

// loadNCCL loads libnccl once.
func loadNCCL() error {
	ncclOnce.Do(func() {
		lib, err := purego.Dlopen("libnccl.so.2", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			ncclErr = err
			return
		}
		for _, f := range []struct {
			fptr any
			name string
		}{
			{&ncclCommInitAll, "ncclCommInitAll"},
			{&ncclCommDestroy, "ncclCommDestroy"},
			{&ncclCommGetAsyncError, "ncclCommGetAsyncError"},
			{&ncclGetErrorString, "ncclGetErrorString"},
			{&ncclGroupStart, "ncclGroupStart"},
			{&ncclGroupEnd, "ncclGroupEnd"},
			{&ncclAllReduce, "ncclAllReduce"},
			{&ncclReduce, "ncclReduce"},
			{&ncclBroadcast, "ncclBroadcast"},
			{&ncclAllGather, "ncclAllGather"},
			{&ncclReduceScatter, "ncclReduceScatter"},
		} {
			sym, err := purego.Dlsym(lib, f.name)
			if err != nil {
				ncclErr = err
				return
			}
			purego.RegisterFunc(f.fptr, sym)
		}
	})
	return ncclErr
}

// checkNCCL returns nil if r is success and otherwise an error naming
// what failed.
func checkNCCL(r ncclResult, what string) error {
	if r == ncclSuccess {
		return nil
	}
	return fmt.Errorf("cuda: NCCL %s: %s: %w", what, ncclGetErrorString(r), backend.ErrInternal)
}

// Group returns a Communicator over devices, distinct CUDA devices of
// this process, made with ncclCommInitAll. While it is open the devices
// are asynchronous (see Asynchronous). It needs libnccl.so.2, as the
// nvidia/cuda images and the nvidia-nccl-cu12 wheel install it, and
// reports ErrUnavailable without it.
func (b *Backend) Group(devices []backend.Backend) (backend.Communicator, error) {
	if len(devices) == 0 {
		return nil, fmt.Errorf("cuda: group of no devices: %w", backend.ErrShape)
	}
	c := &communicator{devices: make([]*Backend, len(devices))}
	ordinals := make([]int32, len(devices))
	for i, d := range devices {
		cb, ok := d.(*Backend)
		if !ok {
			return nil, fmt.Errorf("cuda: group with a %s device: %w", d.Name(), backend.ErrDevice)
		}
		if slices.Contains(ordinals[:i], cb.dev) {
			return nil, fmt.Errorf("cuda: device %d appears twice in a group: %w", cb.dev, backend.ErrDevice)
		}
		c.devices[i], ordinals[i] = cb, cb.dev
	}
	if err := loadNCCL(); err != nil {
		return nil, fmt.Errorf("cuda: %v: %w", err, ErrUnavailable)
	}
	c.comms = make([]uintptr, len(devices))
	runtime.LockOSThread()
	r := ncclCommInitAll(&c.comms[0], int32(len(devices)), &ordinals[0])
	runtime.UnlockOSThread()
	if err := checkNCCL(r, "create communicators"); err != nil {
		return nil, err
	}
	for _, d := range c.devices {
		d.groups.Add(1)
	}
	return c, nil
}

// communicator runs collectives over CUDA devices on NCCL.
type communicator struct {
	devices []*Backend
	comms   []uintptr // the NCCL communicator of each rank

	mu     sync.Mutex
	closed bool
}

func (c *communicator) Backends() []backend.Backend {
	out := make([]backend.Backend, len(c.devices))
	for i, d := range c.devices {
		out[i] = d
	}
	return out
}

// Close destroys the communicators once the collectives queued so far
// have run, and makes the devices synchronous again when no other group
// holds them.
func (c *communicator) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	var err error
	for i, d := range c.devices {
		if serr := d.Sync(context.Background()); err == nil {
			err = serr
		}
		if derr := checkNCCL(ncclCommDestroy(c.comms[i]), "destroy communicator"); err == nil {
			err = derr
		}
		d.groups.Add(-1)
	}
	return err
}

// Run launches cs on the devices' streams as one NCCL group.
func (c *communicator) Run(ctx context.Context, cs []backend.Collective) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	n := len(c.devices)
	if err := backend.CheckCollectives(cs, n); err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	type launch struct {
		in, out []uint64
		dt      int32
		op      int32
		count   uintptr
	}
	launches := make([]launch, len(cs))
	for i := range cs {
		col := &cs[i]
		l := &launches[i]
		l.in, l.out = make([]uint64, n), make([]uint64, n)
		for r := range n {
			for _, p := range []struct {
				buf backend.Buffer
				ptr *uint64
			}{{col.In[r], &l.in[r]}, {col.Out[r], &l.out[r]}} {
				if p.buf == nil {
					continue
				}
				cb, err := c.devices[r].own(p.buf)
				if err != nil {
					return err
				}
				*p.ptr = cb.ptr
			}
		}
		var dt backend.DType
		for _, b := range slices.Concat(col.In, col.Out) {
			if b != nil {
				dt = b.DType()
				break
			}
		}
		switch dt {
		case backend.Float32:
			l.dt = ncclFloat32
		case backend.Int32:
			l.dt = ncclInt32
		case backend.Float16:
			l.dt = ncclFloat16
		case backend.BFloat16:
			l.dt = ncclBfloat16
		}
		l.op = [...]int32{backend.ReduceSum: ncclSum, backend.ReduceMean: ncclAvg, backend.ReduceMax: ncclMax, backend.ReduceMin: ncclMin}[col.Reduce]
		switch col.Kind {
		case backend.Broadcast:
			l.count = uintptr(col.In[col.Root].Len())
			if l.out[col.Root] == 0 {
				l.out[col.Root] = l.in[col.Root]
			}
		case backend.ReduceScatter:
			l.count = uintptr(col.Out[0].Len())
		default:
			l.count = uintptr(col.In[0].Len())
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("cuda: run on a closed group: %w", backend.ErrInternal)
	}
	for i, comm := range c.comms {
		var async ncclResult
		if err := checkNCCL(ncclCommGetAsyncError(comm, &async), "query"); err != nil {
			return err
		}
		if err := checkNCCL(async, fmt.Sprintf("rank %d", i)); err != nil {
			return err
		}
	}
	// Each device's lock keeps other goroutines off its stream while
	// the group is queued; taking them in the order of the devices keeps
	// two groups over the same devices from waiting for each other. NCCL
	// keeps the state of a group per thread.
	byDevice := slices.SortedFunc(slices.Values(c.devices), func(a, b *Backend) int { return cmp.Compare(a.dev, b.dev) })
	for _, d := range byDevice {
		d.mu.Lock()
		defer d.mu.Unlock()
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	// A collective may write over memory that copies to other devices
	// read.
	for _, d := range byDevice {
		if len(d.reads) > 0 {
			cuCtxSetCurrent(d.ctx)
			if err := d.settleAll(); err != nil {
				return err
			}
		}
	}
	if err := checkNCCL(ncclGroupStart(), "group start"); err != nil {
		return err
	}
	var err error
	for i := range cs {
		col, l := &cs[i], &launches[i]
		for r, d := range c.devices {
			if err != nil {
				break
			}
			comm, s := c.comms[r], d.stream
			var res ncclResult
			switch col.Kind {
			case backend.AllReduce:
				res = ncclAllReduce(l.in[r], l.out[r], l.count, l.dt, l.op, comm, s)
			case backend.Reduce:
				res = ncclReduce(l.in[r], cmp.Or(l.out[r], l.in[r]), l.count, l.dt, l.op, int32(col.Root), comm, s)
			case backend.Broadcast:
				res = ncclBroadcast(cmp.Or(l.in[r], l.out[r]), l.out[r], l.count, l.dt, int32(col.Root), comm, s)
			case backend.AllGather:
				res = ncclAllGather(l.in[r], l.out[r], l.count, l.dt, comm, s)
			case backend.ReduceScatter:
				res = ncclReduceScatter(l.in[r], l.out[r], l.count, l.dt, l.op, comm, s)
			}
			err = checkNCCL(res, col.Kind.String())
		}
	}
	if eerr := checkNCCL(ncclGroupEnd(), "group end"); err == nil {
		err = eerr
	}
	return err
}
