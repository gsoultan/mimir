package tensor

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/backend/cuda"
	"github.com/gsoultan/mimir/backend/metal"
	"github.com/gsoultan/mimir/backend/rocm"
	"github.com/gsoultan/mimir/internal/graph"
	"github.com/gsoultan/mimir/internal/sched"
)

// Device is where tensors live and are computed: the CPU or a GPU. The
// zero Device is the default device.
//
// A tensor uploaded with FromSlice lives on the default device, and one
// computed from others lives where they do; combining tensors on different
// devices fails with ErrDevice. Constants and random tensors, such as
// Zeros and Randn, are computed wherever the tensors they are combined
// with live, or on the default device on their own. To moves a tensor to
// another device.
type Device struct {
	be backend.Backend
}

// ErrDevice reports tensors on different devices combined in one
// operation.
var ErrDevice = backend.ErrDevice

var (
	cpuBackend = cpu.New()

	gpuOnce sync.Once
	gpu     backend.Backend // the GPU backend, or nil
	gpuErr  error
)

// CPU returns the host CPU.
func CPU() Device { return Device{cpuBackend} }

// GPU returns the GPU, or an error saying why there is none that mimir can
// use. mimir uses Metal on Apple silicon Macs, CUDA on NVIDIA GPUs under
// Linux and ROCm on AMD GPUs under Linux, in that order of preference:
// a Linux machine with both NVIDIA and AMD GPUs uses CUDA unless
// MIMIR_DEVICE is rocm.
func GPU() (Device, error) {
	gpuOnce.Do(func() { gpu, gpuErr = openGPU(os.Getenv("MIMIR_DEVICE")) })
	if gpu == nil {
		return Device{}, fmt.Errorf("tensor: no GPU: %w", gpuErr)
	}
	return Device{gpu}, nil
}

// gpuKinds are the GPU backends, in order of preference.
var gpuKinds = []gpuKind{
	{"metal", func() (backend.Backend, error) { return opened(metal.New()) }},
	{"cuda", func() (backend.Backend, error) { return opened(cuda.New()) }},
	{"rocm", func() (backend.Backend, error) { return opened(rocm.New()) }},
}

// gpuKind is a GPU backend: its name and how to open its device.
type gpuKind struct {
	name string
	open func() (backend.Backend, error)
}

// opened returns the backend b that a New returned, or err, as an
// interface that is nil when b is.
func opened[B backend.Backend](b B, err error) (backend.Backend, error) {
	if err != nil {
		return nil, err
	}
	return b, nil
}

// openGPU returns the first GPU backend that has a device, or only the
// backend kind names if kind is metal, cuda or rocm. If there is none, it
// reports why: the errors of the backend asked for, or else of the
// backends this platform could have, Metal on a Mac and CUDA and ROCm
// elsewhere.
func openGPU(kind string) (backend.Backend, error) {
	only := slices.ContainsFunc(gpuKinds, func(g gpuKind) bool { return g.name == kind })
	mac := runtime.GOOS == "darwin"
	var errs []error
	for _, g := range gpuKinds {
		if only && g.name != kind {
			continue
		}
		be, err := g.open()
		if err == nil {
			return be, nil
		}
		if only || (g.name == "metal") == mac || g.name == "metal" && !errors.Is(err, metal.ErrUnavailable) {
			errs = append(errs, err)
		}
	}
	return nil, errors.Join(errs...)
}

// Auto returns the GPU if there is one, and otherwise the CPU.
func Auto() Device {
	if d, err := GPU(); err == nil {
		return d
	}
	return CPU()
}

// Devices returns every device mimir can use: the CPU, then the GPU if
// there is one.
func Devices() []Device {
	ds := []Device{CPU()}
	if d, err := GPU(); err == nil {
		ds = append(ds, d)
	}
	return ds
}

// backend returns the backend of d, resolving the zero Device to the
// default device.
func (d Device) backend() backend.Backend {
	if d.be == nil {
		return defaultDevice().be
	}
	return d.be
}

// Name returns the device's kind: "cpu", "metal", "cuda" or "rocm".
func (d Device) Name() string { return d.backend().Name() }

// String returns the device's name and, for a GPU, its model, such as
// "metal (Apple M5 Pro)".
func (d Device) String() string {
	if m, ok := d.backend().(interface{ DeviceName() string }); ok && m.DeviceName() != "" {
		return fmt.Sprintf("%s (%s)", d.Name(), m.DeviceName())
	}
	return d.Name()
}

// Equal reports whether d and e are the same device.
func (d Device) Equal(e Device) bool { return d.backend() == e.backend() }

var (
	defaultMu  sync.Mutex
	defaultDev Device
	// defaultErr is why the device MIMIR_DEVICE names cannot be used. Every
	// tensor carries it, so that a test run meant for that device fails
	// instead of silently running on the CPU.
	defaultErr error
)

func init() {
	defaultDev = CPU()
	switch name := os.Getenv("MIMIR_DEVICE"); name {
	case "", "cpu":
	case "gpu", "metal", "cuda", "rocm":
		defaultDev, defaultErr = GPU()
		if defaultErr == nil && name != "gpu" && defaultDev.Name() != name {
			defaultDev, defaultErr = CPU(), fmt.Errorf("tensor: MIMIR_DEVICE=%s, but the GPU is %s", name, defaultDev)
		}
	case "auto":
		defaultDev = Auto()
	default:
		defaultErr = fmt.Errorf("tensor: MIMIR_DEVICE=%q, want cpu, gpu, metal, cuda, rocm or auto", name)
	}
}

func defaultDevice() Device {
	defaultMu.Lock()
	defer defaultMu.Unlock()
	return defaultDev
}

// DefaultDevice returns the device that FromSlice uploads to and that
// computes tensors made of constants alone. It is the CPU unless
// SetDefaultDevice or the MIMIR_DEVICE environment variable (cpu, gpu,
// metal, cuda, rocm or auto) changes it.
func DefaultDevice() Device { return defaultDevice() }

// SetDefaultDevice makes d the default device and returns the previous
// one.
func SetDefaultDevice(d Device) Device {
	defaultMu.Lock()
	defer defaultMu.Unlock()
	prev := defaultDev
	if d.be != nil {
		defaultDev = d
	}
	return prev
}

// SetMemoryBudget limits the memory that computations on d may hold to
// bytes, or removes the limit when bytes is 0. A computation that would
// take the device past its budget fails with ErrMemoryBudget before it
// allocates or computes anything, instead of running out of memory
// midway. The budget counts the memory of computed tensors on d, held by
// them or kept by the device for reuse, until they are garbage collected;
// tensors uploaded with FromSlice are not counted, unless they reuse
// memory the device kept, as the next batch after a donated one does (see
// Donate). train.MemoryBudget
// trains within a budget by splitting batches that do not fit.
func SetMemoryBudget(d Device, bytes int64) { sched.SetBudget(d.backend(), bytes) }

// MemoryBudget returns the memory budget of d, or 0 if it has none.
func MemoryBudget(d Device) int64 { return sched.Budget(d.backend()) }

// backendOf returns the backend that computes n: the device of its inputs
// or memory, or the default device.
func backendOf(n *graph.Node) backend.Backend {
	if be := sched.Device(n); be != nil {
		return be
	}
	return defaultDevice().be
}

// Device returns the device t lives on, or would be computed on now.
func (t *Tensor) Device() Device {
	if firstErr(t) != nil {
		return Device{}
	}
	return Device{backendOf(t.node)}
}

// To returns t's values on d. If t is on d already, it returns t; otherwise
// it computes t and copies its values, and gradients flow back through the
// copy to t. A sharded tensor's shards are gathered, and To a mesh's
// device shards t whole on every device of the mesh (see Shard).
func (t *Tensor) To(d Device) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	from, to := backendOf(t.node), d.backend()
	if from == to {
		return t
	}
	if md, ok := to.(*meshDevice); ok {
		return Shard(t, md.mesh, nil)
	}
	src := Device{from}
	toRule := rule{
		vjp: func(g *Tensor, _ []*Tensor, _ *Tensor) []*Tensor { return grads(g.To(src)) },
		jvp: func(t, _ []*Tensor, _ *Tensor) *Tensor { return t[0].To(d) },
	}
	if s, ok := t.Sharding(); ok {
		toRule.vjp = func(g *Tensor, _ []*Tensor, _ *Tensor) []*Tensor { return grads(Shard(g, s.Mesh, s.Spec)) }
	}
	if l := batchLevel(t); l != 0 {
		return record(batchTo(l, t, d), toRule, t)
	}
	if pc, ok := to.(backend.PeerCopier); ok && pc.CopiesFrom(from) && meshOf(t.node) == nil && !t.node.DType.IsQuantized() {
		// The devices copy between themselves, and the host waits for
		// neither.
		return record(wrap(sched.Copy(context.Background(), from, pc, t.node, lazy.values)), toRule, t)
	}
	data, err := hostBytes(context.Background(), t.node)
	if err != nil {
		return fail(err)
	}
	return record(upload(to, t.node.DType, data, t.node.Shape), toRule, t)
}

// upload returns a tensor of shape holding data, as elements of dt, on be.
func upload(be backend.Backend, dt DType, data []byte, shape []int) *Tensor {
	if defaultErr != nil {
		return fail(defaultErr)
	}
	return wrap(sched.Upload(context.Background(), be, dt, data, shape))
}
