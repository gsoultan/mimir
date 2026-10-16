package tensor

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"

	"github.com/gsoultan/mimir/backend"
	"github.com/gsoultan/mimir/backend/cpu"
	"github.com/gsoultan/mimir/backend/cuda"
	"github.com/gsoultan/mimir/backend/metal"
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
	gpu     backend.Backend // the Metal backend, or nil
	gpuErr  error
)

// CPU returns the host CPU.
func CPU() Device { return Device{cpuBackend} }

// GPU returns the GPU, or an error saying why there is none that mimir can
// use. mimir uses Metal on Apple silicon Macs and CUDA on NVIDIA GPUs
// under Linux.
func GPU() (Device, error) {
	gpuOnce.Do(func() {
		m, metalErr := metal.New()
		if metalErr == nil {
			gpu = m
			return
		}
		c, cudaErr := cuda.New()
		if cudaErr == nil {
			gpu = c
			return
		}
		// Report the error of the GPU this platform could have.
		gpuErr = cudaErr
		if runtime.GOOS == "darwin" || !errors.Is(metalErr, metal.ErrUnavailable) {
			gpuErr = metalErr
		}
	})
	if gpu == nil {
		return Device{}, fmt.Errorf("tensor: no GPU: %w", gpuErr)
	}
	return Device{gpu}, nil
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

// Name returns the device's kind: "cpu", "metal" or "cuda".
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
	case "gpu", "metal", "cuda":
		defaultDev, defaultErr = GPU()
		if defaultErr == nil && name != "gpu" && defaultDev.Name() != name {
			defaultDev, defaultErr = CPU(), fmt.Errorf("tensor: MIMIR_DEVICE=%s, but the GPU is %s", name, defaultDev)
		}
	case "auto":
		defaultDev = Auto()
	default:
		defaultErr = fmt.Errorf("tensor: MIMIR_DEVICE=%q, want cpu, gpu, metal, cuda or auto", name)
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
// metal, cuda or auto) changes it.
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
// copy to t.
func (t *Tensor) To(d Device) *Tensor {
	if err := firstErr(t); err != nil {
		return fail(err)
	}
	from, to := backendOf(t.node), d.backend()
	if from == to {
		return t
	}
	ctx := context.Background()
	data := make([]byte, t.node.Size()*t.node.DType.Size())
	if err := sched.Download(ctx, from, t.node, data); err != nil {
		return fail(err)
	}
	out := upload(to, t.node.DType, data, t.node.Shape)
	src := Device{from}
	return track(out, func(g *Tensor) []*Tensor { return []*Tensor{g.To(src)} }, t)
}

// upload returns a tensor of shape holding data, as elements of dt, on be.
func upload(be backend.Backend, dt DType, data []byte, shape []int) *Tensor {
	if defaultErr != nil {
		return fail(defaultErr)
	}
	n, _ := backend.NumElements(shape)
	buf, err := be.Alloc(dt, n)
	if err != nil {
		return fail(err)
	}
	if err := be.Upload(context.Background(), buf, data); err != nil {
		be.Free(buf)
		return fail(err)
	}
	return wrap(graph.Load(be, buf, shape))
}
