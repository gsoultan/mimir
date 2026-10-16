package cpu

import (
	"context"
	"fmt"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/backend"
)

// Stream is a CPU device that runs kernels in order on a goroutine of its
// own, as a GPU runs them on a stream: Run and RunBatch queue them and
// return. Several Streams share the host's processors as separate devices,
// each with workers goroutines per kernel, so that the kernels of one run
// while the host queues the work of the others, as the devices of a
// data-parallel group do (see Group).
//
// A Stream is Asynchronous: the errors of its kernels surface at the next
// Sync, Upload or Download. Its buffers are those of Backend, and a
// Backend and Streams form groups together. It is safe for concurrent
// use.
type Stream struct {
	be *Backend

	mu      sync.Mutex
	idle    sync.Cond // signalled when the queue empties
	jobs    []func() error
	head    int   // the next job to run
	running bool  // a goroutine is running the queue
	err     error // the first error since the last wait
}

var (
	_ backend.Batcher      = (*Stream)(nil)
	_ backend.Asynchronous = (*Stream)(nil)
	_ backend.Grouper      = (*Stream)(nil)
)

// NewStream returns a Stream whose kernels use up to workers goroutines,
// at least one.
func NewStream(workers int) *Stream {
	workers = max(workers, 1)
	s := &Stream{be: &Backend{workers: workers, team: newTeam(workers - 1)}}
	s.idle.L = &s.mu
	return s
}

// Name returns "cpu".
func (*Stream) Name() string { return "cpu" }

// Workers returns the most goroutines a kernel of s uses.
func (s *Stream) Workers() int { return s.be.workers }

// Asynchronous reports true: kernels run after Run returns.
func (*Stream) Asynchronous() bool { return true }

// Alloc returns a zeroed buffer of n elements, as Backend.Alloc does.
func (s *Stream) Alloc(dt backend.DType, n int) (backend.Buffer, error) { return s.be.Alloc(dt, n) }

// Free releases b once the kernels queued so far have run.
func (s *Stream) Free(b backend.Buffer) {
	s.queue(func() error {
		s.be.Free(b)
		return nil
	})
}

// Upload copies src into dst once the kernels queued so far have run, and
// reports their first error since the last Sync.
func (s *Stream) Upload(ctx context.Context, dst backend.Buffer, src []byte) error {
	err := s.wait()
	if uerr := s.be.Upload(ctx, dst, src); err == nil {
		err = uerr
	}
	return err
}

// Download copies src into dst once the kernels queued so far have run,
// and reports their first error since the last Sync.
func (s *Stream) Download(ctx context.Context, dst []byte, src backend.Buffer) error {
	err := s.wait()
	if derr := s.be.Download(ctx, dst, src); err == nil {
		err = derr
	}
	return err
}

// Sync waits for the kernels queued so far and returns their first error
// since the last Sync.
func (s *Stream) Sync(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return s.wait()
}

// Run checks k and queues it. Cancelling ctx stops a kernel from being
// queued but not one that is queued already, which runs to its end as a
// GPU's does.
func (s *Stream) Run(ctx context.Context, k *backend.Kernel, outs []backend.Buffer, ins []backend.Buffer) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := backend.CheckRun(k, outs, ins); err != nil {
		return fmt.Errorf("cpu: %w", err)
	}
	// The caller may change k's arguments and reuse the slices once Run
	// returns, as a replayed schedule does, so the job keeps copies.
	kc, oc, ic := snapshot(k), slices.Clone(outs), slices.Clone(ins)
	ctx = context.WithoutCancel(ctx)
	s.queue(func() error { return s.be.Run(ctx, kc, oc, ic) })
	return nil
}

// RunBatch checks the kernels of b and queues them as one job.
func (s *Stream) RunBatch(ctx context.Context, b *backend.Batch) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	ks := make([]backend.Kernel, len(b.Kernels))
	outs := make([][]backend.Buffer, len(b.Kernels))
	ins := make([][]backend.Buffer, len(b.Kernels))
	for i, k := range b.Kernels {
		if err := backend.CheckRun(k, b.Outs[i], b.Ins[i]); err != nil {
			return fmt.Errorf("cpu: %w", err)
		}
		ks[i] = *snapshot(k)
		outs[i], ins[i] = slices.Clone(b.Outs[i]), slices.Clone(b.Ins[i])
	}
	ctx = context.WithoutCancel(ctx)
	s.queue(func() error {
		for i := range ks {
			if err := s.be.Run(ctx, &ks[i], outs[i], ins[i]); err != nil {
				return err
			}
		}
		return nil
	})
	return nil
}

// Group returns a Communicator over devices, as Backend.Group does.
func (s *Stream) Group(devices []backend.Backend) (backend.Communicator, error) {
	return newCommunicator(devices)
}

// snapshot returns a copy of k that keeps its arguments, which a
// scheduler replaying k changes in place: the Value of constants in its
// program, besides Value and Seed.
func snapshot(k *backend.Kernel) *backend.Kernel {
	c := *k
	if c.Attrs.Program != nil {
		c.Attrs.Program = slices.Clone(c.Attrs.Program)
	}
	return &c
}

// queue appends job to the queue, starting a goroutine to run the queue
// if none is running. The goroutine exits when the queue is empty.
func (s *Stream) queue(job func() error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.jobs = append(s.jobs, job)
	if !s.running {
		s.running = true
		go s.drain()
	}
}

// drain runs the queued jobs in order until none is left, keeping the
// first error.
func (s *Stream) drain() {
	s.mu.Lock()
	for s.head < len(s.jobs) {
		job := s.jobs[s.head]
		s.jobs[s.head] = nil
		s.head++
		s.mu.Unlock()
		err := runJob(job)
		s.mu.Lock()
		if err != nil && s.err == nil {
			s.err = err
		}
	}
	s.jobs, s.head = s.jobs[:0], 0
	s.running = false
	s.idle.Broadcast()
	s.mu.Unlock()
}

// runJob calls job, reporting a panic as backend.ErrInternal.
func runJob(job func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("cpu: panic in stream: %v: %w", r, backend.ErrInternal)
		}
	}()
	return job()
}

// wait blocks until the queue is empty and returns, and forgets, the
// first error of the jobs that ran since the last wait.
func (s *Stream) wait() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for s.running {
		s.idle.Wait()
	}
	err := s.err
	s.err = nil
	return err
}
