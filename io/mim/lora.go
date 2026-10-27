package mim

import (
	"context"
	"errors"
	"fmt"
	"strings"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/tensor"
)

// loraPair is the adapter of one weight: the keys of its A and B in the
// model's consts, and the scale of their product.
type loraPair struct {
	a, b  uint32
	scale float64
}

// ApplyLoRA applies the LoRA adapter in a, an adapter file read by Open or
// Parse (see spec/adapters.md of the format), to m: from then on the model
// computes as if each weight W the adapter names were W + α/r·B·A, while
// W itself stays as the file holds it, quantized or not. sha256 is the
// lowercase hex SHA-256 of m's file, checked against the base the adapter
// names, or "" to skip that check.
//
// Only weights that the graph reads as the weight of linear.v1 or a
// dequant_matmul composite can be adapted; an adapter of another fails with
// an *UnsupportedError. A malformed adapter, or one of another base, fails
// with ErrFormat. A model takes one adapter, before its first run: an
// adapter applied while the model runs is not safe. a is not needed once
// ApplyLoRA returns.
func (m *Model) ApplyLoRA(a *Model, sha256 string) error {
	if a == nil || a == m {
		return errors.New("mim: no adapter")
	}
	l, err := a.LoRA()
	if err != nil {
		return err
	}

	m.mu.Lock()
	closed, g := m.closed, m.g
	m.mu.Unlock()
	switch {
	case closed:
		return errors.New("mim: adapter for a closed model")
	case g == nil:
		return fmt.Errorf("%w: an adapter for a file with no graph", ErrFormat)
	case m.lora != nil:
		return errors.New("mim: the model already has an adapter")
	}
	if err := l.CheckBase(m.f, sha256); err != nil {
		return fmt.Errorf("%w: adapter: %w", ErrFormat, err)
	}

	// The weight values each target is read through.
	byName := map[string][]uint32{}
	for i, v := range g.Values {
		if v.Kind == mimfile.KindWeight && int(v.Ref) < m.f.NumTensors() {
			name := m.f.TensorAt(int(v.Ref)).Name
			byName[name] = append(byName[name], uint32(i))
		}
	}
	targets := map[uint32]bool{}
	for _, p := range l.Layers {
		vs := byName[p.Target]
		if len(vs) == 0 {
			return &UnsupportedError{Op: "adapter", Detail: fmt.Sprintf("the graph doesn't read %s", p.Target)}
		}
		for _, vi := range vs {
			targets[vi] = true
		}
	}
	for si := range g.Subgraphs {
		for oi := range g.Subgraphs[si].Ops {
			op := &g.Subgraphs[si].Ops[oi]
			for j, vi := range op.Inputs {
				if targets[vi] && (j != 1 || !adaptable(opName(op))) {
					return &UnsupportedError{Op: "adapter", Detail: fmt.Sprintf("%s reads %q as input %d", opName(op), valueName(g.Values[vi], vi), j)}
				}
			}
		}
	}

	// The pairs, in Float32, as consts of their own past the graph's
	// values, so a run on another device copies them with the weights.
	pairs := map[uint32]loraPair{}
	extra := map[uint32]*tensor.Tensor{}
	next := uint32(len(g.Values))
	for _, p := range l.Layers {
		at, err := a.Tensor(p.A.Name)
		if err != nil {
			return fmt.Errorf("%w: adapter: %w", ErrFormat, err)
		}
		bt, err := a.Tensor(p.B.Name)
		if err != nil {
			return fmt.Errorf("%w: adapter: %w", ErrFormat, err)
		}
		at, bt = at.Cast(tensor.Float32), bt.Cast(tensor.Float32)
		if err := tensor.Realize(context.Background(), at, bt); err != nil {
			return fmt.Errorf("mim: adapter %s: %w", p.Target, err)
		}
		pr := loraPair{a: next, b: next + 1, scale: l.Scale(p)}
		extra[pr.a], extra[pr.b] = at, bt
		next += 2
		for _, vi := range byName[p.Target] {
			pairs[vi] = pr
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return errors.New("mim: adapter for a closed model")
	}
	for k, t := range extra {
		m.consts[k] = t
	}
	m.lora = pairs
	// Weights already copied to a device, and recorded runs, are of the
	// model without the adapter.
	clear(m.onDevice)
	clear(m.recorded)
	return nil
}

// LoRA reads the model's file as a LoRA adapter, checking it against the
// rules of spec/adapters.md. It fails with ErrFormat for a file that isn't
// one.
func (m *Model) LoRA() (*mimfile.LoRA, error) {
	m.mu.Lock()
	f := m.f
	m.mu.Unlock()
	if f == nil {
		return nil, errors.New("mim: adapter of a closed model")
	}
	l, err := f.LoRA()
	if err != nil {
		return nil, fmt.Errorf("%w: adapter: %w", ErrFormat, err)
	}
	return l, nil
}

// adaptable reports whether the call op names reads its input 1 as the
// weight of a product x·Wᵀ, which an adapter can add its pair's product
// to.
func adaptable(op string) bool {
	name, ok := strings.CutPrefix(op, "call ")
	if !ok {
		return false
	}
	return name == "linear.v1" || strings.HasPrefix(name, "dequant_matmul.") || strings.HasPrefix(name, "dequant_matmul_a8.")
}

// adapt adds to y, the product of x with the call's weight, the product
// with the weight's adapter, if it has one.
func (c *call) adapt(y, x *tensor.Tensor) *tensor.Tensor {
	p, ok := c.r.m.lora[c.op.Inputs[1]]
	if !ok || y.Err() != nil {
		return y
	}
	a, b := c.r.consts[p.a], c.r.consts[p.b]
	d := x.MatMulT(a).MatMulT(b)
	if p.scale != 1 {
		d = d.Mul(tensor.Full(tensor.Float32, p.scale).To(d.Device()))
	}
	return y.Add(d)
}
