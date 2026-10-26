package llm

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/io/mim"
	"github.com/gsoultan/mimir/nn"
	"github.com/gsoultan/mimir/tensor"
)

// LoRAConfig says where AddLoRA puts pairs and how large they are.
type LoRAConfig struct {
	// Rank is the rank of every pair; 0 means 8.
	Rank int
	// Alpha scales every pair by Alpha/Rank; 0 means twice the rank.
	Alpha float64
	// Targets are the projections of every block that get a pair, of
	// Projections; nil means DefaultTargets.
	Targets []string
	// Seed draws the pairs' starting A.
	Seed uint64
}

// DefaultTargets are the projections AddLoRA adapts when a LoRAConfig
// names none: those of attention.
var DefaultTargets = []string{"attn_q", "attn_k", "attn_v", "attn_output"}

// AddLoRA puts a trainable pair of c's rank next to the projections c
// targets in every block, which then learn alone: the model's own weights
// are buffers. A fresh pair changes nothing until it trains. The model
// takes pairs once, from AddLoRA or ReadAdapter.
func (m *Model) AddLoRA(c LoRAConfig) error {
	if c.Rank == 0 {
		c.Rank = 8
	}
	if c.Alpha == 0 {
		c.Alpha = 2 * float64(c.Rank)
	}
	if c.Targets == nil {
		c.Targets = DefaultTargets
	}
	if c.Rank < 0 || !(c.Alpha > 0) || math.IsInf(c.Alpha, 0) || len(c.Targets) == 0 {
		return fmt.Errorf("llm: LoRA of rank %d, alpha %v, for %v: %w", c.Rank, c.Alpha, c.Targets, tensor.ErrShape)
	}
	for _, t := range c.Targets {
		if !slices.Contains(Projections, t) {
			return fmt.Errorf("llm: no projection %q to adapt, want one of %v: %w", t, Projections, tensor.ErrShape)
		}
	}
	if len(m.Adapters()) > 0 {
		return errors.New("llm: the model already has LoRA pairs")
	}
	n := uint64(0)
	for _, b := range m.Blocks {
		ps := b.projections()
		for i, name := range Projections {
			if !slices.Contains(c.Targets, name) {
				continue
			}
			l := nn.NewLoRA(linearOf(*ps[i]), c.Rank, c.Alpha, nn.Seed(c.Seed+n))
			if err := l.A.Value().Err(); err != nil {
				return err
			}
			*ps[i] = l
			n++
		}
	}
	m.alpha = c.Alpha
	return nil
}

// pairs yields the pairs of the model, each with the name of the weight
// it adapts.
func (m *Model) pairs(yield func(string, *nn.LoRA) bool) {
	for i, b := range m.Blocks {
		for j, p := range b.projections() {
			if l, ok := (*p).(*nn.LoRA); ok && !yield("blk."+strconv.Itoa(i)+"."+Projections[j]+".weight", l) {
				return
			}
		}
	}
}

// Adapters returns the trainable parameters of the pairs, A then B of
// each, block by block: what an optimizer of a fine-tune steps.
func (m *Model) Adapters() []*tensor.Param {
	var ps []*tensor.Param
	for _, l := range m.pairs {
		ps = append(ps, l.A, l.B)
	}
	return ps
}

// WriteAdapter writes the model's pairs to w as an adapter file of the
// format (spec/adapters.md), in Float32. base is the lowercase hex SHA-256
// of the base model's file, which the adapter names so that it is applied
// to no other, or "" to leave it unnamed. name, if set, becomes the
// file's general.name.
func (m *Model) WriteAdapter(ctx context.Context, w io.Writer, base, name string) error {
	fw := mimfile.NewWriter()
	n := 0
	for target, l := range m.pairs {
		for _, t := range []struct {
			suffix string
			p      *tensor.Param
		}{{mimfile.LoRASuffixA, l.A}, {mimfile.LoRASuffixB, l.B}} {
			v := t.p.Value()
			data, err := v.Cast(tensor.Float32).Data[float32](ctx)
			if err != nil {
				return fmt.Errorf("llm: write adapter: %s: %w", target, err)
			}
			b := make([]byte, 4*len(data))
			for i, x := range data {
				binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
			}
			shape := v.Shape()
			if err := fw.AddTensor(target+t.suffix, mimfile.F32, []uint64{uint64(shape[0]), uint64(shape[1])}, b); err != nil {
				return fmt.Errorf("llm: write adapter: %w", err)
			}
		}
		n++
	}
	if n == 0 {
		return errors.New("llm: write adapter: the model has no LoRA pairs")
	}
	md := mimfile.LoRAMetadata(m.alpha, base)
	md["general.architecture"] = m.Config.Arch
	if name != "" {
		md["general.name"] = name
	}
	mb, err := md.Encode()
	if err != nil {
		return fmt.Errorf("llm: write adapter: %w", err)
	}
	if err := fw.AddSection(mimfile.SectionMetadata, false, mb); err != nil {
		return fmt.Errorf("llm: write adapter: %w", err)
	}
	_, err = fw.WriteTo(w)
	return err
}

// ReadAdapter puts the pairs of the adapter file a on the model, as
// trainable parameters, to fine-tune it further or to run it with them.
// It fails with tensor.ErrShape for an adapter of weights the model
// doesn't have or of other shapes, and the model is left as it was.
func (m *Model) ReadAdapter(a *mim.Model) error {
	if len(m.Adapters()) > 0 {
		return errors.New("llm: the model already has LoRA pairs")
	}
	l, err := a.LoRA()
	if err != nil {
		return err
	}
	dev := m.Embed.Weight.Value().Device()
	type change struct {
		at *nn.Module
		l  *nn.LoRA
	}
	var changes []change
	for _, p := range l.Layers {
		rest, ok := strings.CutPrefix(p.Target, "blk.")
		i, proj, ok2 := strings.Cut(rest, ".")
		bi, err := strconv.Atoi(i)
		j := slices.Index(Projections, strings.TrimSuffix(proj, ".weight"))
		if !ok || !ok2 || err != nil || bi < 0 || bi >= len(m.Blocks) || j < 0 || !strings.HasSuffix(proj, ".weight") {
			return fmt.Errorf("llm: adapter of %s, which is not a projection of the model: %w", p.Target, tensor.ErrShape)
		}
		at := m.Blocks[bi].projections()[j]
		base := linearOf(*at)
		if s := base.Weight.Value().Shape(); !slices.Equal(s, []int{p.Out, p.In}) {
			return fmt.Errorf("llm: adapter of %s as [%d %d]; the model's is %v: %w", p.Target, p.Out, p.In, s, tensor.ErrShape)
		}
		pair := func(name string) (*tensor.Param, error) {
			t, err := a.Tensor(name)
			if err != nil {
				return nil, err
			}
			p := tensor.NewParam(t.Cast(tensor.Float32))
			if err := p.Value().Err(); err != nil {
				return nil, err
			}
			return p, p.To(dev)
		}
		la, err := pair(p.A.Name)
		if err != nil {
			return err
		}
		lb, err := pair(p.B.Name)
		if err != nil {
			return err
		}
		changes = append(changes, change{at, &nn.LoRA{Base: base, A: la, B: lb, Scale: l.Scale(p)}})
	}
	for _, c := range changes {
		*c.at = c.l
	}
	m.alpha = l.Alpha
	return nil
}
