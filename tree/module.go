// Copyright (c) 2026 Gembit Soultan Shirazi
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of mimir, a pure-Go machine learning framework,
// created and owned by Gembit Soultan Shirazi.
// It is licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License in the LICENSE file at the root
// of this repository or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

package tree

import (
	"fmt"
	"math"
	"slices"

	"github.com/gsoultan/mimir/tensor"
)

// Shape is what sizes a Module: enough to build an empty one that saved
// weights load into, as nn.Module files are loaded.
type Shape struct {
	Trees, Nodes int // trees, and nodes in all of them
	Depth        int // the most splits from a root to a leaf
	Features     int
	Outputs      int // the numbers each leaf holds
	Classes      int // 0 for Regression
	Link         Link
	// Categories are the model's category columns, and CatSlots the codes
	// the sets of all its category splits cover.
	Categories []Category `json:",omitempty"`
	CatSlots   int        `json:",omitempty"`
}

// Module is a Model as an nn.Module, so it trains nowhere but runs, saves
// and loads wherever a network does: on every device, and in Mimir Model
// files. Forward maps x, shape [batch, Features], to [batch, Classes]
// logits whose softmax is the model's probabilities, or to [batch, 1]
// values for Regression.
//
// Its tables are buffers, not trained parameters. Each tree is walked one
// level a step, every row and tree at once, by gathering the node's
// feature, threshold and children: a leaf leads to itself, so Depth steps
// reach every leaf.
type Module struct {
	Roots     *tensor.Param // Int32 [Trees], each tree's root node
	Feature   *tensor.Param // Int32 [Nodes], 0 for a leaf
	Threshold *tensor.Param // Float32 [Nodes]
	Left      *tensor.Param // Int32 [Nodes], the node a row at most the threshold goes to
	Right     *tensor.Param // Int32 [Nodes], the node a row above it goes to
	Missing   *tensor.Param // Int32 [Nodes], the node a row missing the feature goes to
	Value     *tensor.Param // Float32 [Nodes*Outputs], scaled by the model's Scale
	Base      *tensor.Param // Float32 [1, Outputs]
	// Cat, for a model with category columns, holds its category splits.
	Cat   *CatTables
	shape Shape
}

// CatTables hold a Module's category splits. A node splitting on a
// category reads its column's code, past the row's own features, and goes
// left when the code's entry of its set is 1.
type CatTables struct {
	Offset *tensor.Param // Int32 [Nodes], where a node's set starts in Left
	Size   *tensor.Param // Int32 [Nodes], the codes its set covers, 0 for a split on an amount
	Left   *tensor.Param // Int32 [CatSlots], 1 for each code sent left
}

// NewModule returns a Module of shape s whose tables are zero, for
// weights to be loaded into.
func NewModule(s Shape) *Module {
	nodes := max(s.Nodes, 1)
	return &Module{
		Roots:     tensor.NewBuffer(tensor.Zeros(tensor.Int32, max(s.Trees, 1))),
		Feature:   tensor.NewBuffer(tensor.Zeros(tensor.Int32, nodes)),
		Threshold: tensor.NewBuffer(tensor.Zeros(tensor.Float32, nodes)),
		Left:      tensor.NewBuffer(tensor.Zeros(tensor.Int32, nodes)),
		Right:     tensor.NewBuffer(tensor.Zeros(tensor.Int32, nodes)),
		Missing:   tensor.NewBuffer(tensor.Zeros(tensor.Int32, nodes)),
		Value:     tensor.NewBuffer(tensor.Zeros(tensor.Float32, nodes*max(s.Outputs, 1))),
		Base:      tensor.NewBuffer(tensor.Zeros(tensor.Float32, 1, max(s.Outputs, 1))),
		Cat:       newCatTables(s, nodes),
		shape:     s,
	}
}

func newCatTables(s Shape, nodes int) *CatTables {
	if len(s.Categories) == 0 {
		return nil
	}
	return &CatTables{
		Offset: tensor.NewBuffer(tensor.Zeros(tensor.Int32, nodes)),
		Size:   tensor.NewBuffer(tensor.Zeros(tensor.Int32, nodes)),
		Left:   tensor.NewBuffer(tensor.Zeros(tensor.Int32, max(s.CatSlots, 1))),
	}
}

// Shape returns the shape of m's Module.
func (m *Model) Shape() Shape {
	s := Shape{Trees: len(m.Trees), Features: m.Features, Outputs: m.Outputs, Classes: m.Classes, Link: m.Link,
		Categories: slices.Clone(m.Categories)}
	for i := range m.Trees {
		s.Nodes += m.Trees[i].Nodes()
		s.Depth = max(s.Depth, m.Trees[i].depth(0))
		for _, set := range m.Trees[i].Cats {
			s.CatSlots += 64 * len(set)
		}
	}
	return s
}

// depth returns the most splits below node n.
func (t *Tree) depth(n int32) int {
	if t.Feature[n] < 0 {
		return 0
	}
	return 1 + max(t.depth(t.Left[n]), t.depth(t.Right[n]))
}

// Module returns m as a Module.
func (m *Model) Module() *Module {
	s := m.Shape()
	roots := make([]int32, 0, s.Trees)
	feature, left, right, missing := make([]int32, 0, s.Nodes), make([]int32, 0, s.Nodes), make([]int32, 0, s.Nodes), make([]int32, 0, s.Nodes)
	threshold, value := make([]float32, 0, s.Nodes), make([]float32, 0, s.Nodes*s.Outputs)
	var catOff, catSize, catLeft []int32
	group := map[int32]int32{} // a category column's first feature, to its code's column
	for g, ct := range m.Categories {
		group[int32(ct.Feature)] = int32(m.Features + g)
	}
	for i := range m.Trees {
		t := &m.Trees[i]
		off := int32(len(feature))
		roots = append(roots, off)
		for n, f := range t.Feature {
			self := off + int32(n)
			if len(m.Categories) > 0 {
				var set []uint64
				if t.Cats != nil && f >= 0 {
					set = t.Cats[n]
				}
				off := int32(0) // a node with no set reads entry 0, unused
				if set != nil {
					off = int32(len(catLeft))
				}
				catOff, catSize = append(catOff, off), append(catSize, int32(64*len(set)))
				for c := range 64 * len(set) {
					catLeft = append(catLeft, int32(set[c>>6]>>(c&63)&1))
				}
				if set != nil {
					f = group[f]
				}
			}
			if f < 0 {
				feature, threshold = append(feature, 0), append(threshold, 0)
				left, right, missing = append(left, self), append(right, self), append(missing, self)
			} else {
				feature, threshold = append(feature, f), append(threshold, t.Threshold[n])
				left, right = append(left, off+t.Left[n]), append(right, off+t.Right[n])
				if t.MissingLeft[n] {
					missing = append(missing, off+t.Left[n])
				} else {
					missing = append(missing, off+t.Right[n])
				}
			}
		}
		for _, v := range t.Value {
			value = append(value, m.Scale*v)
		}
	}
	if len(roots) == 0 {
		return NewModule(s)
	}
	var cat *CatTables
	if len(m.Categories) > 0 {
		if len(catLeft) == 0 {
			catLeft = []int32{0}
		}
		cat = &CatTables{
			Offset: tensor.NewBuffer(tensor.FromSlice(catOff)),
			Size:   tensor.NewBuffer(tensor.FromSlice(catSize)),
			Left:   tensor.NewBuffer(tensor.FromSlice(catLeft)),
		}
	}
	return &Module{
		Cat:       cat,
		Roots:     tensor.NewBuffer(tensor.FromSlice(roots)),
		Feature:   tensor.NewBuffer(tensor.FromSlice(feature)),
		Threshold: tensor.NewBuffer(tensor.FromSlice(threshold)),
		Left:      tensor.NewBuffer(tensor.FromSlice(left)),
		Right:     tensor.NewBuffer(tensor.FromSlice(right)),
		Missing:   tensor.NewBuffer(tensor.FromSlice(missing)),
		Value:     tensor.NewBuffer(tensor.FromSlice(value)),
		Base:      tensor.NewBuffer(tensor.FromSlice(slices.Clone(m.Base), 1, s.Outputs)),
		shape:     s,
	}
}

// Shape returns the shape mod was built with.
func (mod *Module) Shape() Shape { return mod.shape }

// Forward answers each row of x, shape [batch, Features].
func (mod *Module) Forward(x *tensor.Tensor) *tensor.Tensor {
	s := mod.shape
	if err := x.Err(); err != nil {
		return x
	}
	if x.Rank() != 2 || x.Shape()[1] != s.Features {
		return tensor.Failed(fmt.Errorf("tree: input %v for rows of %d features: %w", x.Shape(), s.Features, tensor.ErrShape))
	}
	n, k := x.Shape()[0], s.Outputs
	trees, nodes := mod.Roots.Value().Shape()[0], mod.Feature.Value().Shape()[0]
	// lookup reads table at each row's node in each tree.
	lookup := func(table *tensor.Param, node *tensor.Tensor) *tensor.Tensor {
		return table.Value().Unsqueeze(0).Expand(n, nodes).Gather(1, node)
	}
	node := mod.Roots.Value().Unsqueeze(0).Expand(n, trees)
	if mod.Cat != nil {
		x = withCodes(x, s.Categories)
	}
	for range s.Depth {
		v := x.Gather(1, lookup(mod.Feature, node))
		l, r := lookup(mod.Left, node), lookup(mod.Right, node)
		next := tensor.Where(v.Greater(lookup(mod.Threshold, node)), r, l)
		if mod.Cat != nil {
			size := lookup(mod.Cat.Size, node)
			sizeF := size.Cast(tensor.Float32)
			// The code's entry in the node's set, read at a code held
			// within the set, so a missing or unknown one reads safely.
			at := tensor.Where(v.Equal(v), v, tensor.Scalar[float32](0)).
				Maximum(tensor.Scalar[float32](0)).Minimum(sizeF.Sub(tensor.Scalar[float32](1)).Maximum(tensor.Scalar[float32](0)))
			idx := at.Cast(tensor.Int32).Add(lookup(mod.Cat.Offset, node))
			slots := mod.Cat.Left.Value().Shape()[0]
			bit := mod.Cat.Left.Value().Unsqueeze(0).Expand(n, slots).Gather(1, idx)
			in := tensor.Where(v.Less(tensor.Scalar[float32](0)), r, tensor.Where(v.Less(sizeF), l, r))
			catNext := tensor.Where(bit.Greater(tensor.Scalar[int32](0)), in, r)
			next = tensor.Where(size.Greater(tensor.Scalar[int32](0)), catNext, next)
		}
		node = tensor.Where(v.Equal(v), next, lookup(mod.Missing, node))
	}
	idx := node.Unsqueeze(2).Mul(tensor.Scalar(int32(k))).Add(tensor.Arange[int32](0, int32(k), 1)).Reshape(n, trees*k)
	out := mod.Value.Value().Unsqueeze(0).Expand(n, nodes*k).Gather(1, idx).Reshape(n, trees, k).Sum(1).Add(mod.Base.Value())
	switch {
	case s.Link == Sigmoid:
		// Logits 0 and the log-odds have the sigmoid's probabilities.
		return tensor.Cat(1, tensor.Zeros(tensor.Float32, n, 1), out)
	case s.Link == Isolation:
		return out.Neg().Exp2()
	case s.Link == Exp:
		return out.Exp()
	case s.Link == Identity && s.Classes > 0:
		// A forest averages probabilities: their logs are logits.
		return out.Maximum(tensor.Scalar(float32(math.SmallestNonzeroFloat32))).Log()
	}
	return out
}

// withCodes returns x, shape [batch, Features], with a column more for
// each category column: its code, or NaN when it is missing, as codes
// reads one.
func withCodes(x *tensor.Tensor, cats []Category) *tensor.Tensor {
	n := x.Shape()[0]
	cols := []*tensor.Tensor{x}
	nan := tensor.Scalar(float32(math.NaN()))
	for _, ct := range cats {
		if ct.Width == 0 {
			cols = append(cols, x.Shrink([][2]int{{0, n}, {ct.Feature, ct.Feature + 1}}))
			continue
		}
		set := x.Shrink([][2]int{{0, n}, {ct.Feature, ct.Feature + ct.Width}}).Greater(tensor.Scalar[float32](0.5)).Cast(tensor.Float32)
		code := set.Argmax(1).Cast(tensor.Float32).Unsqueeze(1)
		cols = append(cols, tensor.Where(set.Max(1).Unsqueeze(1).Greater(tensor.Scalar[float32](0.5)), code, nan))
	}
	return tensor.Cat(1, cols...)
}
