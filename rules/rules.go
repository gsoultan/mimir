// Package rules finds what goes together in baskets of items, such as
// the products of each order, as association rules: "customers who buy
// A and B also buy C", with how often the whole set is bought (support),
// how often C follows A and B (confidence) and how much more often than
// C is bought anyway (lift). It runs the Apriori algorithm, which counts
// sets level by level and only extends the frequent ones.
//
//	found, err := rules.Mine(baskets, rules.MinSupport(0.01), rules.MinConfidence(0.3))
//	if err != nil { ... }
//	for _, r := range found { fmt.Println(r.If, "->", r.Then, r.Lift) }
//
// Items are strings; an item repeated in a basket counts once.
package rules

import (
	"context"
	"fmt"
	"slices"
	"sort"

	"github.com/gsoultan/mimir/tensor"
)

// Rule says baskets with every item of If tend to hold Then too.
type Rule struct {
	If   []string
	Then string
	// Support is the share of baskets holding If and Then, Confidence
	// the share of those holding If that hold Then too, and Lift
	// Confidence over the share of all baskets holding Then: above 1,
	// If makes Then likelier.
	Support, Confidence, Lift float64
	// Count is the number of baskets holding If and Then.
	Count int
}

// Option configures Mine.
type Option func(*config)

type config struct {
	support    float64
	confidence float64
	maxItems   int
	top        int
}

// MinSupport sets the least share of baskets, 0 to 1, a set of items
// must appear in. The default is 0.01.
func MinSupport(s float64) Option { return func(c *config) { c.support = s } }

// MinConfidence sets the least confidence a rule must have. The default
// is 0.5.
func MinConfidence(f float64) Option { return func(c *config) { c.confidence = f } }

// MaxItems sets the most items in a rule, If and Then together. The
// default is 3; counting grows quickly with it.
func MaxItems(n int) Option { return func(c *config) { c.maxItems = n } }

// Top keeps only the n rules of highest lift. The default is 0, all.
func Top(n int) Option { return func(c *config) { c.top = n } }

// itemset is a sorted set of item numbers, as a string for map keys.
type itemset string

func key(items []int32) itemset {
	b := make([]byte, 0, 4*len(items))
	for _, it := range items {
		b = append(b, byte(it), byte(it>>8), byte(it>>16), byte(it>>24))
	}
	return itemset(b)
}

func items(k itemset) []int32 {
	out := make([]int32, len(k)/4)
	for i := range out {
		out[i] = int32(k[4*i]) | int32(k[4*i+1])<<8 | int32(k[4*i+2])<<16 | int32(k[4*i+3])<<24
	}
	return out
}

// Mine returns the rules of the baskets, by falling lift, then support.
func Mine(ctx context.Context, baskets [][]string, opts ...Option) ([]Rule, error) {
	c := &config{support: 0.01, confidence: 0.5, maxItems: 3}
	for _, o := range opts {
		o(c)
	}
	switch {
	case !(c.support > 0 && c.support <= 1):
		return nil, fmt.Errorf("rules: min support %v: %w", c.support, tensor.ErrShape)
	case !(c.confidence >= 0 && c.confidence <= 1):
		return nil, fmt.Errorf("rules: min confidence %v: %w", c.confidence, tensor.ErrShape)
	case c.maxItems < 2:
		return nil, fmt.Errorf("rules: max items %d: %w", c.maxItems, tensor.ErrShape)
	case len(baskets) == 0:
		return nil, fmt.Errorf("rules: no baskets: %w", tensor.ErrShape)
	}
	// Number the items and count them.
	number := map[string]int32{}
	var names []string
	rows := make([][]int32, len(baskets))
	single := map[int32]int{}
	for i, b := range baskets {
		seen := make([]int32, 0, len(b))
		for _, name := range b {
			id, ok := number[name]
			if !ok {
				id = int32(len(names))
				number[name] = id
				names = append(names, name)
			}
			if !slices.Contains(seen, id) {
				seen = append(seen, id)
			}
		}
		slices.Sort(seen)
		rows[i] = seen
		for _, id := range seen {
			single[id]++
		}
	}
	n := len(baskets)
	minCount := max(1, int(c.support*float64(n)+0.999999))
	// frequent holds the count of each frequent set, by level.
	frequent := map[itemset]int{}
	var level []itemset
	for id, cnt := range single {
		if cnt >= minCount {
			k := key([]int32{id})
			frequent[k] = cnt
			level = append(level, k)
		}
	}
	// Drop the rare items from the rows so the subsets stay small.
	for i, r := range rows {
		rows[i] = slices.DeleteFunc(r, func(id int32) bool { return single[id] < minCount })
	}
	for size := 2; size <= c.maxItems && len(level) > 0; size++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		// Candidates join two frequent sets that differ in their last
		// item, Apriori's way, and every subset of a candidate must be
		// frequent.
		slices.Sort(level)
		candidates := map[itemset]int{}
		for i, a := range level {
			ai := items(a)
			for _, b := range level[i+1:] {
				bi := items(b)
				if size > 2 && !slices.Equal(ai[:size-2], bi[:size-2]) {
					break
				}
				cand := append(slices.Clone(ai), bi[size-2])
				slices.Sort(cand)
				if subsetsFrequent(cand, frequent) {
					candidates[key(cand)] = 0
				}
			}
		}
		if len(candidates) == 0 {
			break
		}
		for i, r := range rows {
			if i%1024 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			if len(r) < size {
				continue
			}
			subsets(r, size, func(s []int32) {
				if k := key(s); hasKey(candidates, k) {
					candidates[k]++
				}
			})
		}
		level = level[:0]
		for k, cnt := range candidates {
			if cnt >= minCount {
				frequent[k] = cnt
				level = append(level, k)
			}
		}
	}
	// A rule for every frequent set of two or more, every item of it
	// as Then.
	var found []Rule
	for k, cnt := range frequent {
		set := items(k)
		if len(set) < 2 {
			continue
		}
		for i, then := range set {
			ante := slices.Concat(set[:i], set[i+1:])
			anteCount := frequent[key(ante)]
			if anteCount == 0 {
				continue
			}
			conf := float64(cnt) / float64(anteCount)
			if conf < c.confidence {
				continue
			}
			ifNames := make([]string, len(ante))
			for j, id := range ante {
				ifNames[j] = names[id]
			}
			found = append(found, Rule{
				If:         ifNames,
				Then:       names[then],
				Support:    float64(cnt) / float64(n),
				Confidence: conf,
				Lift:       conf / (float64(single[then]) / float64(n)),
				Count:      cnt,
			})
		}
	}
	sort.Slice(found, func(i, j int) bool {
		a, b := found[i], found[j]
		if a.Lift != b.Lift {
			return a.Lift > b.Lift
		}
		if a.Support != b.Support {
			return a.Support > b.Support
		}
		if c := slices.Compare(a.If, b.If); c != 0 {
			return c < 0
		}
		return a.Then < b.Then
	})
	if c.top > 0 && len(found) > c.top {
		found = found[:c.top]
	}
	return found, nil
}

func hasKey(m map[itemset]int, k itemset) bool {
	_, ok := m[k]
	return ok
}

// subsetsFrequent reports whether every subset of cand one item smaller
// is frequent.
func subsetsFrequent(cand []int32, frequent map[itemset]int) bool {
	buf := make([]int32, 0, len(cand)-1)
	for i := range cand {
		buf = append(buf[:0], cand[:i]...)
		buf = append(buf, cand[i+1:]...)
		if _, ok := frequent[key(buf)]; !ok {
			return false
		}
	}
	return true
}

// subsets calls f with every subset of size k of the sorted r, in order.
func subsets(r []int32, k int, f func([]int32)) {
	buf := make([]int32, k)
	var rec func(start, depth int)
	rec = func(start, depth int) {
		if depth == k {
			f(buf)
			return
		}
		for i := start; i <= len(r)-(k-depth); i++ {
			buf[depth] = r[i]
			rec(i+1, depth+1)
		}
	}
	rec(0, 0)
}
