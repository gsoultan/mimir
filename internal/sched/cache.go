package sched

import (
	"slices"
	"sync/atomic"
)

// Each pool caches the programs of its recent schedules by key (see
// key.go), so that a schedule of a graph it has seen, such as every
// training step after the first, skips sorting, fusing, planning its
// memory and compiling, and runs the program it compiled the first time
// with the new graph's buffers and arguments. A key holds everything a
// plan depends on, and a lookup compares it word for word, so a graph
// that differs in anything but its buffers, constants and seeds compiles
// a program of its own. Whether a run fits the memory budget is decided
// at each run, so a change of budget needs no new program.

// maxPrograms is the most programs a pool caches. A training loop runs a
// few schedules over and over: the step, the optimizer's update, an
// evaluation.
const maxPrograms = 64

// caching turns the cache on and off, for tests and benchmarks.
var caching = true

// SetCaching turns the schedule cache on or off and returns the previous
// setting. With it off every schedule is planned and compiled anew.
func SetCaching(on bool) (was bool) {
	mu.Lock()
	defer mu.Unlock()
	was, caching = caching, on
	return was
}

// compiled counts the programs compiled, cached or not, for tests.
var compiled atomic.Int64

// Compiled returns the number of programs compiled so far, for tests that
// check a schedule replays its program.
func Compiled() int {
	return int(compiled.Load())
}

// cache holds the programs of a pool.
type cache struct {
	byHash map[uint64][]*cached
	n      int
	clock  uint64

	hits, misses int // lookups, for tests
}

type cached struct {
	key  []uint64
	prog *program
	used uint64 // the clock when it last ran
}

// lookup returns the program of key, or nil.
func (c *cache) lookup(hash uint64, key []uint64) *program {
	for _, e := range c.byHash[hash] {
		if slices.Equal(e.key, key) {
			c.clock++
			e.used = c.clock
			c.hits++
			return e.prog
		}
	}
	c.misses++
	return nil
}

// add caches prog under key, making room by dropping the program run
// longest ago.
func (c *cache) add(hash uint64, key []uint64, prog *program) {
	if c.byHash == nil {
		c.byHash = map[uint64][]*cached{}
	}
	if c.n >= maxPrograms {
		c.evict()
	}
	c.clock++
	c.byHash[hash] = append(c.byHash[hash], &cached{slices.Clone(key), prog, c.clock})
	c.n++
}

// evict drops the program run longest ago.
func (c *cache) evict() {
	var oldest *cached
	var at uint64
	for h, es := range c.byHash {
		for _, e := range es {
			if oldest == nil || e.used < oldest.used {
				oldest, at = e, h
			}
		}
	}
	if oldest == nil {
		return
	}
	es := slices.DeleteFunc(c.byHash[at], func(e *cached) bool { return e == oldest })
	if len(es) == 0 {
		delete(c.byHash, at)
	} else {
		c.byHash[at] = es
	}
	c.n--
}
