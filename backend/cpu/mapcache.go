package cpu

import (
	"math/bits"
	"os"
	"runtime"
	"sync"
)

// Mappings are reused. A mapped buffer that Free frees, or that the
// garbage collector finds unreachable, leaves its mapping in a cache, and
// Alloc takes a mapping of the same size class from the cache before it
// maps a new one. A new mapping costs a page fault, and the kernel's
// zeroing, for every page the buffer touches, which for a buffer of
// megabytes costs more than most kernels' work over it, and a step that
// uploads a batch, or returns a large result, maps new memory every
// time. A reused mapping's pages are resident already, so Alloc only
// clears the bytes the buffer covers.
//
// The cache is bounded. Sizes round up to one of eight classes per power
// of two, so that a mapping serves requests up to an eighth smaller. The
// cache holds at most mapCacheMin bytes, or half the bytes of the
// buffers in use if that is more, and unmaps the mappings cached longest
// ago to stay within that. A mapping that no Alloc takes before
// mapCacheAge garbage collections have passed is unmapped, so memory
// that one phase of a program used goes back to the operating system
// when another phase follows. On Linux each cached mapping is also
// marked MADV_FREE, which lets the kernel take its pages under memory
// pressure without unmapping it; a page taken reads as zero when the
// mapping is reused, and costs a fault again.

// mapCacheMin is the most bytes the cache holds when the buffers in use
// hold less than twice as much.
const mapCacheMin = 64 << 20

// mapClassBits is the number of bits after the leading one that a size
// class keeps: eight classes per power of two.
const mapClassBits = 3

// mapCacheAge is how many garbage collections a cached mapping waits
// for an Alloc before it is unmapped.
const mapCacheAge = 2

var pageSize = os.Getpagesize()

// mapClass returns the size of the class of a request of size bytes: size
// rounded up to whole pages, and then to the class, which keeps the
// leading mapClassBits+1 bits.
func mapClass(size int) int {
	size = (size + pageSize - 1) &^ (pageSize - 1)
	shift := bits.Len(uint(size)) - 1 - mapClassBits
	if shift <= 0 {
		return size
	}
	step := 1 << shift
	return (size + step - 1) &^ (step - 1)
}

// mapping is a cached anonymous mapping and the function that unmaps it.
type mapping struct {
	mem   []byte // the whole mapping, of a class size
	unmap func() error
	seq   uint64 // when it was cached, in puts
	gc    uint64 // when it was cached, in collections
}

// mapCache holds mappings for reuse. It is safe for concurrent use: the
// buffers whose mappings it takes are freed by Free and by cleanups.
type mapCache struct {
	mu     sync.Mutex
	free   map[int][]mapping // by class size, in the order cached
	bytes  int64             // in all the mappings of free
	puts   uint64
	gcs    uint64 // collections counted since the cache was first used
	armed  bool   // the next collection ages the cache
	hits   int64  // Allocs served from the cache, for tests
	misses int64  // Allocs that mapped new memory, for tests
}

var mappings mapCache

// CachedMapBytes returns the number of bytes of anonymous mappings that
// the CPU backend keeps for reuse (see Alloc). They are not counted by
// OffHeapBytes.
func CachedMapBytes() int64 {
	mappings.mu.Lock()
	defer mappings.mu.Unlock()
	return mappings.bytes
}

// take returns a cached mapping of class bytes and reports whether there
// was one, or else maps a new one.
func (c *mapCache) take(class int) (mem []byte, unmap func() error, reused bool, err error) {
	c.mu.Lock()
	if ms := c.free[class]; len(ms) > 0 {
		m := ms[len(ms)-1]
		ms[len(ms)-1] = mapping{}
		if len(ms) == 1 {
			delete(c.free, class) // every list in free holds a mapping
		} else {
			c.free[class] = ms[:len(ms)-1]
		}
		c.bytes -= int64(class)
		c.hits++
		c.mu.Unlock()
		return m.mem, m.unmap, true, nil
	}
	c.misses++
	c.mu.Unlock()
	mem, unmap, err = mapAnon(class)
	return mem, unmap, false, err
}

// put caches mem, a whole mapping of a class size that no buffer uses
// any more, or unmaps it if it is larger than the cache may hold. To make
// room it unmaps the mappings cached longest ago.
func (c *mapCache) put(mem []byte, unmap func() error) {
	size := int64(len(mem))
	limit := max(mapCacheMin, offHeapBytes.Load()/2)
	if size > limit {
		_ = unmap()
		return
	}
	lazyFree(mem)
	c.mu.Lock()
	var drop []mapping
	for c.bytes+size > limit {
		drop = append(drop, c.evictOldest())
	}
	if c.free == nil {
		c.free = map[int][]mapping{}
	}
	c.puts++
	c.free[len(mem)] = append(c.free[len(mem)], mapping{mem, unmap, c.puts, c.gcs})
	c.bytes += size
	if !c.armed {
		c.armed = true
		armAging(c)
	}
	c.mu.Unlock()
	for _, m := range drop {
		_ = m.unmap()
	}
}

// evictOldest takes the mapping cached longest ago out of the cache,
// which holds one, and returns it for the caller to unmap. The caller
// holds mu.
func (c *mapCache) evictOldest() mapping {
	class := -1
	for k, ms := range c.free {
		if class < 0 || ms[0].seq < c.free[class][0].seq {
			class = k
		}
	}
	ms := c.free[class]
	m := ms[0]
	if len(ms) == 1 {
		delete(c.free, class)
	} else {
		ms[0] = mapping{}
		c.free[class] = ms[1:]
	}
	c.bytes -= int64(class)
	return m
}

// sentinel is an object whose collection marks a garbage collection. It
// is too large to share a tiny allocation with other objects, which
// would keep it alive as long as them.
type sentinel struct{ _ [16]byte }

// armAging has the next garbage collection age c.
func armAging(c *mapCache) {
	runtime.AddCleanup(new(sentinel), func(c *mapCache) { c.age() }, c)
}

// age counts a garbage collection and unmaps the mappings that have
// waited mapCacheAge collections, and has the next collection age the
// cache again while it holds any.
func (c *mapCache) age() {
	c.mu.Lock()
	c.gcs++
	var drop []mapping
	for class, ms := range c.free {
		old := 0
		for old < len(ms) && c.gcs-ms[old].gc >= mapCacheAge {
			old++
		}
		if old == 0 {
			continue
		}
		drop = append(drop, ms[:old]...)
		c.bytes -= int64(old * class)
		if old == len(ms) {
			delete(c.free, class)
			continue
		}
		n := copy(ms, ms[old:])
		clear(ms[n:])
		c.free[class] = ms[:n]
	}
	if c.armed = len(c.free) > 0; c.armed {
		armAging(c)
	}
	c.mu.Unlock()
	for _, m := range drop {
		_ = m.unmap()
	}
}
