package cpu

import (
	"runtime"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/gsoultan/mimir/backend"
)

func TestMapClassesBoundTheirWaste(t *testing.T) {
	prev := 0
	for size := offHeapMin; size < 1<<34; size += size/7 + 4093 {
		c := mapClass(size)
		if c < size || c%pageSize != 0 {
			t.Fatalf("class of %d bytes is %d", size, c)
		}
		if waste := c - size; waste > size/8+pageSize {
			t.Fatalf("class of %d bytes is %d, %d bytes too many", size, c, waste)
		}
		if c < prev {
			t.Fatalf("class of %d bytes is %d, smaller than a smaller request's %d", size, c, prev)
		}
		if mapClass(c) != c {
			t.Fatalf("class %d is not its own class (%d)", c, mapClass(c))
		}
		prev = c
	}
}

// counts returns the cache's hits and misses.
func counts() (hits, misses int64) {
	mappings.mu.Lock()
	defer mappings.mu.Unlock()
	return mappings.hits, mappings.misses
}

// dataOf returns the address and elements of a float32 buffer.
func dataOf(b backend.Buffer) (uintptr, []float32) {
	d := b.(*buffer[float32]).data
	return uintptr(unsafe.Pointer(&d[0])), d
}

func TestFreedMappingsServeLaterAllocs(t *testing.T) {
	if !offHeap {
		t.Skipf("no anonymous mappings on %s", runtime.GOOS)
	}
	b := New()
	// Sizes of their own, which no other test frees, in one class.
	const n = (5<<20 + 12344) / 4
	first, err := b.Alloc(backend.Float32, n)
	if err != nil {
		t.Fatal(err)
	}
	at, data := dataOf(first)
	for i := range data {
		data[i] = 7
	}
	b.Free(first)
	hits, _ := counts()
	second, err := b.Alloc(backend.Float32, n-1000)
	if err != nil {
		t.Fatal(err)
	}
	if h, _ := counts(); h != hits+1 {
		t.Fatal("a freed mapping of the class did not serve the next Alloc")
	}
	again, data := dataOf(second)
	if again != at {
		t.Errorf("reused mapping at %#x, want the freed one at %#x", again, at)
	}
	for i, v := range data {
		if v != 0 {
			t.Fatalf("element %d of a reused mapping = %v, want 0", i, v)
		}
	}
	if OffHeapBytes() < int64(4*(n-1000)) {
		t.Errorf("a reused mapping is not counted off the heap")
	}
	b.Free(second)
}

func TestMapCacheIsBounded(t *testing.T) {
	if !offHeap {
		t.Skipf("no anonymous mappings on %s", runtime.GOOS)
	}
	b := New()
	// A class whose one mapping an Alloc took, then more mappings than
	// the cache holds.
	for range 2 {
		buf, err := b.Alloc(backend.Float32, 7<<20/4)
		if err != nil {
			t.Fatal(err)
		}
		b.Free(buf)
	}
	const n = 9 << 20 / 4 // 9 MiB, untouched
	var bufs []backend.Buffer
	for range 2 * mapCacheMin / (4 * n) {
		buf, err := b.Alloc(backend.Float32, n)
		if err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs, buf)
	}
	for _, buf := range bufs {
		b.Free(buf)
	}
	if got, limit := CachedMapBytes(), max(int64(mapCacheMin), OffHeapBytes()/2); got > limit {
		t.Errorf("%d MiB cached, over the limit of %d MiB", got>>20, limit>>20)
	}
}

func TestMapCacheEmptiesOverCollections(t *testing.T) {
	if !offHeap {
		t.Skipf("no anonymous mappings on %s", runtime.GOOS)
	}
	b := New()
	buf, err := b.Alloc(backend.Float32, 3<<20/4)
	if err != nil {
		t.Fatal(err)
	}
	b.Free(buf)
	if CachedMapBytes() == 0 {
		t.Fatal("a freed mapping was not cached")
	}
	deadline := time.Now().Add(10 * time.Second)
	for CachedMapBytes() != 0 {
		if time.Now().After(deadline) {
			t.Fatalf("%d bytes still cached after collections", CachedMapBytes())
		}
		runtime.GC()
		time.Sleep(time.Millisecond)
	}
}

func TestDroppedBuffersLeaveTheirMappingsToTheCache(t *testing.T) {
	if !offHeap {
		t.Skipf("no anonymous mappings on %s", runtime.GOOS)
	}
	settle(t)
	b := New()
	func() {
		if _, err := b.Alloc(backend.Float32, (6<<20+4096)/4); err != nil {
			t.Fatal(err)
		}
	}()
	hits, _ := counts()
	runtime.GC()
	deadline := time.Now().Add(10 * time.Second)
	for {
		buf, err := b.Alloc(backend.Float32, (6<<20+4096)/4)
		if err != nil {
			t.Fatal(err)
		}
		b.Free(buf)
		if h, _ := counts(); h > hits {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("the mapping of a collected buffer never served an Alloc")
		}
		runtime.GC()
		time.Sleep(time.Millisecond)
	}
}

// TestMapCacheIsSafeForConcurrentUse allocates, writes and frees buffers
// from several goroutines while collections age the cache, for the race
// detector.
func TestMapCacheIsSafeForConcurrentUse(t *testing.T) {
	b := New()
	var wg sync.WaitGroup
	for g := range 4 {
		wg.Go(func() {
			for i := range 40 {
				n := (1<<20 + (g*40+i)%7*300000) / 4
				buf, err := b.Alloc(backend.Float32, n)
				if err != nil {
					t.Error(err)
					return
				}
				data := buf.(*buffer[float32]).data
				for j := 0; j < len(data); j += 997 {
					if data[j] != 0 {
						t.Errorf("element %d of a new buffer = %v", j, data[j])
						return
					}
					data[j] = float32(g)
				}
				if i%3 == 0 {
					runtime.GC()
				} else {
					b.Free(buf)
				}
			}
		})
	}
	wg.Wait()
}
