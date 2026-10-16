//go:build linux

package cpu

import "syscall"

// madvFree is MADV_FREE, which the syscall package does not name on
// every architecture.
const madvFree = 8

// lazyFree lets the kernel take the pages of mem, a mapping that no
// buffer uses, when it runs short of memory, without unmapping it. A page
// it takes reads as zero afterwards; one it leaves keeps its contents and
// costs no fault when it is written again. A kernel older than Linux 4.5
// does not know MADV_FREE, and the pages stay resident.
func lazyFree(mem []byte) { _ = syscall.Madvise(mem, madvFree) }
