//go:build !purego

package cpu

import "syscall"

// amxPermitted asks Linux for the AMX tile state. It is too large for the
// kernel to save for every process, so each asks for it first.
func amxPermitted() bool {
	const archReqXcompPerm, xfeatureXtiledata = 0x1023, 18
	_, _, errno := syscall.RawSyscall(syscall.SYS_ARCH_PRCTL, archReqXcompPerm, xfeatureXtiledata, 0)
	return errno == 0
}
