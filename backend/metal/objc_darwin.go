//go:build darwin && arm64

package metal

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// The Objective-C runtime and Metal are called through purego, without
// cgo: every message is objc_msgSend with integer and pointer arguments.
// On arm64 a struct argument larger than 16 bytes, such as MTLSize, is
// passed as a pointer to a copy, so no message needs more.

// id is an Objective-C object pointer.
type id uintptr

// sel is a registered selector.
type sel uintptr

var (
	objcOnce sync.Once
	objcErr  error

	msgSend    uintptr
	selRegName func(name string) sel
	getClass   func(name string) id
	poolPush   func() uintptr
	poolPop    func(uintptr)
	createDev  func() id
)

// loadObjC loads libobjc and Metal once.
func loadObjC() error {
	objcOnce.Do(func() {
		objc, err := purego.Dlopen("/usr/lib/libobjc.A.dylib", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			objcErr = err
			return
		}
		if _, err := purego.Dlopen("/System/Library/Frameworks/Foundation.framework/Foundation", purego.RTLD_GLOBAL|purego.RTLD_NOW); err != nil {
			objcErr = err
			return
		}
		mtl, err := purego.Dlopen("/System/Library/Frameworks/Metal.framework/Metal", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			objcErr = err
			return
		}
		if msgSend, err = purego.Dlsym(objc, "objc_msgSend"); err != nil {
			objcErr = err
			return
		}
		purego.RegisterLibFunc(&selRegName, objc, "sel_registerName")
		purego.RegisterLibFunc(&getClass, objc, "objc_getClass")
		purego.RegisterLibFunc(&poolPush, objc, "objc_autoreleasePoolPush")
		purego.RegisterLibFunc(&poolPop, objc, "objc_autoreleasePoolPop")
		purego.RegisterLibFunc(&createDev, mtl, "MTLCreateSystemDefaultDevice")
		initSelectors()
	})
	return objcErr
}

// send sends the message s to receiver r and returns the result.
//
//go:uintptrescapes
func send(r id, s sel, args ...uintptr) uintptr {
	all := make([]uintptr, 0, 2+len(args))
	all = append(all, uintptr(r), uintptr(s))
	all = append(all, args...)
	r1, _, _ := purego.SyscallN(msgSend, all...)
	return r1
}

// Selectors, registered once.
var (
	selAlloc, selInit, selNew, selRelease, selRetain                               sel
	selInitWithBytes, selUTF8String, selLocalizedDescription                       sel
	selName, selNewCommandQueue, selNewBuffer, selContents                         sel
	selSetFastMath, selNewLibrary, selNewFunction, selNewPipeline, selMaxThreads   sel
	selCommandBuffer, selComputeEncoder, selSetPipeline, selSetBuffer, selSetBytes sel
	selDispatch, selEndEncoding, selCommit, selWait, selStatus, selError           sel
	selHasUnifiedMemory, selRecommendedWorkingSetSize, selSupportsFamily           sel
)

func initSelectors() {
	for p, name := range map[*sel]string{
		&selAlloc:                     "alloc",
		&selInit:                      "init",
		&selNew:                       "new",
		&selRelease:                   "release",
		&selRetain:                    "retain",
		&selInitWithBytes:             "initWithBytes:length:encoding:",
		&selUTF8String:                "UTF8String",
		&selLocalizedDescription:      "localizedDescription",
		&selName:                      "name",
		&selNewCommandQueue:           "newCommandQueue",
		&selNewBuffer:                 "newBufferWithLength:options:",
		&selContents:                  "contents",
		&selSetFastMath:               "setFastMathEnabled:",
		&selNewLibrary:                "newLibraryWithSource:options:error:",
		&selNewFunction:               "newFunctionWithName:",
		&selNewPipeline:               "newComputePipelineStateWithFunction:error:",
		&selMaxThreads:                "maxTotalThreadsPerThreadgroup",
		&selCommandBuffer:             "commandBuffer",
		&selComputeEncoder:            "computeCommandEncoder",
		&selSetPipeline:               "setComputePipelineState:",
		&selSetBuffer:                 "setBuffer:offset:atIndex:",
		&selSetBytes:                  "setBytes:length:atIndex:",
		&selDispatch:                  "dispatchThreadgroups:threadsPerThreadgroup:",
		&selEndEncoding:               "endEncoding",
		&selCommit:                    "commit",
		&selWait:                      "waitUntilCompleted",
		&selStatus:                    "status",
		&selError:                     "error",
		&selHasUnifiedMemory:          "hasUnifiedMemory",
		&selRecommendedWorkingSetSize: "recommendedMaxWorkingSetSize",
		&selSupportsFamily:            "supportsFamily:",
	} {
		*p = selRegName(name)
	}
}

// nsString returns a new NSString holding s, which the caller releases.
func nsString(s string) id {
	const utf8 = 4 // NSUTF8StringEncoding
	b := append([]byte(s), 0)
	str := id(send(id(getClass("NSString")), selAlloc))
	str = id(send(str, selInitWithBytes, uintptr(unsafe.Pointer(&b[0])), uintptr(len(s)), utf8))
	runtime.KeepAlive(b)
	return str
}

// goString returns the contents of the NSString s.
func goString(s id) string {
	if s == 0 {
		return ""
	}
	p := cPointer(send(s, selUTF8String))
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(p, n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(p), n))
}

// cPointer converts the address of memory outside the Go heap, which a
// message returned, to a pointer.
func cPointer(addr uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&addr))
}

// nsError returns the description of the NSError e as an error.
func nsError(what string, e id) error {
	if e == 0 {
		return errors.New(what)
	}
	return fmt.Errorf("%s: %s", what, goString(id(send(e, selLocalizedDescription))))
}

// autoreleased runs fn on one OS thread inside an autorelease pool, which
// drains the objects Metal returns autoreleased, such as command buffers.
func autoreleased(fn func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	p := poolPush()
	defer poolPop(p)
	fn()
}
