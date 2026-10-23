//go:build !purego

package cpu

// tileAMX is in amx_amd64.s.
//
//go:noescape
func tileAMX(a *uint16, lda int, b *uint16, c *float32, ldc int, k int, cfg *byte, accumulate bool)

// cvtBF16 is in amx_amd64.s.
//
//go:noescape
func cvtBF16(dst *uint16, src *float32, n int)

func init() {
	if amxSupported() {
		amxTile, amxConvert = tileAMX, cvtBF16
	}
}

// amxSupported reports whether the CPU multiplies bfloat16 tiles with AMX
// and the operating system lets this process use them.
func amxSupported() bool {
	maxID, _, _, _ := cpuid(0, 0)
	if maxID < 0xd {
		return false
	}
	_, _, _, edx7 := cpuid(7, 0)
	const amxBF16, amxTile = 1 << 22, 1 << 24
	if edx7&amxBF16 == 0 || edx7&amxTile == 0 {
		return false
	}
	// VCVTNEPS2BF16, which packing uses, is AVX512_BF16.
	eax71, _, _, _ := cpuid(7, 1)
	const avx512BF16 = 1 << 5
	if eax71&avx512BF16 == 0 || !x86Features().avx512 {
		return false
	}
	return amxPermitted()
}
