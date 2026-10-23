//go:build !purego

#include "textflag.h"

// The Go assembler has no AMX instructions, so they are spelled out as
// bytes. Each is VEX-encoded with RAX..RDI only, so the VEX register
// extension bits are all clear: C4 E2, then W0, vvvv, L0 and the prefix
// (78 none, 7A F3, 7B F2), the opcode, ModRM, SIB and any displacement.

// func tileAMX(a *uint16, lda int, b *uint16, c *float32, ldc int, k int, cfg *byte, accumulate bool)
//
// tileAMX sets the 32×32 float32 tile at c, whose rows are ldc bytes
// apart, to the product of the bfloat16 matrices a, 32 rows of k elements
// lda bytes apart, and b, k rows of 32 columns with each pair of rows
// interleaved (VNNI order: element (r, j) at (r/2*32+j)*2 + r%2), adding
// it to the tile's contents if accumulate is set. k is a positive
// multiple of 32. cfg is the tile configuration it loads first; it
// releases the tiles before it returns.
TEXT ·tileAMX(SB), NOSPLIT, $0-57
	MOVQ cfg+48(FP), AX
	// LDTILECFG (AX)
	BYTE $0xC4; BYTE $0xE2; BYTE $0x78; BYTE $0x49; BYTE $0x00

	MOVQ c+24(FP), DI
	MOVQ ldc+32(FP), CX
	MOVQ CX, BX
	SHLQ $4, BX
	ADDQ DI, BX
	MOVB accumulate+56(FP), R10
	TESTB R10, R10
	JZ zero
	// TILELOADD tmm0, (DI)(CX*1)
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x4B; BYTE $0x04; BYTE $0x0F
	// TILELOADD tmm1, 64(DI)(CX*1)
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x4B; BYTE $0x4C; BYTE $0x0F; BYTE $0x40
	// TILELOADD tmm2, (BX)(CX*1)
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x4B; BYTE $0x14; BYTE $0x0B
	// TILELOADD tmm3, 64(BX)(CX*1)
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x4B; BYTE $0x5C; BYTE $0x0B; BYTE $0x40
	JMP loop0
zero:
	// TILEZERO tmm0..tmm3
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x49; BYTE $0xC0
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x49; BYTE $0xC8
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x49; BYTE $0xD0
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x49; BYTE $0xD8
loop0:
	MOVQ a+0(FP), AX
	MOVQ lda+8(FP), CX
	MOVQ CX, BX
	SHLQ $4, BX
	ADDQ AX, BX
	MOVQ b+16(FP), SI
	MOVQ $128, DX
	MOVQ k+40(FP), R9
loop:
	// TILELOADD tmm4, (AX)(CX*1)
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x4B; BYTE $0x24; BYTE $0x08
	// TILELOADD tmm5, (BX)(CX*1)
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x4B; BYTE $0x2C; BYTE $0x0B
	// TILELOADD tmm6, (SI)(DX*1)
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x4B; BYTE $0x34; BYTE $0x16
	// TILELOADD tmm7, 64(SI)(DX*1)
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7B; BYTE $0x4B; BYTE $0x7C; BYTE $0x16; BYTE $0x40
	// TDPBF16PS tmm0, tmm4, tmm6
	BYTE $0xC4; BYTE $0xE2; BYTE $0x4A; BYTE $0x5C; BYTE $0xC4
	// TDPBF16PS tmm1, tmm4, tmm7
	BYTE $0xC4; BYTE $0xE2; BYTE $0x42; BYTE $0x5C; BYTE $0xCC
	// TDPBF16PS tmm2, tmm5, tmm6
	BYTE $0xC4; BYTE $0xE2; BYTE $0x4A; BYTE $0x5C; BYTE $0xD5
	// TDPBF16PS tmm3, tmm5, tmm7
	BYTE $0xC4; BYTE $0xE2; BYTE $0x42; BYTE $0x5C; BYTE $0xDD
	ADDQ $64, AX
	ADDQ $64, BX
	ADDQ $2048, SI
	SUBQ $32, R9
	JNZ loop

	MOVQ c+24(FP), DI
	MOVQ ldc+32(FP), CX
	MOVQ CX, BX
	SHLQ $4, BX
	ADDQ DI, BX
	// TILESTORED (DI)(CX*1), tmm0
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7A; BYTE $0x4B; BYTE $0x04; BYTE $0x0F
	// TILESTORED 64(DI)(CX*1), tmm1
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7A; BYTE $0x4B; BYTE $0x4C; BYTE $0x0F; BYTE $0x40
	// TILESTORED (BX)(CX*1), tmm2
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7A; BYTE $0x4B; BYTE $0x14; BYTE $0x0B
	// TILESTORED 64(BX)(CX*1), tmm3
	BYTE $0xC4; BYTE $0xE2; BYTE $0x7A; BYTE $0x4B; BYTE $0x5C; BYTE $0x0B; BYTE $0x40
	// TILERELEASE
	BYTE $0xC4; BYTE $0xE2; BYTE $0x78; BYTE $0x49; BYTE $0xC0
	RET

// func cvtBF16(dst *uint16, src *float32, n int)
//
// cvtBF16 sets the n elements at dst, a positive multiple of 16, to those
// at src rounded to bfloat16, ties to even, with VCVTNEPS2BF16, which
// reads subnormal inputs as zero.
TEXT ·cvtBF16(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX

cvt:
	VMOVUPS (SI), Z0
	// VCVTNEPS2BF16 Y1, Z0: EVEX.512.F3.0F38.W0 72 /r
	BYTE $0x62; BYTE $0xF2; BYTE $0x7E; BYTE $0x48; BYTE $0x72; BYTE $0xC8
	VMOVDQU Y1, (DI)
	ADDQ    $64, SI
	ADDQ    $32, DI
	SUBQ    $16, CX
	JNZ     cvt
	VZEROUPPER
	RET
