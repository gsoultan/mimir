package data

import "testing"

func TestChunkRows(t *testing.T) {
	for _, tc := range []struct{ features, want int }{
		{1, 8192},
		{32, 8192},
		{127, 8192},
		{128, 8128},
		{784, 1335},
		{1024, 1023},
		{3072, 341},
		{4096, 255}, // below the batch of TestStreamBatchLargerThanChunk
		{1 << 20, 1},
		{1 << 24, 1},
	} {
		got := chunkRows(tc.features)
		if got != tc.want {
			t.Errorf("chunkRows(%d) = %d, want %d", tc.features, got, tc.want)
		}
		if size := 4 * (tc.features + 1) * got; size > chunkBytes && got > 1 {
			t.Errorf("chunkRows(%d) makes chunks of %d bytes, over %d", tc.features, size, chunkBytes)
		}
	}
}
