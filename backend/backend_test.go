package backend

import "testing"

func TestDTypes(t *testing.T) {
	cases := []struct {
		dt    DType
		size  int
		float bool
		name  string
	}{
		{Float32, 4, true, "float32"},
		{Int32, 4, false, "int32"},
		{BFloat16, 2, true, "bfloat16"},
		{Float16, 2, true, "float16"},
		{Invalid, 0, false, "DType(0)"},
	}
	for _, c := range cases {
		if c.dt.Size() != c.size || c.dt.IsFloat() != c.float || c.dt.String() != c.name {
			t.Errorf("%s: Size %d, IsFloat %v, want %d, %v, %q", c.dt, c.dt.Size(), c.dt.IsFloat(), c.size, c.float, c.name)
		}
	}
}
