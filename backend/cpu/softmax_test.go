package cpu

import (
	"math"
	"math/rand/v2"
	"testing"
)

// withRoutines runs f with each set of row routines this CPU has, and
// with none, under its name.
func withRoutines(t *testing.T, f func(t *testing.T)) {
	saved := rowVector
	defer func() { rowVector = saved }()
	sets := rowRoutineSets()
	if sets == nil {
		sets = map[string]*rowRoutines{}
	}
	sets["go"] = nil
	for name, r := range sets {
		rowVector = r
		t.Run(name, f)
	}
}

func TestExpShift(t *testing.T) {
	withRoutines(t, func(t *testing.T) {
		r := rand.New(rand.NewPCG(1, 2))
		for _, n := range []int{1, 7, 8, 16, 37, 64, 129} {
			x := make([]float32, n)
			for i := range x {
				x[i] = r.Float32()*120 - 100
			}
			x[0] = float32(math.Inf(-1))
			if n > 3 {
				x[3] = 3 // above the shift
			}
			want := make([]float64, n)
			var sum float64
			for i, v := range x {
				d := float64(v - 2) // as rounded in float32
				want[i] = math.Exp(d)
				if d < -87.33654 {
					want[i] = 0
				}
				sum += want[i]
			}
			got := expShift(x, 2)
			for i, w := range want {
				if math.Abs(float64(x[i])-w) > 4e-7*w {
					t.Errorf("n=%d: e^(%d) = %v, want %v", n, i, x[i], w)
				}
			}
			if math.Abs(float64(got)-sum) > 1e-6*sum {
				t.Errorf("n=%d: sum %v, want %v", n, got, sum)
			}
		}
	})
}

func TestRowOperations(t *testing.T) {
	withRoutines(t, func(t *testing.T) {
		r := rand.New(rand.NewPCG(3, 4))
		for _, n := range []int{1, 9, 16, 33, 128} {
			x, dp := make([]float32, n), make([]float32, n)
			for i := range x {
				x[i], dp[i] = r.Float32()*4-2, r.Float32()*4-2
			}
			x[n/2] = 5
			if got := rowMax(x); got != 5 {
				t.Errorf("n=%d: max %v, want 5", n, got)
			}
			p := append([]float32(nil), x...)
			softmaxGrad(p, dp, 0.5)
			s := append([]float32(nil), x...)
			scaleRow(s, 3)
			for i := range x {
				if want := x[i] * (dp[i] - 0.5); p[i] != want {
					t.Errorf("n=%d: softmax gradient %d = %v, want %v", n, i, p[i], want)
				}
				if s[i] != 3*x[i] {
					t.Errorf("n=%d: scaled %d = %v, want %v", n, i, s[i], 3*x[i])
				}
			}
		}
	})
}
