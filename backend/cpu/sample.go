package cpu

import (
	"context"
	"fmt"
	"math"

	"github.com/gsoultan/mimir/backend"
)

// The samplers draw each element from a stream of Philox blocks of its
// own, (i, j, 0, word) for element i and draw j, so the elements are
// independent of each other and of how the work is split, and need not
// agree on how many draws each takes. They compute in float64.

// Words that tell the samplers' streams apart from each other and from
// OpRandUniform and OpRandNormal, whose blocks end in 0.
const (
	gammaWord   = 0x67616d6d // "gamm"
	poissonWord = 0x706f6973 // "pois"
)

// sampleGrain is the fewest elements worth a goroutine: a sample costs
// tens of nanoseconds, an implicit gradient a few hundred.
const sampleGrain = 1 << 10

// maxDraws bounds the rejection loops. Each accepts a draw with
// probability over a half, so it is never reached; it guards against an
// endless loop on a parameter no test foresaw.
const maxDraws = 1 << 12

// stream is the Philox stream of one element.
type stream struct {
	key  [2]uint32
	i    int
	word uint32
}

// block returns the four words of draw j.
func (s *stream) block(j int) [4]uint32 {
	return philox([4]uint32{uint32(uint64(s.i)), uint32(uint64(s.i) >> 32), uint32(j), s.word}, s.key)
}

// open returns w as a uniform float64 in (0, 1).
func open(w uint32) float64 { return (float64(w) + 0.5) / (1 << 32) }

// runSample runs OpRandGamma, OpRandPoisson or OpGammaGrad.
func runSample(ctx context.Context, b *Backend, k *backend.Kernel, outBuf backend.Buffer, ins []backend.Buffer) error {
	out, err := data[float32](outBuf)
	if err != nil {
		return err
	}
	in := make([][]float32, len(ins))
	for i, buf := range ins {
		if in[i], err = data[float32](buf); err != nil {
			return err
		}
	}
	if len(in) != 1+boolInt(k.Op == backend.OpGammaGrad) {
		return fmt.Errorf("%s of %d inputs: %w", k.Op, len(in), backend.ErrInternal)
	}
	key := [2]uint32{uint32(k.Attrs.Seed), uint32(k.Attrs.Seed >> 32)}
	shape, views := backend.Coalesce(k.Shape, k.In)
	return b.parallelFor(ctx, len(out), sampleGrain, func(lo, hi int) error {
		c := newCursor(shape, views, lo)
		for i := lo; i < hi; i++ {
			x := float64(in[0][c.offs[0]])
			var v float64
			switch k.Op {
			case backend.OpRandGamma:
				v = max(sampleGamma(x, &stream{key: key, i: i, word: gammaWord}), 0x1p-126)
			case backend.OpRandPoisson:
				v = samplePoisson(x, &stream{key: key, i: i, word: poissonWord})
			default:
				v = gammaGrad(x, float64(in[1][c.offs[1]]))
			}
			out[i] = float32(v)
			c.next()
		}
		return nil
	})
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

// sampleGamma draws from Gamma(alpha, 1) by Marsaglia and Tsang's
// method, "A simple method for generating gamma variables" (2000): draw j
// takes a normal from words 0 and 1 of block j by Box-Muller and the
// uniform of its test from word 2. For alpha < 1 it draws from
// Gamma(alpha+1) and multiplies by U^(1/alpha), with U word 3 of block 0.
// A parameter that is not positive gives NaN.
func sampleGamma(alpha float64, s *stream) float64 {
	switch {
	case !(alpha > 0):
		return math.NaN()
	case math.IsInf(alpha, 1):
		return alpha
	}
	a, boost := alpha, 1.0
	if alpha < 1 {
		a++
		boost = math.Exp(math.Log(open(s.block(0)[3])) / alpha)
	}
	d := a - 1.0/3
	c := 1 / math.Sqrt(9*d)
	for j := range maxDraws {
		w := s.block(j)
		z := math.Sqrt(-2*math.Log(open(w[0]))) * math.Cos(2*math.Pi*open(w[1]))
		v := 1 + c*z
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := open(w[2])
		z2 := z * z
		if u < 1-0.0331*z2*z2 || math.Log(u) < 0.5*z2+d*(1-v+math.Log(v)) {
			return d * v * boost
		}
	}
	return d * boost
}

// samplePoisson draws from Poisson(lambda): below 10 by multiplying
// uniforms, four words a block, until their product falls below
// e^-lambda, and above by the transformed rejection with squeeze of
// Hörmann, "The transformed rejection method for generating Poisson
// random variables" (1993), as NumPy and PyTorch do, with the two
// uniforms of draw j words 0 and 1 of block j. A negative or NaN
// parameter gives NaN.
func samplePoisson(lambda float64, s *stream) float64 {
	switch {
	case lambda == 0:
		return 0
	case !(lambda > 0):
		return math.NaN()
	case math.IsInf(lambda, 1):
		return lambda
	}
	if lambda < 10 {
		limit, p := math.Exp(-lambda), 1.0
		for j := range maxDraws {
			for l, w := range s.block(j) {
				if p *= open(w); p <= limit {
					return float64(4*j + l)
				}
			}
		}
		return math.Floor(lambda)
	}
	slam, loglam := math.Sqrt(lambda), math.Log(lambda)
	b := 0.931 + 2.53*slam
	a := -0.059 + 0.02483*b
	invAlpha := 1.1239 + 1.1328/(b-3.4)
	vr := 0.9277 - 3.6224/(b-2)
	for j := range maxDraws {
		w := s.block(j)
		u, v := open(w[0])-0.5, open(w[1])
		us := 0.5 - math.Abs(u)
		k := math.Floor((2*a/us+b)*u + lambda + 0.43)
		if us >= 0.07 && v <= vr {
			return k
		}
		if k < 0 || us < 0.013 && v > us {
			continue
		}
		lg, _ := math.Lgamma(k + 1)
		if math.Log(v)+math.Log(invAlpha)-math.Log(a/(us*us)+b) <= -lambda+k*loglam-lg {
			return k
		}
	}
	return math.Floor(lambda)
}

// gammaGrad returns the derivative of the sample x of Gamma(alpha, 1)
// with respect to alpha at a fixed CDF P(alpha, x), the implicit
// reparameterization gradient of Figurnov et al. (2018):
//
//	dx/dα = -(∂P(α, x)/∂α) / p(x; α)
//
// ∂P/∂α is a central difference of P, or of Q = 1 - P where Q is the
// smaller, each computed to float64 precision, so the derivative is good
// to about 1e-9. For alpha above 1e5, where those series grow long, it
// differentiates the Wilson-Hilferty approximation x = α(1 - 1/(9α) +
// z/(3√α))³ at the z of x instead, which is closer than float32 there.
func gammaGrad(alpha, x float64) float64 {
	switch {
	case !(alpha > 0) || !(x >= 0) || math.IsInf(alpha, 1):
		return math.NaN()
	case x == 0 || math.IsInf(x, 1):
		return 0
	}
	if alpha > 1e5 {
		s := 1 / (9 * alpha)
		z := (math.Cbrt(x/alpha) - 1 + s) / math.Sqrt(s)
		r := 1 - s + z*math.Sqrt(s)
		// d/dα of α·r³ with r = 1 - 1/(9α) + z/(3√α).
		return r*r*r + 3*alpha*r*r*(s/alpha-z*math.Sqrt(s)/(2*alpha))
	}
	h := 1e-5 * alpha
	lower := x < alpha+1
	f := func(a float64) float64 {
		if lower {
			return gammaP(a, x)
		}
		return -gammaQ(a, x)
	}
	dP := (f(alpha+h) - f(alpha-h)) / (2 * h)
	lg, _ := math.Lgamma(alpha)
	logp := (alpha-1)*math.Log(x) - x - lg
	return -dP * math.Exp(-logp)
}

// gammaP returns the regularized lower incomplete gamma function P(a, x)
// for x < a + 1 by its series, to float64 precision.
func gammaP(a, x float64) float64 {
	sum, term := 1/a, 1/a
	for n := 1; n < 1<<20; n++ {
		term *= x / (a + float64(n))
		sum += term
		if term < sum*1e-17 {
			break
		}
	}
	lg, _ := math.Lgamma(a)
	return sum * math.Exp(a*math.Log(x)-x-lg)
}

// gammaQ returns the regularized upper incomplete gamma function Q(a, x)
// for x ≥ a + 1 by its continued fraction, evaluated by Lentz's method,
// to float64 precision.
func gammaQ(a, x float64) float64 {
	const tiny = 1e-300
	b := x + 1 - a
	c, d := 1/tiny, 1/b
	h := d
	for i := 1; i < 1<<20; i++ {
		an := -float64(i) * (float64(i) - a)
		b += 2
		d = an*d + b
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = b + an/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		del := d * c
		h *= del
		if math.Abs(del-1) < 1e-16 {
			break
		}
	}
	lg, _ := math.Lgamma(a)
	return h * math.Exp(a*math.Log(x)-x-lg)
}
