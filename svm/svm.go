// Package svm fits support vector machines to tables of numbers, like
// scikit-learn's SVC, SVR and LinearSVC: a maximum-margin classifier or
// an ε-insensitive regression, with a linear, RBF, polynomial or sigmoid
// kernel.
//
//	m, err := svm.Fit(ctx, d, tabular.Classification, svm.Kernel(svm.RBF), svm.C(10))
//	if err != nil { ... }
//	p := m.Predict(rows) // a probability for each class of each row
//
// Kernel machines are solved by sequential minimal optimization with
// libsvm's second-order working-set selection; a linear classifier by
// LIBLINEAR's dual coordinate descent, which needs no kernel matrix and
// scales to millions of rows. More than two classes are one against the
// rest. Probabilities come from Platt scaling of each decision value.
//
// Each feature is standardized first, so C and the kernel's gamma mean
// the same whatever the features' units; a missing value, NaN, counts as
// the feature's mean. A Model holds only exported fields, so
// encoding/json saves and loads it; Module serves it as an nn.Module.
package svm

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"

	"github.com/gsoultan/mimir/tabular"
	"github.com/gsoultan/mimir/tensor"
)

// The kernels, by the names Model.Kernel holds.
const (
	Linear     = "linear"  // x·z
	RBF        = "rbf"     // exp(-gamma·|x-z|²)
	Polynomial = "poly"    // (gamma·x·z + coef0)^degree
	Sigmoid    = "sigmoid" // tanh(gamma·x·z + coef0)
)

// Model is a fitted support vector machine. Its decision value for
// output o is Σ_s Coef[o,s]·K(x, Support[s]) + Bias[o], x standardized.
type Model struct {
	Task     tabular.Task
	Features int
	// Classes is the number of classes of a Classification model, and 0
	// for Regression.
	Classes int
	// Outputs is 1 for Regression and two classes, Classes otherwise.
	Outputs int
	Kernel  string
	Gamma   float64
	Coef0   float64
	Degree  int
	// Mean and Std standardize each feature.
	Mean, Std []float32
	// Support holds the support vectors, standardized, Features numbers
	// each. A linear model keeps one weight vector per output instead.
	Support []float32
	// Coef holds a weight per support vector for each output, output by
	// output, and Bias each output's offset.
	Coef []float32
	Bias []float32
	// A and B turn output o's decision value f into a probability,
	// sigmoid(A[o]·f + B[o]), for Classification.
	A, B []float32
	// YMean and YStd undo the standardization of a Regression's answers.
	YMean, YStd float32
	// Iterations is how many steps the longest solve took.
	Iterations int
}

// SupportVectors returns the number of support vectors.
func (m *Model) SupportVectors() int { return len(m.Support) / max(m.Features, 1) }

// Option configures Fit.
type Option func(*config)

type config struct {
	c, gamma, coef0, eps, tol float64
	degree, iters, workers    int
	classes                   int
	kernel                    string
	cacheMB                   int
}

// C sets the penalty for each point inside the margin: larger fits the
// rows more closely. The default is 1.
func C(c float64) Option { return func(o *config) { o.c = c } }

// Kernel sets the kernel: Linear, RBF (the default), Polynomial or
// Sigmoid.
func Kernel(k string) Option { return func(o *config) { o.kernel = k } }

// Gamma sets the kernel's scale. The default is 1/Features, which
// scikit-learn calls "scale" for standardized features.
func Gamma(g float64) Option { return func(o *config) { o.gamma = g } }

// Coef0 sets the constant of the polynomial and sigmoid kernels. The
// default is 0.
func Coef0(c float64) Option { return func(o *config) { o.coef0 = c } }

// Degree sets the polynomial kernel's degree. The default is 3.
func Degree(d int) Option { return func(o *config) { o.degree = d } }

// Epsilon sets the width of the tube within which a Regression's errors
// cost nothing, in standard deviations of the answers. The default is
// 0.1.
func Epsilon(e float64) Option { return func(o *config) { o.eps = e } }

// Tol sets the stopping tolerance on the optimality gap. The default is
// 1e-3, libsvm's.
func Tol(t float64) Option { return func(o *config) { o.tol = t } }

// MaxIter caps the solver's steps for each output. The default is
// max(10⁷, 100·rows).
func MaxIter(n int) Option { return func(o *config) { o.iters = n } }

// Classes sets the number of classes, for when the last ones are missing
// from the training rows. The default counts them in d.Y.
func Classes(k int) Option { return func(o *config) { o.classes = k } }

// Workers sets the goroutines that fit outputs and compute kernel rows.
// The default is GOMAXPROCS.
func Workers(n int) Option { return func(o *config) { o.workers = n } }

// CacheMB sets the memory for kernel rows kept between steps. The
// default is 256.
func CacheMB(mb int) Option { return func(o *config) { o.cacheMB = mb } }

// Fit fits a support vector machine of d for task.
func Fit(ctx context.Context, d tabular.Data, task tabular.Task, opts ...Option) (*Model, error) {
	c := config{c: 1, kernel: RBF, degree: 3, eps: 0.1, tol: 1e-3, cacheMB: 256}
	for _, o := range opts {
		o(&c)
	}
	if err := d.Check(task); err != nil {
		return nil, err
	}
	switch {
	case !(c.c > 0) || math.IsInf(c.c, 0):
		return nil, fmt.Errorf("svm: C %v: %w", c.c, tensor.ErrShape)
	case !slices.Contains([]string{Linear, RBF, Polynomial, Sigmoid}, c.kernel):
		return nil, fmt.Errorf("svm: kernel %q: %w", c.kernel, tensor.ErrShape)
	case c.gamma < 0 || math.IsNaN(c.gamma) || c.eps < 0 || !(c.tol > 0) || c.kernel == Polynomial && c.degree < 1:
		return nil, fmt.Errorf("svm: gamma %v, epsilon %v, tol %v, degree %d: %w", c.gamma, c.eps, c.tol, c.degree, tensor.ErrShape)
	}
	if c.gamma == 0 {
		c.gamma = 1 / float64(d.Features)
	}
	if c.workers <= 0 {
		c.workers = runtime.GOMAXPROCS(0)
	}
	n := d.Rows()
	if c.iters <= 0 {
		c.iters = max(10_000_000, 100*n)
	}
	m := &Model{Task: task, Features: d.Features, Kernel: c.kernel, Gamma: c.gamma, Coef0: c.coef0, Degree: c.degree, YStd: 1}
	x := m.standardize(d)
	if task == tabular.Regression {
		return m, m.fitRegression(ctx, x, d.Y, c)
	}
	m.Classes = d.Classes()
	if c.classes > 0 {
		if c.classes < m.Classes || c.classes > tabular.MaxClasses {
			return nil, fmt.Errorf("svm: %d classes for answers up to %d: %w", c.classes, m.Classes-1, tensor.ErrShape)
		}
		m.Classes = c.classes
	}
	return m, m.fitClasses(ctx, x, d.Y, c)
}

// standardize sets m.Mean and m.Std from d and returns d.X standardized,
// a missing value 0.
func (m *Model) standardize(d tabular.Data) []float32 {
	n, f := d.Rows(), d.Features
	mean, sq, cnt := make([]float64, f), make([]float64, f), make([]float64, f)
	for i := range n {
		for j, v := range d.Row(i) {
			if v == v {
				mean[j] += float64(v)
				cnt[j]++
			}
		}
	}
	for j := range f {
		mean[j] /= max(cnt[j], 1)
	}
	for i := range n {
		for j, v := range d.Row(i) {
			if v == v {
				dv := float64(v) - mean[j]
				sq[j] += dv * dv
			}
		}
	}
	m.Mean, m.Std = make([]float32, f), make([]float32, f)
	for j := range f {
		m.Mean[j] = float32(mean[j])
		s := math.Sqrt(sq[j] / max(cnt[j], 1))
		if s == 0 || math.IsNaN(s) {
			s = 1
		}
		m.Std[j] = float32(s)
	}
	x := make([]float32, len(d.X))
	m.scale(d.X, x)
	return x
}

// scale writes the rows of src, standardized, to dst.
func (m *Model) scale(src, dst []float32) {
	f := m.Features
	for i, v := range src {
		j := i % f
		if v != v {
			dst[i] = 0
		} else {
			dst[i] = (v - m.Mean[j]) / m.Std[j]
		}
	}
}

// fitClasses fits one machine per output: the log-odds of class 1 for
// two classes, each class against the rest for more.
func (m *Model) fitClasses(ctx context.Context, x, y []float32, c config) error {
	n, f := len(y), m.Features
	m.Outputs = m.Classes
	if m.Classes == 2 {
		m.Outputs = 1
	}
	signs := func(o int) []int8 {
		s := make([]int8, n)
		class := float32(o)
		if m.Outputs == 1 {
			class = 1
		}
		for i, v := range y {
			s[i] = -1
			if v == class {
				s[i] = 1
			}
		}
		return s
	}
	type result struct {
		coef  []float64 // a weight per row, or per feature for a linear model
		bias  float64
		iters int
		err   error
	}
	res := make([]result, m.Outputs)
	var k *kernel
	if c.kernel != Linear {
		k = newKernel(m, x, n, c)
	}
	var wg sync.WaitGroup
	sem := make(chan struct{}, c.workers)
	for o := range m.Outputs {
		wg.Go(func() {
			sem <- struct{}{}
			defer func() { <-sem }()
			s := signs(o)
			if c.kernel == Linear {
				w, b, it, err := linearDual(ctx, x, f, s, c)
				res[o] = result{w, b, it, err}
				return
			}
			p := make([]float64, n)
			for i := range p {
				p[i] = -1
			}
			alpha, rho, it, err := solve(ctx, problem{n: n, y: s, p: p, c: c.c, row: func(i int) []float32 { return k.row(i) }, diag: k.diag}, c)
			for i := range alpha {
				alpha[i] *= float64(s[i])
			}
			res[o] = result{alpha, -rho, it, err}
		})
	}
	wg.Wait()
	for _, r := range res {
		if r.err != nil {
			return r.err
		}
		m.Iterations = max(m.Iterations, r.iters)
	}
	coefs := make([][]float64, m.Outputs)
	m.Bias = make([]float32, m.Outputs)
	for o, r := range res {
		coefs[o], m.Bias[o] = r.coef, float32(r.bias)
	}
	if c.kernel == Linear {
		m.keepWeights(coefs)
	} else {
		m.keepSupport(x, coefs)
	}
	// Platt scaling of each output's decisions on the training rows.
	dec := m.decisions(x, false)
	m.A, m.B = make([]float32, m.Outputs), make([]float32, m.Outputs)
	for o := range m.Outputs {
		s := signs(o)
		fo := make([]float64, n)
		for i := range n {
			fo[i] = float64(dec[i*m.Outputs+o])
		}
		a, b := platt(fo, s)
		m.A[o], m.B[o] = float32(a), float32(b)
	}
	return nil
}

// fitRegression fits ε-insensitive support vector regression of the
// standardized answers.
func (m *Model) fitRegression(ctx context.Context, x, y []float32, c config) error {
	n := len(y)
	var mean, sq float64
	for _, v := range y {
		mean += float64(v)
	}
	mean /= float64(n)
	for _, v := range y {
		sq += (float64(v) - mean) * (float64(v) - mean)
	}
	std := math.Sqrt(sq / float64(n))
	if std == 0 {
		std = 1
	}
	m.Outputs, m.YMean, m.YStd = 1, float32(mean), float32(std)
	// libsvm's formulation: 2n variables, α then α*, and Q = ±K.
	s, p := make([]int8, 2*n), make([]float64, 2*n)
	for i, v := range y {
		z := (float64(v) - mean) / std
		s[i], s[i+n] = 1, -1
		p[i], p[i+n] = c.eps-z, c.eps+z
	}
	k := newKernel(m, x, n, c)
	diag := make([]float64, 2*n)
	copy(diag, k.diag)
	copy(diag[n:], k.diag)
	alpha, rho, it, err := solve(ctx, problem{n: 2 * n, y: s, p: p, c: c.c, diag: diag, row: func(i int) []float32 { return k.row(i % n) }, wrap: n}, c)
	if err != nil {
		return err
	}
	m.Iterations = it
	coef := make([]float64, n)
	for i := range n {
		coef[i] = alpha[i] - alpha[i+n]
	}
	m.Bias = []float32{float32(-rho)}
	if c.kernel == Linear {
		w := make([]float64, m.Features)
		for i, a := range coef {
			if a != 0 {
				for j, v := range x[i*m.Features : (i+1)*m.Features] {
					w[j] += a * float64(v)
				}
			}
		}
		m.keepWeights([][]float64{w})
	} else {
		m.keepSupport(x, [][]float64{coef})
	}
	return nil
}

// keepWeights keeps a linear model's weight vectors as its support, with
// each output weighing only its own.
func (m *Model) keepWeights(ws [][]float64) {
	k := len(ws)
	m.Support = make([]float32, 0, k*m.Features)
	m.Coef = make([]float32, k*k)
	for o, w := range ws {
		for _, v := range w {
			m.Support = append(m.Support, float32(v))
		}
		m.Coef[o*k+o] = 1
	}
}

// keepSupport keeps the rows any output weighs.
func (m *Model) keepSupport(x []float32, coefs [][]float64) {
	n, f := len(coefs[0]), m.Features
	var keep []int
	for i := range n {
		for _, c := range coefs {
			if c[i] != 0 {
				keep = append(keep, i)
				break
			}
		}
	}
	m.Support = make([]float32, 0, len(keep)*f)
	m.Coef = make([]float32, len(coefs)*len(keep))
	for s, i := range keep {
		m.Support = append(m.Support, x[i*f:(i+1)*f]...)
		for o, c := range coefs {
			m.Coef[o*len(keep)+s] = float32(c[i])
		}
	}
}

// k returns the kernel of two standardized rows.
func (m *Model) k(a, b []float32) float64 {
	switch m.Kernel {
	case RBF:
		var d float64
		for j, v := range a {
			t := float64(v) - float64(b[j])
			d += t * t
		}
		return math.Exp(-m.Gamma * d)
	}
	var dot float64
	for j, v := range a {
		dot += float64(v) * float64(b[j])
	}
	switch m.Kernel {
	case Polynomial:
		return math.Pow(m.Gamma*dot+m.Coef0, float64(m.Degree))
	case Sigmoid:
		return math.Tanh(m.Gamma*dot + m.Coef0)
	}
	return dot
}

// decisions returns each standardized row's decision values, Outputs per
// row, in parallel when parallel is set.
func (m *Model) decisions(x []float32, parallel bool) []float32 {
	f, k := m.Features, m.Outputs
	sv := m.SupportVectors()
	if f <= 0 || len(x)%f != 0 {
		panic(fmt.Sprintf("svm: %d numbers for rows of %d features", len(x), f))
	}
	n := len(x) / f
	out := make([]float32, n*k)
	rows := func(lo, hi int) {
		kr := make([]float64, sv)
		for i := lo; i < hi; i++ {
			row := x[i*f : (i+1)*f]
			for s := range sv {
				kr[s] = m.k(row, m.Support[s*f:(s+1)*f])
			}
			for o := range k {
				v := float64(m.Bias[o])
				for s, c := range m.Coef[o*sv : (o+1)*sv] {
					v += float64(c) * kr[s]
				}
				out[i*k+o] = float32(v)
			}
		}
	}
	workers := 1
	if parallel {
		workers = min(runtime.GOMAXPROCS(0), max(n/64, 1))
	}
	var wg sync.WaitGroup
	for w := range workers {
		wg.Go(func() { rows(w*n/workers, (w+1)*n/workers) })
	}
	wg.Wait()
	return out
}

// Decision returns each row's decision values, Outputs per row: the
// signed distances from the margin, before Platt scaling.
func (m *Model) Decision(x []float32) []float32 {
	z := make([]float32, len(x))
	m.scale(x, z)
	return m.decisions(z, true)
}

// Predict answers each row of x, which holds rows of m.Features numbers
// one after another: for Classification m.Classes probabilities per row,
// for Regression one value per row. It panics if x is not a whole number
// of rows.
func (m *Model) Predict(x []float32) []float32 {
	dec := m.Decision(x)
	if m.Task == tabular.Regression {
		for i, v := range dec {
			dec[i] = v*m.YStd + m.YMean
		}
		return dec
	}
	n := len(dec) / m.Outputs
	out := make([]float32, n*m.Classes)
	for i := range n {
		if m.Outputs == 1 {
			p := sigmoid(float64(m.A[0])*float64(dec[i]) + float64(m.B[0]))
			out[2*i], out[2*i+1] = float32(1-p), float32(p)
			continue
		}
		var sum float64
		ps := out[i*m.Classes : (i+1)*m.Classes]
		for o := range m.Outputs {
			p := sigmoid(float64(m.A[o])*float64(dec[i*m.Outputs+o]) + float64(m.B[o]))
			ps[o] = float32(p)
			sum += p
		}
		for o := range ps {
			ps[o] = float32(float64(ps[o]) / max(sum, 1e-30))
		}
	}
	return out
}

func sigmoid(z float64) float64 { return 1 / (1 + math.Exp(-z)) }

// platt fits sigmoid(a·f + b) to the classes s of decisions f, as Platt
// did with his targets softened by the class counts, by libsvm's Newton
// method with backtracking.
func platt(f []float64, s []int8) (a, b float64) {
	var pos, neg float64
	for _, v := range s {
		if v > 0 {
			pos++
		} else {
			neg++
		}
	}
	hi, lo := (pos+1)/(pos+2), 1/(neg+2)
	t := make([]float64, len(f))
	for i, v := range s {
		t[i] = lo
		if v > 0 {
			t[i] = hi
		}
	}
	// libsvm's sigmoid_train fits P = 1/(1+exp(A·f+B)); it returns -A, -B.
	A, B := 0.0, math.Log((neg+1)/(pos+1))
	obj := func(A, B float64) float64 {
		var v float64
		for i, x := range f {
			z := x*A + B
			if z >= 0 {
				v += t[i]*z + math.Log1p(math.Exp(-z))
			} else {
				v += (t[i]-1)*z + math.Log1p(math.Exp(z))
			}
		}
		return v
	}
	fval := obj(A, B)
	const sigma, eps = 1e-12, 1e-5
	for range 100 {
		h11, h22, h21, g1, g2 := sigma, sigma, 0.0, 0.0, 0.0
		for i, x := range f {
			z := x*A + B
			var p, q float64
			if z >= 0 {
				p, q = math.Exp(-z)/(1+math.Exp(-z)), 1/(1+math.Exp(-z))
			} else {
				p, q = 1/(1+math.Exp(z)), math.Exp(z)/(1+math.Exp(z))
			}
			d2 := p * q
			h11 += x * x * d2
			h22 += d2
			h21 += x * d2
			d1 := t[i] - p
			g1 += x * d1
			g2 += d1
		}
		if math.Abs(g1) < eps && math.Abs(g2) < eps {
			break
		}
		det := h11*h22 - h21*h21
		dA, dB := -(h22*g1-h21*g2)/det, -(-h21*g1+h11*g2)/det
		gd := g1*dA + g2*dB
		step := 1.0
		for step >= 1e-10 {
			nA, nB := A+step*dA, B+step*dB
			if nf := obj(nA, nB); nf < fval+1e-4*step*gd {
				A, B, fval = nA, nB, nf
				break
			}
			step /= 2
		}
		if step < 1e-10 {
			break
		}
	}
	return -A, -B
}
