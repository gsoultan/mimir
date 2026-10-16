package distributions_test

import (
	"strings"
	"testing"

	"github.com/gsoultan/mimir/distributions"
	"github.com/gsoultan/mimir/internal/goldentest"
	"github.com/gsoultan/mimir/tensor"
)

// tol bounds the differences of float32 results and gradients from the
// float64 oracle's. Log-gamma terms of a few units cancel in some
// results, which leaves an absolute error of a few float32 ulps of them.
var tol = goldentest.Tol{Atol: 3e-5, Rtol: 2e-4}

type fn = func(in ...*tensor.Tensor) *tensor.Tensor

// goldenCases mirrors distribution_cases in tools/golden/gen.py, but for
// the special_ cases, which package tensor tests.
var goldenCases = map[string]fn{
	"normal_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewNormal(in[0], in[1]).LogProb(in[2]) },
	"normal_entropy":  func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewNormal(in[0], in[1]).Entropy() },
	"normal_cdf":      func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewNormal(in[0], in[1]).CDF(in[2]) },
	"normal_icdf":     func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewNormal(in[0], in[1]).ICDF(in[2]) },
	"normal_kl": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.KL(distributions.NewNormal(in[0], in[1]), distributions.NewNormal(in[2], in[3]))
	},

	"mvn_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.NewMultivariateNormal(in[0], in[1]).LogProb(in[2])
	},
	"mvn_entropy": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.NewMultivariateNormal(in[0], in[1]).Entropy()
	},
	"mvn_variance": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.NewMultivariateNormal(in[0], in[1]).Variance()
	},
	"mvn_kl": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.KL(distributions.NewMultivariateNormal(in[0], in[1]), distributions.NewMultivariateNormal(in[2], in[3]))
	},

	"bernoulli_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.NewBernoulliLogits(in[0]).LogProb(in[1])
	},
	"bernoulli_probs_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewBernoulli(in[0]).LogProb(in[1]) },
	"bernoulli_entropy":        func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewBernoulliLogits(in[0]).Entropy() },
	"bernoulli_kl": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.KL(distributions.NewBernoulli(in[0]), distributions.NewBernoulli(in[1]))
	},
	"categorical_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.NewCategoricalLogits(in[0]).LogProb(in[1])
	},
	"categorical_probs_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.NewCategorical(in[0]).LogProb(in[1])
	},
	"categorical_entropy": func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewCategoricalLogits(in[0]).Entropy() },
	"categorical_kl": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.KL(distributions.NewCategoricalLogits(in[0]), distributions.NewCategoricalLogits(in[1]))
	},

	"uniform_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.NewUniform(in[0], in[1]).LogProb(in[2])
	},
	"uniform_entropy": func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewUniform(in[0], in[1]).Entropy() },
	"uniform_kl": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.KL(distributions.NewUniform(in[0], in[1]), distributions.NewUniform(in[2], in[3]))
	},
	"exponential_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewExponential(in[0]).LogProb(in[1]) },
	"exponential_entropy":  func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewExponential(in[0]).Entropy() },
	"exponential_kl": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.KL(distributions.NewExponential(in[0]), distributions.NewExponential(in[1]))
	},
	"laplace_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.NewLaplace(in[0], in[1]).LogProb(in[2])
	},
	"laplace_entropy": func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewLaplace(in[0], in[1]).Entropy() },
	"laplace_kl": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.KL(distributions.NewLaplace(in[0], in[1]), distributions.NewLaplace(in[2], in[3]))
	},

	"gamma_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewGamma(in[0], in[1]).LogProb(in[2]) },
	"gamma_entropy":  func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewGamma(in[0], in[1]).Entropy() },
	"gamma_kl": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.KL(distributions.NewGamma(in[0], in[1]), distributions.NewGamma(in[2], in[3]))
	},
	"beta_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewBeta(in[0], in[1]).LogProb(in[2]) },
	"beta_entropy":  func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewBeta(in[0], in[1]).Entropy() },
	"beta_kl": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.KL(distributions.NewBeta(in[0], in[1]), distributions.NewBeta(in[2], in[3]))
	},
	"dirichlet_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewDirichlet(in[0]).LogProb(in[1]) },
	"dirichlet_entropy":  func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewDirichlet(in[0]).Entropy() },
	"dirichlet_variance": func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewDirichlet(in[0]).Variance() },
	"dirichlet_kl": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.KL(distributions.NewDirichlet(in[0]), distributions.NewDirichlet(in[1]))
	},

	"poisson_log_prob": func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewPoisson(in[0]).LogProb(in[1]) },
	"poisson_entropy":  func(in ...*tensor.Tensor) *tensor.Tensor { return distributions.NewPoisson(in[0]).Entropy() },
	"poisson_kl": func(in ...*tensor.Tensor) *tensor.Tensor {
		return distributions.KL(distributions.NewPoisson(in[0]), distributions.NewPoisson(in[1]))
	},
}

// TestGolden checks log densities, entropies, KL divergences and moments,
// and their gradients, against PyTorch or NumPy (see tools/golden/gen.py).
func TestGolden(t *testing.T) {
	cases, meta := goldentest.Load(t, "distributions")
	t.Logf("golden oracle: %s", meta["oracle"])
	for name := range goldenCases {
		if _, ok := cases[name]; !ok {
			t.Errorf("case %q has no golden data; regenerate with tools/golden/gen.py", name)
		}
	}
	for name, c := range cases {
		if strings.HasPrefix(name, "special_") {
			continue
		}
		t.Run(name, func(t *testing.T) {
			f, ok := goldenCases[name]
			if !ok {
				t.Fatalf("golden case %q has no Go counterpart", name)
			}
			in := goldentest.Inputs(t, c)
			grads := map[int]goldentest.Grader{}
			for i := range c.Grads {
				grads[i] = in[i]
			}
			goldentest.CheckGradients(t, c, f(in...), grads, tol)
		})
	}
}
