package forecast

import (
	"fmt"
	"math"

	"github.com/gsoultan/mimir/tensor"
)

// Smoothing forecasts one series by Holt-Winters exponential smoothing:
// a level, a damped trend and an additive season, each updated a little
// by every new value, as statsmodels' ExponentialSmoothing and R's ets
// do. It needs no other series and a few dozen steps, so it suits a
// short history that boosted trees cannot learn from, and Fit's Error
// and Naive compare the two on the same held out steps.
type Smoothing struct {
	Season int
	// Alpha, Beta and Gamma are how much of each new value the level,
	// the trend and the season take, 0 to 1, and Phi how much of the
	// trend carries on each step, so a forecast flattens over time.
	Alpha, Beta, Gamma, Phi float64
	// Level and Trend are the state after the last step, and Seasonal
	// the season's effects, Seasonal[t%Season] at step t, in units of
	// Scale.
	Level, Trend float64
	Seasonal     []float64
	Scale        float32
	Steps        int
	// Error is the mean absolute error of one-step forecasts of the held
	// out steps, scaled, and Naive that of repeating the value a season
	// back, or the last value without a season.
	Error, Naive float64
}

// Smooth fits exponential smoothing to one series of at least 3 steps,
// 2 seasons when there is a season, picking the smoothing by the error
// of its one-step forecasts. A value of NaN is missing.
func Smooth(series []float32, opts ...Option) (*Smoothing, error) {
	c := &config{}
	for _, o := range opts {
		o(c)
	}
	n := len(series)
	if c.season < 0 || c.holdout < 0 || c.season == 1 {
		return nil, fmt.Errorf("forecast: season %d, holdout %d: %w", c.season, c.holdout, tensor.ErrShape)
	}
	if n < 3 || c.season > 0 && n < 2*c.season {
		return nil, fmt.Errorf("forecast: %d steps for a season of %d: %w", n, c.season, tensor.ErrShape)
	}
	k := scale(series)
	z := make([]float64, n)
	present := 0
	for i, v := range series {
		z[i] = float64(v / k)
		if v == v {
			present++
		}
	}
	if present < 2 {
		return nil, fmt.Errorf("forecast: %d steps present: %w", present, tensor.ErrShape)
	}
	hold := c.holdout
	if hold == 0 {
		hold = max(1, min(n/10, max(c.season, 14)))
	}
	hold = min(hold, n-2)
	m := &Smoothing{Season: c.season, Scale: k, Steps: n}
	fit := z[:n-hold]
	if c.season > 0 && len(fit) < 2*c.season {
		fit = z // too short to hold any out
	}
	// Coordinate search over the smoothing, coarse then fine, on the
	// one-step error of the steps fitted on.
	p := [4]float64{0.3, 0.05, 0.1, 0.98}
	lo := [4]float64{0.01, 0, 0, 0.8}
	hi := [4]float64{0.99, 0.5, 0.99, 1}
	if c.season == 0 {
		hi[2] = 0
		p[2] = 0
	}
	best := math.Inf(1)
	for step := 0.1; step > 1e-3; step /= 3 {
		for round := range 3 {
			improved := false
			for i := range p {
				if hi[i] == lo[i] {
					continue
				}
				for _, cand := range []float64{p[i] - step, p[i] + step} {
					cand = min(max(cand, lo[i]), hi[i])
					q := p
					q[i] = cand
					if e := m.sse(fit, q); e < best {
						best, p, improved = e, q, true
					}
				}
			}
			if !improved && round > 0 {
				break
			}
		}
	}
	m.Alpha, m.Beta, m.Gamma, m.Phi = p[0], p[1], p[2], p[3]
	// Run the state over every step, scoring the held out ones.
	m.init(z)
	var naive, naiveN, errN float64
	for t, v := range z {
		if t >= n-hold && v == v {
			m.Error += math.Abs(m.next(t) - v)
			errN++
			back := 1
			if c.season > 0 && t >= c.season {
				back = c.season
			}
			if t-back >= 0 && z[t-back] == z[t-back] {
				naive += math.Abs(z[t-back] - v)
				naiveN++
			}
		}
		m.update(t, v)
	}
	m.Error /= max(errN, 1)
	if naiveN > 0 {
		m.Naive = naive / naiveN
	}
	return m, nil
}

// init sets the state from the first values of z: the level and trend
// from the first season or two, the season from the first season's
// departures from the level.
func (m *Smoothing) init(z []float64) {
	s := m.Season
	mean := func(v []float64) (float64, bool) {
		var sum float64
		n := 0
		for _, x := range v {
			if x == x {
				sum += x
				n++
			}
		}
		return sum / float64(max(n, 1)), n > 0
	}
	m.Seasonal = nil
	if s == 0 {
		m.Level, _ = mean(z[:1])
		m.Trend = 0
		return
	}
	first, _ := mean(z[:s])
	second, ok := mean(z[s:min(2*s, len(z))])
	m.Level = first
	m.Trend = 0
	if ok {
		m.Trend = (second - first) / float64(s)
	}
	m.Seasonal = make([]float64, s)
	for i := range s {
		if v := z[i]; v == v {
			m.Seasonal[i] = v - first
		}
	}
}

// next returns the forecast of the value at step t from the state after
// step t-1.
func (m *Smoothing) next(t int) float64 {
	v := m.Level + m.Phi*m.Trend
	if m.Season > 0 {
		v += m.Seasonal[t%m.Season]
	}
	return v
}

// update takes step t's value v into the state, or only carries the
// state forward when v is missing.
func (m *Smoothing) update(t int, v float64) {
	level, trend := m.Level, m.Trend
	if v != v {
		m.Level = level + m.Phi*trend
		m.Trend = m.Phi * trend
		return
	}
	seasonal := 0.0
	if m.Season > 0 {
		seasonal = m.Seasonal[t%m.Season]
	}
	m.Level = m.Alpha*(v-seasonal) + (1-m.Alpha)*(level+m.Phi*trend)
	m.Trend = m.Beta*(m.Level-level) + (1-m.Beta)*m.Phi*trend
	if m.Season > 0 {
		m.Seasonal[t%m.Season] = m.Gamma*(v-m.Level) + (1-m.Gamma)*seasonal
	}
}

// sse returns the sum of squared one-step errors over z with the
// smoothing p, after the first season or step, which the state is set
// from.
func (m *Smoothing) sse(z []float64, p [4]float64) float64 {
	m.Alpha, m.Beta, m.Gamma, m.Phi = p[0], p[1], p[2], p[3]
	m.init(z)
	skip := max(m.Season, 1)
	var sse float64
	for t, v := range z {
		if t >= skip && v == v {
			e := m.next(t) - v
			sse += e * e
		}
		m.update(t, v)
	}
	return sse
}

// Forecast returns the next steps values after the series.
func (m *Smoothing) Forecast(steps int) []float32 {
	out := make([]float32, steps)
	damp := 0.0
	for h := 1; h <= steps; h++ {
		damp += math.Pow(m.Phi, float64(h))
		v := m.Level + damp*m.Trend
		if m.Season > 0 {
			v += m.Seasonal[(m.Steps+h-1)%m.Season]
		}
		out[h-1] = float32(v) * m.Scale
	}
	return out
}
