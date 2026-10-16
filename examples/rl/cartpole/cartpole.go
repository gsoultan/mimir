// Package cartpole simulates the cart-pole balancing task of Barto,
// Sutton and Anderson, with the equations, constants and limits of
// Gymnasium's CartPole-v1, so that reinforcement learning examples need
// no external simulator.
//
// A pole is hinged to a cart that moves along a track. Each step the
// agent pushes the cart left or right with a fixed force and receives a
// reward of 1. An episode terminates when the pole leans more than 12
// degrees from upright or the cart leaves the track, 2.4 units from the
// centre, and is truncated after MaxSteps steps. The task counts as
// solved when the mean return over 100 consecutive episodes reaches 195.
package cartpole

import (
	"math"
	"math/rand/v2"
)

// Physical constants of CartPole-v1, in SI units.
const (
	Gravity    = 9.8
	MassCart   = 1.0
	MassPole   = 0.1
	HalfPole   = 0.5  // half the pole's length
	ForceMag   = 10.0 // the force of a push
	Tau        = 0.02 // seconds per step
	totalMass  = MassCart + MassPole
	poleMoment = MassPole * HalfPole
)

// Limits past which an episode terminates.
const (
	XLimit     = 2.4
	ThetaLimit = 12 * 2 * math.Pi / 360
)

// Actions: push the cart left or right.
const (
	Left  = 0
	Right = 1
)

// Observation is the state the agent sees: the cart's position and
// velocity, and the pole's angle from upright, in radians, and angular
// velocity.
type Observation [4]float32

// Env is one cart-pole simulation. It is not safe for concurrent use;
// run one Env per goroutine.
type Env struct {
	// MaxSteps truncates an episode after this many steps; 500 unless set.
	MaxSteps int

	x, xDot, theta, thetaDot float64
	steps                    int
	over                     bool
	rng                      *rand.Rand
}

// New returns an environment whose episodes start from states drawn
// from seed, the same states for the same seed. Call Reset to start.
func New(seed uint64) *Env {
	return &Env{MaxSteps: 500, rng: rand.New(rand.NewPCG(seed, 0x5DEECE66D)), over: true}
}

// Reset starts an episode from a state each of whose components is drawn
// uniformly from [-0.05, 0.05], and returns its observation.
func (e *Env) Reset() Observation {
	u := func() float64 { return e.rng.Float64()*0.1 - 0.05 }
	e.x, e.xDot, e.theta, e.thetaDot = u(), u(), u(), u()
	e.steps, e.over = 0, false
	return e.observe()
}

func (e *Env) observe() Observation {
	return Observation{float32(e.x), float32(e.xDot), float32(e.theta), float32(e.thetaDot)}
}

// Step pushes the cart by action, Left or Right, for one step of Tau
// seconds by explicit Euler integration, and returns the observation,
// the reward of 1, whether the episode terminated because the pole fell
// or the cart left the track, and whether it was truncated at MaxSteps.
// Stepping an episode that is over, or one never Reset, starts a new
// one first. Any action other than Left pushes right.
func (e *Env) Step(action int) (obs Observation, reward float64, terminated, truncated bool) {
	if e.over {
		e.Reset()
	}
	force := ForceMag
	if action == Left {
		force = -ForceMag
	}
	sin, cos := math.Sincos(e.theta)
	temp := (force + poleMoment*e.thetaDot*e.thetaDot*sin) / totalMass
	thetaAcc := (Gravity*sin - cos*temp) / (HalfPole * (4.0/3.0 - MassPole*cos*cos/totalMass))
	xAcc := temp - poleMoment*thetaAcc*cos/totalMass
	e.x += Tau * e.xDot
	e.xDot += Tau * xAcc
	e.theta += Tau * e.thetaDot
	e.thetaDot += Tau * thetaAcc
	e.steps++
	terminated = e.x < -XLimit || e.x > XLimit || e.theta < -ThetaLimit || e.theta > ThetaLimit
	truncated = !terminated && e.MaxSteps > 0 && e.steps >= e.MaxSteps
	e.over = terminated || truncated
	return e.observe(), 1, terminated, truncated
}

// Steps returns the number of steps of the current episode.
func (e *Env) Steps() int { return e.steps }
