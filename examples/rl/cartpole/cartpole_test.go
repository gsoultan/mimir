package cartpole

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestStepFollowsTheEquations(t *testing.T) {
	e := New(1)
	e.Reset()
	e.x, e.xDot, e.theta, e.thetaDot = 0.1, -0.2, 0.05, 0.3
	obs, r, term, trunc := e.Step(Right)
	// CartPole-v1's update, written out for this state and a push right.
	sin, cos := math.Sin(0.05), math.Cos(0.05)
	temp := (10 + 0.05*0.3*0.3*sin) / 1.1
	thetaAcc := (9.8*sin - cos*temp) / (0.5 * (4.0/3 - 0.1*cos*cos/1.1))
	xAcc := temp - 0.05*thetaAcc*cos/1.1
	want := Observation{float32(0.1 + 0.02*-0.2), float32(-0.2 + 0.02*xAcc), float32(0.05 + 0.02*0.3), float32(0.3 + 0.02*thetaAcc)}
	if obs != want || r != 1 || term || trunc {
		t.Errorf("step gave %v, %v, %v, %v; want %v, 1, false, false", obs, r, term, trunc, want)
	}
}

func TestResetIsSeededAndSmall(t *testing.T) {
	a, b, c := New(7), New(7), New(8)
	for range 3 {
		oa, ob, oc := a.Reset(), b.Reset(), c.Reset()
		if oa != ob || oa == oc {
			t.Fatalf("resets %v, %v and %v", oa, ob, oc)
		}
		for _, v := range oa {
			if v < -0.05 || v > 0.05 {
				t.Fatalf("start %v outside [-0.05, 0.05]", oa)
			}
		}
	}
}

// episode runs one episode under policy and returns its return and how
// it ended.
func episode(e *Env, policy func(Observation) int) (ret float64, terminated bool) {
	obs := e.Reset()
	for {
		var r float64
		var trunc bool
		obs, r, terminated, trunc = e.Step(policy(obs))
		ret += r
		if terminated || trunc {
			return ret, terminated
		}
	}
}

func TestPolesFallUnlessBalanced(t *testing.T) {
	e := New(3)
	// Pushing one way topples the pole within a few dozen steps.
	if ret, term := episode(e, func(Observation) int { return Right }); !term || ret > 40 {
		t.Errorf("pushing right: return %v, terminated %v", ret, term)
	}
	// Random pushes last about 22 steps on average.
	rng := rand.New(rand.NewPCG(1, 2))
	var sum float64
	for range 200 {
		ret, _ := episode(e, func(Observation) int { return rng.IntN(2) })
		sum += ret
	}
	if mean := sum / 200; mean < 15 || mean > 30 {
		t.Errorf("random policy: mean return %v", mean)
	}
	// Pushing the cart under the pole as it leans keeps it up until the
	// episode is truncated.
	balance := func(o Observation) int {
		if o[2]+0.5*o[3]+0.01*o[0]+0.1*o[1] > 0 {
			return Right
		}
		return Left
	}
	if ret, term := episode(e, balance); term || ret != 500 {
		t.Errorf("balancing: return %v, terminated %v", ret, term)
	}
	e.MaxSteps = 50
	if ret, term := episode(e, balance); term || ret != 50 {
		t.Errorf("balancing with 50 steps: return %v, terminated %v", ret, term)
	}
}

func TestStepAfterTheEndStartsAgain(t *testing.T) {
	e := New(4)
	e.Step(Left) // never Reset
	if e.Steps() != 1 {
		t.Fatalf("%d steps", e.Steps())
	}
	for {
		if _, _, term, _ := e.Step(Left); term {
			break
		}
	}
	e.Step(Left)
	if e.Steps() != 1 {
		t.Errorf("a step after the end is step %d of the next episode", e.Steps())
	}
}
