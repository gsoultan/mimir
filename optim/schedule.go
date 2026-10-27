package optim

import "math"

// Schedule returns the learning rate for an epoch, counted from 0. Set it
// at the start of each epoch with SetLR, or pass it to train.Schedule.
// Schedules are plain functions, so they compose:
//
//	warm, cos := optim.Warmup(1e-3, 0.1, 5), optim.Cosine(1e-3, 1e-5, 45)
//	s := func(e int) float64 {
//		if e < 5 {
//			return warm(e)
//		}
//		return cos(e - 5)
//	}
type Schedule func(epoch int) float64

// StepDecay multiplies lr by gamma every n epochs, like
// torch.optim.lr_scheduler.StepLR.
func StepDecay(lr float64, n int, gamma float64) Schedule {
	n = max(n, 1)
	return func(epoch int) float64 { return lr * math.Pow(gamma, float64(epoch/n)) }
}

// Cosine anneals from lr to floor over epochs along half a cosine, like
// torch.optim.lr_scheduler.CosineAnnealingLR, and stays at floor after.
func Cosine(lr, floor float64, epochs int) Schedule {
	epochs = max(epochs, 1)
	return func(epoch int) float64 {
		e := float64(min(epoch, epochs))
		return floor + (lr-floor)*(1+math.Cos(math.Pi*e/float64(epochs)))/2
	}
}

// Warmup rises linearly from start*lr to lr over epochs and stays there,
// like torch.optim.lr_scheduler.LinearLR(start, 1, epochs).
func Warmup(lr, start float64, epochs int) Schedule {
	epochs = max(epochs, 1)
	return func(epoch int) float64 {
		f := float64(min(epoch, epochs)) / float64(epochs)
		return lr * (start + (1-start)*f)
	}
}

// Exponential multiplies lr by gamma every epoch, like
// torch.optim.lr_scheduler.ExponentialLR.
func Exponential(lr, gamma float64) Schedule {
	return func(epoch int) float64 { return lr * math.Pow(gamma, float64(epoch)) }
}

// Polynomial falls from lr to 0 over epochs as (1 - epoch/epochs)^power,
// like torch.optim.lr_scheduler.PolynomialLR, and stays at 0 after; power
// 1 is a straight line.
func Polynomial(lr float64, epochs int, power float64) Schedule {
	epochs = max(epochs, 1)
	return func(epoch int) float64 {
		f := float64(min(epoch, epochs)) / float64(epochs)
		return lr * math.Pow(1-f, power)
	}
}

// OneCycle is Smith's one-cycle policy over epochs (or steps, set at each
// one), like torch.optim.lr_scheduler.OneCycleLR with cosine annealing:
// from peak/25 up to peak over the first warm share of them, 0.3 being
// the usual, then down to peak/25e4 by the last. It trains many networks
// to their best in fewer epochs than a fixed rate.
func OneCycle(peak float64, epochs int, warm float64) Schedule {
	total := max(epochs, 2)
	start, end := peak/25, peak/25/1e4
	up := max(warm*float64(total)-1, 1)
	cosine := func(from, to, f float64) float64 { return to + (from-to)*(1+math.Cos(math.Pi*f))/2 }
	return func(epoch int) float64 {
		e := float64(min(max(epoch, 0), total-1))
		if e <= up {
			return cosine(start, peak, e/up)
		}
		return cosine(peak, end, (e-up)/(float64(total-1)-up))
	}
}
