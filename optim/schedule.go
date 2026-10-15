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
