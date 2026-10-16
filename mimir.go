// Package mimir is a machine learning framework in pure Go that trains and
// runs models on the CPU or a GPU from one binary.
//
// This package picks devices. Tensors, layers, optimizers and the training
// loop live in the tensor, nn, optim and train packages:
//
//	dev := mimir.Auto() // the GPU if there is one, otherwise the CPU
//	model := nn.NewSequential(nn.NewLinear(784, 10))
//	if err := nn.To(model, dev); err != nil { ... }
//	err := train.Fit(ctx, model, opt, trainSet, train.Loss(loss.CrossEntropy))
//
// On Apple silicon Macs the GPU runs through Metal, and on Linux machines
// with an NVIDIA GPU through CUDA. Set MIMIR_DEVICE to cpu, gpu or auto to
// choose the default device for tensors that FromSlice creates.
package mimir

import "github.com/gsoultan/mimir/tensor"

// Device is where tensors live and are computed; see tensor.Device.
type Device = tensor.Device

// Auto returns the GPU if there is one mimir can use, and otherwise the
// CPU.
func Auto() Device { return tensor.Auto() }

// Devices returns every device mimir can use: the CPU, then the GPU if
// there is one.
func Devices() []Device { return tensor.Devices() }

// CPU returns the host CPU.
func CPU() Device { return tensor.CPU() }

// GPU returns the GPU, or an error saying why there is none.
func GPU() (Device, error) { return tensor.GPU() }
