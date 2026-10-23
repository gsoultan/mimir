package backend

// Pooler is a Backend that runs the pooling kernels OpMaxPool and
// OpMaxPoolGrad. Package tensor pools on such a backend with one kernel
// each way instead of a reduction over the windows that OpUnfold copies
// out, or a view of them, and the gradient of that reduction, which read
// the windows several times over.
type Pooler interface {
	Backend
	// Pools reports whether the backend runs the pooling kernels.
	Pools() bool
}
