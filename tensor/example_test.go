package tensor_test

import (
	"context"
	"errors"
	"fmt"

	"github.com/gsoultan/mimir/tensor"
)

// A two-layer perceptron's forward pass: nothing runs until Data.
func Example() {
	ctx := context.Background()
	x := tensor.Randn(1, 4, 8)   // batch of 4, 8 features
	w1 := tensor.Randn(2, 8, 16) // seeds make every run identical
	w2 := tensor.Randn(3, 16, 3)
	b2 := tensor.Zeros(tensor.Float32, 3)

	probs := x.MatMul(w1).GELU().MatMul(w2).Add(b2).Softmax(-1)
	fmt.Println(probs.Shape())

	sums, err := probs.Sum(-1).Data[float32](ctx)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, s := range sums {
		fmt.Printf("%.4f ", s)
	}
	fmt.Println()
	// Output:
	// [4 3]
	// 1.0000 1.0000 1.0000 1.0000
}

// Shape errors are reported when the operation is built, without panics.
func ExampleTensor_Err() {
	a := tensor.Ones(tensor.Float32, 2, 3)
	b := tensor.Ones(tensor.Float32, 4, 5)
	c := a.MatMul(b).ReLU() // carries the matmul's error
	fmt.Println(errors.Is(c.Err(), tensor.ErrShape))
	// Output: true
}

func ExampleTensor_Gather() {
	scores := tensor.FromSlice([]float32{0.1, 0.7, 0.2, 0.5, 0.3, 0.2}, 2, 3)
	labels := tensor.FromSlice([]int32{1, 0}, 2, 1)
	picked, _ := scores.Gather(1, labels).Data[float32](context.Background())
	fmt.Println(picked)
	// Output: [0.7 0.5]
}
