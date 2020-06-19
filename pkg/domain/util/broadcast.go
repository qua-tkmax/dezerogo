package util

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

func BroadcastDenses(dense1, dense2 *mat.Dense) (*mat.Dense, *mat.Dense, error) {
	rows1, cols1 := dense1.Dims()
	rows2, cols2 := dense2.Dims()
	switch {
	case rows1 == rows2 && cols1 == cols2:
		return dense1, dense2, nil
	case rows1 == 1 && cols1 == 1:
		newDense1 := mat.NewDense(rows2, cols2, nil)
		newDense1.Apply(func(i, j int, v float64) float64 {
			return dense1.At(0, 0)
		}, newDense1)
		return newDense1, dense2, nil
	case rows1 == 1 && cols1 == cols2:
		newDense1 := mat.NewDense(rows2, cols2, nil)
		newDense1.Apply(func(i, j int, v float64) float64 {
			return dense1.At(0, j)
		}, newDense1)
		return newDense1, dense2, nil
	case rows2 == 1 && cols1 == cols2:
		newDense2 := mat.NewDense(rows1, cols1, nil)
		newDense2.Apply(func(i, j int, v float64) float64 {
			return dense2.At(0, j)
		}, newDense2)
		return dense1, newDense2, nil
	case rows2 == 1 && cols2 == 1:
		newDense2 := mat.NewDense(rows1, cols1, nil)
		newDense2.Apply(func(i, j int, v float64) float64 {
			return dense2.At(0, 0)
		}, newDense2)
		return dense1, newDense2, nil
	case rows1 == rows2 && cols1 == 1:
		newDense1 := mat.NewDense(rows2, cols2, nil)
		newDense1.Apply(func(i, j int, v float64) float64 {
			return dense1.At(i, 0)
		}, newDense1)
		return newDense1, dense2, nil
	case rows1 == rows2 && cols2 == 1:
		newDense2 := mat.NewDense(rows1, cols1, nil)
		newDense2.Apply(func(i, j int, v float64) float64 {
			return dense2.At(i, 0)
		}, newDense2)
		return dense1, newDense2, nil
	default:
		return nil, nil, errors.New("invalid dims")
	}
}
