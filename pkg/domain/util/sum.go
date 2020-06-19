package util

import "gonum.org/v1/gonum/mat"

func SumDenses(dense *mat.Dense, targetRows, targetColumns int) *mat.Dense {
	rows, cols := dense.Dims()
	var result *mat.Dense
	switch {
	case targetRows == 1 && targetColumns == 1:
		result = mat.NewDense(1, 1, []float64{mat.Sum(dense)})
	case targetColumns == 1:
		resultData := make([]float64, 0, rows)
		for r := 0; r < rows; r++ {
			sumResult := 0.0
			for c := 0; c < cols; c++ {
				sumResult += dense.At(r, c)
			}
			resultData = append(resultData, sumResult)
		}
		result = mat.NewDense(rows, 1, resultData)
	case targetRows == 1:
		resultData := make([]float64, 0, cols)
		for c := 0; c < cols; c++ {
			sumResult := 0.0
			for r := 0; r < rows; r++ {
				sumResult += dense.At(r, c)
			}
			resultData = append(resultData, sumResult)
		}
		result = mat.NewDense(1, cols, resultData)
	}
	return result
}
