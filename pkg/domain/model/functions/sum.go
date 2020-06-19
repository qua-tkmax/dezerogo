package functions

import (
	"dezerogo/pkg/domain/model"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func Sum(variable *model.Variable, axis int) *model.Variable {
	return model.CreateFunction(forwardSum, backwardSum, toStringSum, axis).ForwardOne(variable)
}

func forwardSum(values []*model.Variable, args []interface{}) []*model.Variable {
	axis := args[0].(int)
	value := values[0].Data
	rows, cols := value.Dims()
	var result *mat.Dense
	switch axis {
	case 0:
		result = mat.NewDense(1, 1, []float64{mat.Sum(values[0].Data)})
	case 1:
		resultData := make([]float64, 0, rows)
		for r := 0; r < rows; r++ {
			sumResult := 0.0
			for c := 0; c < cols; c++ {
				sumResult += value.At(r, c)
			}
			resultData = append(resultData, sumResult)
		}
		result = mat.NewDense(rows, 1, resultData)
	case 2:
		resultData := make([]float64, 0, cols)
		for c := 0; c < cols; c++ {
			sumResult := 0.0
			for r := 0; r < rows; r++ {
				sumResult += value.At(r, c)
			}
			resultData = append(resultData, sumResult)
		}
		result = mat.NewDense(1, cols, resultData)
	}
	return []*model.Variable{{Data: result}}
}

func backwardSum(values []*model.Variable, gradYs []*model.Variable, args []interface{}) ([]*model.Variable, error) {
	value := values[0].Data
	rows, cols := value.Dims()
	return []*model.Variable{Broadcast(gradYs[0], rows, cols)}, nil
}

func toStringSum(values []*model.Variable, _ []interface{}) string {
	stringValues := make([]string, 0, len(values))
	for _, value := range values {
		stringValues = append(stringValues, value.ToString())
	}
	return strings.Join(stringValues, " + ")
}
