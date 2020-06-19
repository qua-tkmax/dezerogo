package functions

import (
	"dezerogo/pkg/domain/model"
	"errors"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Broadcast(variable *model.Variable, row, col int) *model.Variable {
	return model.CreateFunction(forwardBroadcast, backwardBroadcast, toStringBroadcast, row, col).ForwardOne(variable)
}

func forwardBroadcast(values []*model.Variable, args []interface{}) []*model.Variable {
	input := values[0].Data
	inputRow, inputCol := input.Dims()
	row := args[0].(int)
	col := args[1].(int)
	result := mat.NewDense(row, col, nil)
	switch {
	case inputRow == 1 && inputCol == 1:
		result.Apply(func(i, j int, v float64) float64 {
			return input.At(0, 0)
		}, result)
	case inputRow == row && inputCol == 1:
		result.Apply(func(i, j int, v float64) float64 {
			return input.At(i, 0)
		}, result)
	case inputRow == 1 && inputCol == col:
		result.Apply(func(i, j int, v float64) float64 {
			return input.At(0, j)
		}, result)
	default:
		panic("invalid input")
	}
	return []*model.Variable{{Data: result}}
}

func backwardBroadcast(values []*model.Variable, gradYs []*model.Variable, _ []interface{}) ([]*model.Variable, error) {
	row, col := values[0].Data.Dims()
	axis := 0
	switch {
	case row == 1 && col == 1:
		axis = 0
	case row > 1 && col == 1:
		axis = 1
	case row == 1 && col > 1:
		axis = 2
	default:
		return nil, errors.New("invalid input")
	}
	return []*model.Variable{Sum(gradYs[0], axis)}, nil
}

func toStringBroadcast(values []*model.Variable, args []interface{}) string {
	row := args[0].(int)
	col := args[1].(int)
	return fmt.Sprintf("Reshape(%s -> [%d, %d])", values[0].ToString(), row, col)
}
