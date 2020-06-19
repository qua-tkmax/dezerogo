package functions

import (
	"dezerogo/pkg/domain/model"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Reshape(variable *model.Variable, row, col int) *model.Variable {
	return model.CreateFunction(forwardReshape, backwardReshape, toStringReshape, row, col).ForwardOne(variable)
}

func forwardReshape(values []*model.Variable, args []interface{}) []*model.Variable {
	row := args[0].(int)
	col := args[1].(int)
	result := mat.NewDense(row, col, values[0].Data.RawMatrix().Data) //TODO: not safe
	return []*model.Variable{{Data: result}}
}

func backwardReshape(values []*model.Variable, gradYs []*model.Variable, _ []interface{}) ([]*model.Variable, error) {
	row, col := values[0].Data.Dims()
	return []*model.Variable{Reshape(gradYs[0], row, col)}, nil
}

func toStringReshape(values []*model.Variable, args []interface{}) string {
	row := args[0].(int)
	col := args[1].(int)
	return fmt.Sprintf("Reshape(%s -> [%d, %d])", values[0].ToString(), row, col)
}
