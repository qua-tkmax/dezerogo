package functions

import (
	"dezerogo/pkg/domain/model"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func MatMul(variable1, variable2 *model.Variable) *model.Variable {
	return model.CreateFunction(forwardMatMul, backwardMatMul, toStringMatMul).ForwardOne(variable1, variable2)
}

func forwardMatMul(values []*model.Variable, _ []interface{}) []*model.Variable {
	rows1, cols1 := values[0].Data.Dims()
	rows2, cols2 := values[1].Data.Dims()
	if cols1 != rows2 {
		panic("invalid dims")
	}
	result := mat.NewDense(rows1, cols2, nil)
	result.Mul(values[0].Data, values[1].Data)
	return []*model.Variable{{Data: result}}
}

func backwardMatMul(values []*model.Variable, gradYs []*model.Variable, _ []interface{}) ([]*model.Variable, error) {
	value1 := MatMul(gradYs[0], Transpose(values[1]))
	value2 := MatMul(Transpose(values[0]), gradYs[0])
	return []*model.Variable{value1, value2}, nil
}

func toStringMatMul(values []*model.Variable, _ []interface{}) string {
	return fmt.Sprintf("%s * %s", values[0].ToString(), values[1].ToString())
}
