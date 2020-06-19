package functions

import (
	"dezerogo/pkg/domain/model"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Transpose(variable *model.Variable) *model.Variable {
	return model.CreateFunction(forwardTranspose, backwardTranspose, toStringTranspose).ForwardOne(variable)
}

func forwardTranspose(values []*model.Variable, _ []interface{}) []*model.Variable {
	result := mat.DenseCopyOf(values[0].Data.T())
	return []*model.Variable{{Data: result}}
}

func backwardTranspose(values []*model.Variable, gradYs []*model.Variable, _ []interface{}) ([]*model.Variable, error) {
	return []*model.Variable{Transpose(gradYs[0])}, nil
}

func toStringTranspose(values []*model.Variable, _ []interface{}) string {
	return fmt.Sprintf("Transpose(%s)", values[0].ToString())
}
