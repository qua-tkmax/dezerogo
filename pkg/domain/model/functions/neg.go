package functions

import (
	"dezerogo/pkg/domain/model"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Neg(variable *model.Variable) *model.Variable {
	return model.CreateFunction(forwardNeg, backwardNeg, toStringNeg).ForwardOne(variable)
}

func forwardNeg(values []*model.Variable, _ []interface{}) []*model.Variable {
	var result mat.Dense
	result.Scale(-1, values[0].Data)
	return []*model.Variable{{Data: &result}}
}

func backwardNeg(_ []*model.Variable, gradYs []*model.Variable, _ []interface{}) ([]*model.Variable, error) {
	return []*model.Variable{Neg(gradYs[0])}, nil
}

func toStringNeg(values []*model.Variable, _ []interface{}) string {
	return fmt.Sprintf("-%s", values[0].ToString())
}
