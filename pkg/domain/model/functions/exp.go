package functions

import (
	"dezerogo/pkg/domain/model"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func Exp(variable *model.Variable) *model.Variable {
	return model.CreateFunction(forwardExp, backwardExp, toStringExp).ForwardOne(variable)
}

func forwardExp(values []*model.Variable, _ []interface{}) []*model.Variable {
	var result mat.Dense
	result.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v)
	}, values[0].Data)
	return []*model.Variable{{Data: &result}}
}

func backwardExp(values []*model.Variable, gradYs []*model.Variable, _ []interface{}) ([]*model.Variable, error) {
	return []*model.Variable{Mul(gradYs[0], Exp(values[0]))}, nil
}

func toStringExp(values []*model.Variable, _ []interface{}) string {
	return fmt.Sprintf("exp(%s)", values[0].ToString())
}
