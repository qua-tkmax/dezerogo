package functions

import (
	"dezerogo/pkg/domain/model"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func Sin(variable *model.Variable) *model.Variable {
	return model.CreateFunction(forwardSin, backwardSin, toStringSin).ForwardOne(variable)
}

func forwardSin(values []*model.Variable, _ []interface{}) []*model.Variable {
	var result mat.Dense
	result.Apply(func(i, j int, v float64) float64 {
		return math.Sin(v)
	}, values[0].Data)
	return []*model.Variable{{Data: &result}}
}

func backwardSin(values []*model.Variable, gradYs []*model.Variable, args []interface{}) ([]*model.Variable, error) {
	return []*model.Variable{Mul(gradYs[0], Cos(values[0]))}, nil
}

func toStringSin(values []*model.Variable, _ []interface{}) string {
	return fmt.Sprintf("sin(%s)", values[0].ToString())
}
