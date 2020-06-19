package functions

import (
	"dezerogo/pkg/domain/model"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func Cos(variable *model.Variable) *model.Variable {
	return model.CreateFunction(forwardCos, backwardCos, toStringCos).ForwardOne(variable)
}

func forwardCos(values []*model.Variable, _ []interface{}) []*model.Variable {
	var result mat.Dense
	result.Apply(func(i, j int, v float64) float64 {
		return math.Cos(v)
	}, values[0].Data)
	return []*model.Variable{{Data: &result}}
}

func backwardCos(values []*model.Variable, gradYs []*model.Variable, _ []interface{}) ([]*model.Variable, error) {
	return []*model.Variable{Mul(gradYs[0], Neg(Sin(values[0])))}, nil
}

func toStringCos(values []*model.Variable, _ []interface{}) string {
	return fmt.Sprintf("cos(%s)", values[0].ToString())
}
