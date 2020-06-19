package functions

import (
	"dezerogo/pkg/domain/model"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func Pow(variable *model.Variable, value float64) *model.Variable {
	return model.CreateFunction(forwardPow, backwardPow, toStringPow, value).ForwardOne(variable)
}

func forwardPow(values []*model.Variable, args []interface{}) []*model.Variable {
	arg := args[0].(float64)
	var result mat.Dense
	result.Apply(func(i, j int, v float64) float64 {
		return math.Pow(v, arg)
	}, values[0].Data)
	return []*model.Variable{{Data: &result}}
}

func backwardPow(values []*model.Variable, gradYs []*model.Variable, args []interface{}) ([]*model.Variable, error) {
	arg := args[0].(float64)
	if arg == 0.0 {
		return []*model.Variable{model.CreateScalarVariable(0)}, nil
	}
	return []*model.Variable{Mul(Scale(Pow(values[0], arg-1), arg), gradYs[0])}, nil
}

func toStringPow(values []*model.Variable, args []interface{}) string {
	if args[0] == 0.0 {
		return "1"
	}
	return fmt.Sprintf("%s ^ %v", values[0].ToString(), args[0])
}
