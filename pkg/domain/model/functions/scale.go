package functions

import (
	"dezerogo/pkg/domain/model"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Scale(variable *model.Variable, value float64) *model.Variable {
	return model.CreateFunction(forwardScale, backwardScale, toStringScale, value).ForwardOne(variable)
}

func forwardScale(values []*model.Variable, args []interface{}) []*model.Variable {
	var result mat.Dense
	result.Scale(args[0].(float64), values[0].Data)
	return []*model.Variable{{Data: &result}}
}

func backwardScale(_ []*model.Variable, gradYs []*model.Variable, args []interface{}) ([]*model.Variable, error) {
	return []*model.Variable{Scale(gradYs[0], args[0].(float64))}, nil
}

func toStringScale(values []*model.Variable, args []interface{}) string {
	return fmt.Sprintf("%v * %s", args[0], values[0].ToString())
}
