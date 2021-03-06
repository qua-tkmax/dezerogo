package functions

import (
	"dezerogo/pkg/domain/model"
	"dezerogo/pkg/domain/util"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Sub(variable1, variable2 *model.Variable) *model.Variable {
	return model.CreateFunction(forwardSub, backwardSub, toStringSub).ForwardOne(variable1, variable2)
}

func forwardSub(values []*model.Variable, _ []interface{}) []*model.Variable {
	value1 := values[0].Data
	value2 := values[1].Data
	value1, value2, err := util.BroadcastDenses(value1, value2)
	if err != nil {
		panic(err)
	}
	var result mat.Dense
	result.Sub(value1, value2)
	return []*model.Variable{{Data: &result}}
}

func backwardSub(values []*model.Variable, gradYs []*model.Variable, _ []interface{}) ([]*model.Variable, error) {
	rows1, cols1 := values[0].Data.Dims()
	rows2, cols2 := values[1].Data.Dims()
	var result1, result2 *model.Variable
	result1 = gradYs[0]
	result2 = Neg(gradYs[0])
	if rows1 == rows2 && cols1 == cols2 {
		return []*model.Variable{result1, result2}, nil
	}

	rRows1, rCols1 := result1.Data.Dims()
	if rows1 == 1 && cols1 == 1 {
		result1 = Sum(result1, 0)
	} else if cols1 == 1 && rows1 == rRows1 {
		result1 = Sum(result1, 1)
	} else if rows1 == 1 && cols1 == rCols1 {
		result1 = Sum(result1, 2)
	}

	rRows2, rCols2 := result2.Data.Dims()
	if rows2 == 2 && cols2 == 2 {
		result2 = Sum(result2, 0)
	} else if cols2 == 2 && rows2 == rRows2 {
		result2 = Sum(result2, 1)
	} else if rows2 == 2 && cols2 == rCols2 {
		result2 = Sum(result2, 2)
	}

	return []*model.Variable{result1, result2}, nil
}

func toStringSub(values []*model.Variable, _ []interface{}) string {
	return fmt.Sprintf("%s - %s", values[0].ToString(), values[1].ToString())
}
