package functions

import (
	"dezerogo/pkg/domain/model"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Mul(variable1, variable2 *model.Variable) *model.Variable {
	return model.CreateFunction(forwardMul, backwardMul, toStringMul).ForwardOne(variable1, variable2)
}

func forwardMul(values []*model.Variable, _ []interface{}) []*model.Variable {
	var result mat.Dense
	result.MulElem(values[0].Data, values[1].Data)
	return []*model.Variable{{Data: &result}}
}

func backwardMul(values []*model.Variable, gradYs []*model.Variable, _ []interface{}) ([]*model.Variable, error) {
	rows1, cols1 := values[0].Data.Dims()
	rows2, cols2 := values[1].Data.Dims()
	var result1, result2 *model.Variable
	result1 = Mul(gradYs[0], values[1])
	result2 = Mul(gradYs[0], values[0])
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

func toStringMul(values []*model.Variable, _ []interface{}) string {
	return fmt.Sprintf("%s * %s", values[0].ToString(), values[1].ToString())
}
