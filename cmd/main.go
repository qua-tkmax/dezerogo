package main

import (
	"dezerogo/pkg/domain/model"
	"dezerogo/pkg/domain/model/functions"
	"dezerogo/pkg/domain/util"
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	xSlice, ySlice := createToyDataSet(100)
	x := model.CreateVariable(xSlice).SetName("x")
	y := model.CreateVariable(ySlice).SetName("y")

	inputNum := 1
	hideNum := 10
	outputNum := 1
	w1 := createVariable(inputNum, hideNum, func(i, j int, v float64) float64 {
		return 0.01 * rand.Float64()
	}).SetName("W1")
	b1 := createVariable(1, hideNum, func(i, j int, v float64) float64 {
		return 0
	}).SetName("b1")
	w2 := createVariable(hideNum, outputNum, func(i, j int, v float64) float64 {
		return 0.01 * rand.Float64()
	}).SetName("W2")
	b2 := createVariable(1, outputNum, func(i, j int, v float64) float64 {
		return 0
	}).SetName("b2")

	lr := 0.2
	iters := 10000

	testY := calcTest(w1, b1, w2, b2)
	fmt.Println(testY.DataToString())

	for i := 0; i <= iters; i++ {
		yPred := linearSimple(x, w1, b1)
		yPred = sigmoidSimple(yPred)
		yPred = linearSimple(yPred, w2, b2)
		loss := calcLoss(y, yPred)

		w1.ClearGrad()
		b1.ClearGrad()
		w2.ClearGrad()
		b2.ClearGrad()
		loss.ClearGrad()

		if err := loss.Backward(true, functions.Add); err != nil {
			fmt.Println(err)
			return
		}

		fitData(w1, lr)
		fitData(b1, lr)
		fitData(w2, lr)
		fitData(b2, lr)

		if i%1000 == 0 {
			fmt.Printf("loop:%d loss:%f\n", i, loss.Data.At(0, 0))
		}
	}

	testY = calcTest(w1, b1, w2, b2)
	fmt.Println(testY.DataToString())
}

func calcTest(w1, b1, w2, b2 *model.Variable) *model.Variable {
	testNum := 100
	testXSlice := make([][]float64, 0, testNum)
	for i := 0; i <= testNum; i++ {
		testXSlice = append(testXSlice, []float64{float64(i) / float64(testNum)})
	}
	x := model.CreateVariable(testXSlice).SetName("x")
	y := linearSimple(x, w1, b1)
	y = sigmoidSimple(y)
	y = linearSimple(y, w2, b2)

	return y
}

func createToyDataSet(dataSetNum int) ([][]float64, [][]float64) {
	xSlice := make([][]float64, 0, dataSetNum)
	ySlice := make([][]float64, 0, dataSetNum)
	for i := 0; i < dataSetNum; i++ {
		x := rand.Float64()
		y := math.Sin(2*math.Pi*x) + rand.Float64()
		xSlice = append(xSlice, []float64{x})
		ySlice = append(ySlice, []float64{y})
	}
	return xSlice, ySlice
}

func calcLoss(x0, x1 *model.Variable) *model.Variable {
	rows, _ := x0.Data.Dims()
	return functions.Div(
		functions.Sum(functions.Pow(functions.Sub(x0, x1), 2.0), 0),
		model.CreateScalarVariable(float64(rows)),
	)
}

func createVariable(rows, cols int, f func(i, j int, v float64) float64) *model.Variable {
	dense := mat.NewDense(rows, cols, nil)
	dense.Apply(f, dense)
	return &model.Variable{
		Data: dense,
	}
}

func createOnesVariable(rows, cols int) *model.Variable {
	return createVariable(rows, cols, func(i, j int, v float64) float64 {
		return 1.0
	}).SetName("1")
}

func linearSimple(x, w, b *model.Variable) *model.Variable {
	t := functions.MatMul(x, w)
	if b == nil {
		return t
	}
	y := functions.Add(t, b)
	return y
}

func sigmoidSimple(x *model.Variable) *model.Variable {
	rows, cols := x.Data.Dims()
	return functions.Div(
		createOnesVariable(rows, cols),
		functions.Add(
			createOnesVariable(rows, cols),
			functions.Exp(functions.Neg(x)),
		),
	)
}

func fitData(variable *model.Variable, lr float64) {
	rows, cols := variable.Data.Dims()
	grad := mat.DenseCopyOf(variable.Grad.Data)
	grad = util.SumDenses(grad, rows, cols)
	grad.Scale(lr, grad)

	variable.Data.Sub(variable.Data, grad)
}
