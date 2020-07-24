package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForwardSub(t *testing.T) {
	x1 := model.CreateScalarVariable(3.0)
	x2 := model.CreateScalarVariable(2.0)
	y := forwardSub([]*model.Variable{x1, x2}, nil)
	assert.Equal(t, mat.NewDense(1, 1, []float64{1.0}), y[0].Data)
}

func TestForwardSub_Broadcast(t *testing.T) {
	x1 := model.CreateVariable([][]float64{{1, 2}, {3, 4}})
	x2 := model.CreateScalarVariable(10)
	y := forwardSub([]*model.Variable{x1, x2}, nil)
	assert.Equal(t, mat.NewDense(2, 2, []float64{-9, -8, -7, -6}), y[0].Data)
}

func TestBackwardSub(t *testing.T) {
	yGrad, err := backwardSub(
		[]*model.Variable{
			model.CreateScalarVariable(3.0),
			model.CreateScalarVariable(2.0),
		}, []*model.Variable{
			model.CreateScalarVariable(1.0),
		}, nil)
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(1, 1, []float64{1}), yGrad[0].Data)
	assert.Equal(t, mat.NewDense(1, 1, []float64{-1}), yGrad[1].Data)
}
