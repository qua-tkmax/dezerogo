package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForwardDiv(t *testing.T) {
	x1 := model.CreateScalarVariable(1.0)
	x2 := model.CreateScalarVariable(2.0)
	y := forwardDiv([]*model.Variable{x1, x2}, nil)
	assert.Equal(t, mat.NewDense(1, 1, []float64{0.5}), y[0].Data)
}

func TestForwardDiv_Broadcast(t *testing.T) {
	x1 := model.CreateVariable([][]float64{{1, 2}, {3, 4}})
	x2 := model.CreateScalarVariable(2)
	y := forwardDiv([]*model.Variable{x1, x2}, nil)
	assert.Equal(t, mat.NewDense(2, 2, []float64{0.5, 1, 1.5, 2}), y[0].Data)
}

func TestBackwardDiv(t *testing.T) {
	yGrad, err := backwardDiv(
		[]*model.Variable{
			model.CreateScalarVariable(1.0),
			model.CreateScalarVariable(2.0),
		}, []*model.Variable{
			model.CreateScalarVariable(1.0),
		}, nil)
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(1, 1, []float64{0.5}), yGrad[0].Data)
	assert.Equal(t, mat.NewDense(1, 1, []float64{-0.25}), yGrad[1].Data)
}
