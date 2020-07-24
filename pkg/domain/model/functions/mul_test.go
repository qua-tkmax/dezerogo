package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForwardMul(t *testing.T) {
	x1 := model.CreateScalarVariable(3.0)
	x2 := model.CreateScalarVariable(2.0)
	y := forwardMul([]*model.Variable{x1, x2}, nil)
	assert.Equal(t, mat.NewDense(1, 1, []float64{6.0}), y[0].Data)
}

func TestForwardMul_Broadcast(t *testing.T) {
	x1 := model.CreateVariable([][]float64{{1, 2}, {3, 4}})
	x2 := model.CreateScalarVariable(10)
	y := forwardMul([]*model.Variable{x1, x2}, nil)
	assert.Equal(t, mat.NewDense(2, 2, []float64{10, 20, 30, 40}), y[0].Data)
}

func TestBackwardMul(t *testing.T) {
	yGrad, err := backwardMul(
		[]*model.Variable{
			model.CreateScalarVariable(3.0),
			model.CreateScalarVariable(2.0),
		}, []*model.Variable{
			model.CreateScalarVariable(1.0),
		}, nil)
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(1, 1, []float64{2.0}), yGrad[0].Data)
	assert.Equal(t, mat.NewDense(1, 1, []float64{3.0}), yGrad[1].Data)
}
