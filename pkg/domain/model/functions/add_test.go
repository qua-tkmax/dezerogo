package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForwardAdd(t *testing.T) {
	x1 := model.CreateScalarVariable(2.0)
	x2 := model.CreateScalarVariable(3.0)
	y := forwardAdd([]*model.Variable{x1, x2}, nil)
	assert.Len(t, y, 1)
	assert.Equal(t, mat.NewDense(1, 1, []float64{5.0}), y[0].Data)
}

func TestForwardAdd_Broadcast(t *testing.T) {
	x1 := model.CreateVariable([][]float64{{1, 2}, {3, 4}})
	x2 := model.CreateScalarVariable(10)
	y := forwardAdd([]*model.Variable{x1, x2}, nil)
	assert.Equal(t, mat.NewDense(2, 2, []float64{11, 12, 13, 14}), y[0].Data)
}

func TestBackwardAdd(t *testing.T) {
	yGrad, err := backwardAdd(
		[]*model.Variable{
			model.CreateScalarVariable(3.0),
			model.CreateScalarVariable(3.0),
		}, []*model.Variable{
			model.CreateScalarVariable(1.0),
		}, nil)
	assert.NoError(t, err)
	assert.Len(t, yGrad, 2)
	assert.Equal(t, mat.NewDense(1, 1, []float64{1.0}), yGrad[0].Data)
	assert.Equal(t, mat.NewDense(1, 1, []float64{1.0}), yGrad[1].Data)
}
