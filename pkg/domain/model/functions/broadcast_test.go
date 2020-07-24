package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForwardBroadcast(t *testing.T) {
	x := model.CreateScalarVariable(1.0)
	y := forwardBroadcast([]*model.Variable{x}, []interface{}{2, 3})
	assert.Len(t, y, 1)
	assert.Equal(t, mat.NewDense(2, 3, []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}), y[0].Data)
}

func TestBackwardBroadcast(t *testing.T) {
	yGrad, err := backwardBroadcast(
		[]*model.Variable{
			model.CreateScalarVariable(1.0),
		}, []*model.Variable{
			model.CreateVariable([][]float64{
				{1.0, 1.0, 1.0},
				{1.0, 1.0, 1.0},
			}),
		}, nil)
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(1, 1, []float64{6.0}), yGrad[0].Data)
}
