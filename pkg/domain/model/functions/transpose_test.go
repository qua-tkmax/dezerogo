package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForwardTranspose(t *testing.T) {
	x := model.CreateVariable([][]float64{{1, 2, 3}, {4, 5, 6}})
	y := forwardTranspose([]*model.Variable{x}, nil)
	assert.Equal(t, mat.NewDense(3, 2, []float64{1, 4, 2, 5, 3, 6}), y[0].Data)
}

func TestBackwardTranspose(t *testing.T) {
	yGrad, err := backwardTranspose(
		[]*model.Variable{
			model.CreateVariable([][]float64{{1, 2, 3}, {4, 5, 6}}),
		}, []*model.Variable{
			model.CreateVariable([][]float64{{1, 2}, {3, 4}, {5, 6}}),
		}, nil)
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(2, 3, []float64{1, 3, 5, 2, 4, 6}), yGrad[0].Data)
}
