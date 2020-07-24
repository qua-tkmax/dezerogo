package functions

import (
	"dezerogo/pkg/domain/model"
	"github.com/stretchr/testify/assert"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestForwardReshape(t *testing.T) {
	x := model.CreateVariable([][]float64{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}})
	y := forwardReshape([]*model.Variable{x}, []interface{}{2, 5})
	assert.Equal(t, mat.NewDense(2, 5, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}), y[0].Data)
}

func TestBackwardReshape(t *testing.T) {
	yGrad, err := backwardReshape(
		[]*model.Variable{
			model.CreateVariable([][]float64{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}),
		}, []*model.Variable{
			model.CreateVariable([][]float64{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}),
		}, []interface{}{
			2, 5,
		})
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(1, 10, []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}), yGrad[0].Data)
}
