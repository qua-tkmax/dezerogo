package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForwardSum(t *testing.T) {
	x := model.CreateVariable([][]float64{{1, 2, 3}, {4, 5, 6}})
	y := forwardSum([]*model.Variable{x}, []interface{}{0})
	assert.Equal(t, mat.NewDense(1, 1, []float64{21}), y[0].Data)
}

func TestBackwardSum(t *testing.T) {
	yGrad, err := backwardSum(
		[]*model.Variable{
			model.CreateVariable([][]float64{{1, 2, 3}, {4, 5, 6}}),
		}, []*model.Variable{
			model.CreateScalarVariable(1.0),
		}, []interface{}{
			0,
		})
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(2, 3, []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}), yGrad[0].Data)
}
