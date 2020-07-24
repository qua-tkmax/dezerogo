package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForwardMatMul(t *testing.T) {
	x1 := model.CreateVariable([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	x2 := model.CreateVariable([][]float64{{5.0, 6.0}, {7.0, 8.0}})
	y := forwardMatMul([]*model.Variable{x1, x2}, nil)
	assert.Equal(t, mat.NewDense(2, 2, []float64{19.0, 22.0, 43.0, 50.0}), y[0].Data)
}

func TestBackwardMatMul(t *testing.T) {
	yGrad, err := backwardMatMul(
		[]*model.Variable{
			model.CreateVariable([][]float64{{1.0, 2.0}, {3.0, 4.0}}),
			model.CreateVariable([][]float64{{5.0, 6.0}, {7.0, 8.0}}),
		}, []*model.Variable{
			model.CreateVariable([][]float64{{1.0, 0.0}, {0.0, 1.0}}),
		}, nil)
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(2, 2, []float64{5.0, 7.0, 6.0, 8.0}), yGrad[0].Data)
	assert.Equal(t, mat.NewDense(2, 2, []float64{1.0, 3.0, 2.0, 4.0}), yGrad[1].Data)
}
