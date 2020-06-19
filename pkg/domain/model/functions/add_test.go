package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForward(t *testing.T) {
	x1 := model.CreateScalarVariable(2.0)
	x2 := model.CreateScalarVariable(3.0)
	y := forward([]*model.Variable{x1, x2}, nil)
	assert.Len(t, y, 1)
	assert.Equal(t, mat.NewDense(1, 1, []float64{5.0}), y[0].Data)
}

func TestBackward(t *testing.T) {
	yGrad, err := backward(
		[]*model.Variable{
			model.CreateScalarVariable(3.0),
			model.CreateScalarVariable(3.0),
		}, []*mat.Dense{
			mat.NewDense(1, 1, []float64{1.0}),
		}, nil)
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(1, 1, []float64{1.0}), yGrad[0])
}
