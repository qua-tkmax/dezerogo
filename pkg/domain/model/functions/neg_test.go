package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForwardNeg(t *testing.T) {
	x := model.CreateScalarVariable(2.0)
	y := forwardNeg([]*model.Variable{x}, nil)
	assert.Equal(t, mat.NewDense(1, 1, []float64{-2.0}), y[0].Data)
}

func TestBackwardNeg(t *testing.T) {
	yGrad, err := backwardNeg(
		[]*model.Variable{
			model.CreateScalarVariable(2.0),
		}, []*model.Variable{
			model.CreateScalarVariable(1.0),
		}, nil)
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(1, 1, []float64{-1}), yGrad[0].Data)
}
