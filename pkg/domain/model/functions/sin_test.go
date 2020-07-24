package functions

import (
	"dezerogo/pkg/domain/model"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForwardSin(t *testing.T) {
	x := model.CreateScalarVariable(1.0)
	y := forwardSin([]*model.Variable{x}, nil)
	assert.Equal(t, mat.NewDense(1, 1, []float64{math.Sin(1.0)}), y[0].Data)
}

func TestBackwardSin(t *testing.T) {
	yGrad, err := backwardSin(
		[]*model.Variable{
			model.CreateScalarVariable(1.0),
		}, []*model.Variable{
			model.CreateScalarVariable(1.0),
		}, nil)
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(1, 1, []float64{math.Cos(1.0)}), yGrad[0].Data)
}
