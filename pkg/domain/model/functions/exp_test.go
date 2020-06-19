package functions

import (
	"dezerogo/pkg/domain/model"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForward(t *testing.T) {
	x := model.CreateScalarVariable(2.0)
	y := forward([]*model.Variable{x}, nil)
	assert.Equal(t, mat.NewDense(1, 1, []float64{math.Exp(2.0)}), y[0].Data)
}

func TestBackward(t *testing.T) {
	yGrad, err := backward(
		[]*model.Variable{
			model.CreateScalarVariable(2.0),
		}, []*mat.Dense{
			mat.NewDense(1, 1, []float64{1.0}),
		}, nil)
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(1, 1, []float64{math.Exp(2.0) * 1}), yGrad[0])
}
