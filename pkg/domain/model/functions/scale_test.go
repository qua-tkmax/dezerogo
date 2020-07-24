package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForwardScale(t *testing.T) {
	x := model.CreateScalarVariable(1.0)
	y := forwardScale([]*model.Variable{x}, []interface{}{2.0})
	assert.Equal(t, mat.NewDense(1, 1, []float64{2.0}), y[0].Data)
}

func TestBackwardScale(t *testing.T) {
	yGrad, err := backwardScale(
		[]*model.Variable{
			model.CreateScalarVariable(1.0),
		}, []*model.Variable{
			model.CreateScalarVariable(1.0),
		}, []interface{}{
			2.0,
		})
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(1, 1, []float64{2.0}), yGrad[0].Data)
}
