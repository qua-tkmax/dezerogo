package functions

import (
	"dezerogo/pkg/domain/model"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestForward(t *testing.T) {
	x := model.CreateScalarVariable(2.0)
	y := forward([]*model.Variable{x}, []interface{}{2.0})
	assert.Equal(t, mat.NewDense(1, 1, []float64{4.0}), y[0].Data)
}

func TestBackward(t *testing.T) {
	yGrad, err := backward(
		[]*model.Variable{
			model.CreateScalarVariable(3.0),
		}, []*mat.Dense{
			mat.NewDense(1, 1, []float64{1.0}),
		}, []interface{}{
			2.0,
		})
	assert.NoError(t, err)
	assert.Equal(t, mat.NewDense(1, 1, []float64{6.0}), yGrad[0])
}
