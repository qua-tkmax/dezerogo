package model

import (
	"errors"
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type Variable struct {
	Data       *mat.Dense
	Grad       *Variable
	creator    *Function
	generation int
	Name       string
}

func CreateScalarVariable(value float64) *Variable {
	return &Variable{
		Data: mat.NewDense(1, 1, []float64{value}),
	}
}

func CreateVariable(value [][]float64) *Variable {
	if len(value) == 0 || len(value[0]) == 0 {
		panic("invalid argument")
	}
	row := len(value)
	col := len(value[0])

	data := make([]float64, 0, row*col)
	for _, line := range value {
		if len(line) != col {
			panic("invalid argument")
		}
		for _, v := range line {
			data = append(data, v)
		}
	}

	return &Variable{
		Data: mat.NewDense(row, col, data),
	}
}

func (v *Variable) SetScalarData(value float64) {
	v.Data = mat.NewDense(1, 1, []float64{value})
}

func (v *Variable) GetScalarData() (float64, error) {
	if row, col := v.Data.Dims(); row != 1 || col != 1 {
		return 0, errors.New("data is not scalar value")
	}
	return v.Data.At(0, 0), nil
}

func (v *Variable) DataToString() string {
	return fmt.Sprintf("%+v", v.Data.RawMatrix())
}

func (v *Variable) GradToString() string {
	if v.Grad == nil {
		return "nil"
	}
	return fmt.Sprintf("%+v", v.Grad.Data.RawMatrix())
}

func (v *Variable) ToString() string {
	if v.creator != nil {
		return fmt.Sprintf("(%s)", v.creator.ToString())
	}
	if v.Name == "" {
		if v.Data.RawMatrix().Cols == 1 && v.Data.RawMatrix().Rows == 1 {
			return fmt.Sprintf("%v", v.Data.RawMatrix().Data[0])
		}
		return fmt.Sprintf("%v", v.Data.RawMatrix().Data)
	}
	rows, cols := v.Data.Dims()
	return fmt.Sprintf("%s[%d,%d]", v.Name, rows, cols)
}

func (v *Variable) Backward(retainGrad bool, addFunc func(variable1, variable2 *Variable) *Variable) error {
	if v.Grad == nil {
		grad := mat.DenseCopyOf(v.Data)
		grad.Apply(func(i, j int, v float64) float64 {
			return 1
		}, grad)
		v.Grad = &Variable{
			Data: grad,
		}
	}
	if v.creator == nil {
		return nil
	}
	funcs := []*Function{v.creator}
	seenSet := map[*Function]struct{}{v.creator: {}}
	for len(funcs) > 0 {
		f := funcs[len(funcs)-1]
		funcs = funcs[:len(funcs)-1]
		grads, err := f.backwardFunc(f.Inputs, f.GetOutputGrads(), f.args)
		if err != nil {
			return err
		}
		if !retainGrad {
			for _, output := range f.Outputs {
				output.Grad = nil
			}
		}
		for i, input := range f.Inputs {
			if input.Grad == nil {
				input.Grad = grads[i]
			} else {
				input.Grad = addFunc(input.Grad, grads[i])
			}
			if input.creator != nil {
				if _, ok := seenSet[input.creator]; !ok {
					funcs = append(funcs, input.creator)
					sort.Slice(funcs, func(i, j int) bool {
						return funcs[i].generation < funcs[j].generation
					})
					seenSet[input.creator] = struct{}{}
				}
			}
		}
	}
	return nil
}

func (v *Variable) SetCreator(creator *Function) {
	v.creator = creator
	v.generation = creator.generation + 1
}

func (v *Variable) ClearGrad() {
	v.Grad = nil
}

func (v *Variable) SetName(name string) *Variable {
	v.Name = name
	return v
}
