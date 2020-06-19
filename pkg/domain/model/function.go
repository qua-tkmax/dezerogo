package model

type Function struct {
	forwardFunc  func(values []*Variable, args []interface{}) []*Variable
	backwardFunc func(values []*Variable, gradYs []*Variable, args []interface{}) ([]*Variable, error)
	stringFunc   func(values []*Variable, args []interface{}) string
	Inputs       []*Variable
	Outputs      []*Variable
	generation   int
	args         []interface{}
}

func CreateFunction(
	forwardFunc func(values []*Variable, args []interface{}) []*Variable,
	backwardFunc func(values []*Variable, gradYs []*Variable, args []interface{}) ([]*Variable, error),
	stringFunc func(values []*Variable, args []interface{}) string,
	args ...interface{},
) *Function {
	return &Function{
		forwardFunc:  forwardFunc,
		backwardFunc: backwardFunc,
		stringFunc:   stringFunc,
		args:         args,
	}
}

func (f *Function) ForwardOne(values ...*Variable) *Variable {
	return f.Forward(values...)[0]
}

func (f *Function) Forward(values ...*Variable) []*Variable {
	outputs := f.forwardFunc(values, f.args)

	f.Inputs = values
	var generation int
	for _, input := range f.Inputs {
		if generation < input.generation {
			generation = input.generation
		}
	}
	f.generation = generation
	f.Outputs = outputs
	for _, output := range outputs {
		output.SetCreator(f)
	}

	return outputs
}

func (f *Function) GetOutputGrads() []*Variable {
	grads := make([]*Variable, 0, len(f.Outputs))
	for _, output := range f.Outputs {
		grads = append(grads, output.Grad)
	}
	return grads
}

func (f *Function) ToString() string {
	return f.stringFunc(f.Inputs, f.args)
}
