# curveball-pytorch
An Implementation of "Small steps and giant leaps: Minimal Newton solvers for Deep Learning" In pytorch
link: https://arxiv.org/pdf/1805.08095.pdf
original code: https://github.com/jotaf98/curveball
Authors of the paper: João F. Henriques, Sebastien Ehrhardt, Samuel Albanie, Andrea Vedaldi

# Requirements
pytorch 0.4.0

# Running 
`python opytimizer-tests.py`


# Notes
I had to slightly change the interface for the optimizer to implement the author's more efficient version of computing the hessian-vector product. Including the model / function's output as well as the loss function makes it easy to separate the gradient of the model from the the gradient of the loss function. These terms are reused for the automatic hyper-parameter tuning. pytorch might have a way to do this automatically, without a change of the optimizer's interface, but I haven't devled into it enough to know how to implement it.

when calling optimizer.step, include the output and a function to return the loss, eg.
`optimizer.step(model_output, lambda: loss)`


