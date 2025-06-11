# L2ODLL.jl

!!! warning
    This documentation is a work in progress.
    Please open an issue if content is missing / erroneous.

L2ODLL.jl implements the Dual Lagrangian Learning (DLL) method of [Tanneau and Hentenryck (2024)](https://arxiv.org/pdf/2402.03086) using JuMP.

## Installation

```julia
import Pkg
Pkg.add(; url="https://github.com/LearningToOptimize/L2ODLL.jl")
```

## Usage


This package simplifies the implementation of DLL by taking as input a primal JuMP model, then automatically generating the dual projection and completion functions which can be used in the training and inference of DLL models. The basic usage is as follows:


#### Define your (primal) model using JuMP

For the purposes of this example, we'll use a portfolio optimization problem.
```julia
using JuMP, LinearAlgebra

model = Model()

# define constant problem data
Σ = [166 34 58; 34 64 4; 58 4 100] / 100^2
N = size(Σ, 1)

# define variables
@variable(model, x[1:N])
set_lower_bound.(x, 0)  # we explicitly set upper and lower bounds
set_upper_bound.(x, 1)  #   in order to use the BoundDecomposition

# define parameteric problem data
μ0 = randn(N)
γ0 = rand()
@variable(model, μ[1:N] in MOI.Parameter.(μ0))
@variable(model, γ in MOI.Parameter(γ0))

# define constraints
@constraint(model, simplex, sum(x) == 1)
@constraint(model, risk, [γ; cholesky(Σ).L * x] in SecondOrderCone())

# define objective
@objective(model, Max, dot(μ,x))
```

#### Decompose and build the functions

Since all the variables have finite bounds, L2ODLL will automatically pick the `BoundDecomposition`.
```julia
using L2ODLL

L2ODLL.decompose!(model)
```

Now, L2ODLL has automatically generated the dual projection and completion layer. To compute the dual objective value and gradient with respect to the prediction, use:
```julia
param_value = ... # some values for μ and γ
y_predicted = nn(param_value) # e.g. neural network inference

dobj = L2ODLL.dual_objective(model, y_predicted, param_value)
dobj_wrt_y = L2ODLL.dual_objective_gradient(model, y_predicted, param_value)
```

This also works with batches, using broadcasting:
```julia
dobj = L2ODLL.dual_objective.(model, y_predicted_batch, param_value_batch)
dobj_wrt_y = L2ODLL.dual_objective_gradient.(model, y_predicted_batch, param_value_batch)
```

!!! warning
    These functions currently run on the CPU. A batched GPU-friendly version is coming soon.