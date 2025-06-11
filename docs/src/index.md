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

## Math Background

### Decomposition

In DLL, the primal constraints (dual variables) are decomposed into a predicted set and a completed set.
Consider the primal-dual pair:
```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & x \in \mathbb{R}^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{y} & - b^\top y
\\
& \;\;\text{s.t.} & A^\top y = c
\\
& & y \in \mathcal{C}^*
\end{aligned}
\end{equation}
```
After the decomposition, we have:
```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& \;\;\phantom{\text{s.t.}} & Hx + h \in \mathcal{K}
\\
& & x \in \mathbb{R}^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{y} & - b^\top y
\\
& \;\;\text{s.t.} & A^\top y + H^\top z = c
\\
& & y \in \mathcal{C}^*,\; z \in \mathcal{K}^*
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{z} & - h^\top z - b^\top y
\\
& \;\;\text{s.t.} & H z = c - A^\top y
\\
& & z \in \mathcal{K}^*
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \min\nolimits_{x} & (c-A^\top y)^\top x
\\
& \;\;\phantom{\text{s.t.}} & Hx + h \in \mathcal{K}
\\
& & x \in \mathbb{R}^n
\end{aligned}
\end{equation}
```

To train the neural network, we need the gradient of the optimal value with respect to the predicted $y$. This is $\nabla_y = -b-Ax$ where $x$ is the optimal dual solution corresponding to the affine constraints in the completion model. In the special cases below, we specify just the expression for $x$ in this formula.


#### Bounded Decomposition

When all primal variables have finite upper and lower bounds, a natural way to decompose the constraints is to have $z$ correspond to the bound constraints, and $y$ correspond to the main constraints, i.e.

```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & l \leq x \leq u
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{y,z_l,z_u} & - b^\top y - l^\top z_l - u^\top z_u
\\
& \;\;\text{s.t.} & A^\top y + z_l + z_u = c
\\
& & y \in \mathcal{C}^*,\; z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{z_l,z_u} & - l^\top z_l - u^\top z_u - b^\top y
\\
& \;\;\text{s.t.} & z_l + z_u = c - A^\top y
\\
& & z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \min\nolimits_{x} & (c-A^\top y)^\top x
\\
& & l \leq x \leq u
\end{aligned}
\end{equation}
```

This model admits a closed form solution, $z_l = |c-A^\top y|^+$ and $z_u = -|c-A^\top y|^-$. Furthermore, the $x$ that defines the (sub-)gradient is given element-wise by $l$ if $c-A^\top y > 0$, $u$ if $c-A^\top y < 0$, and $x\in[l,u]$ otherwise.


#### (Strictly) Convex QP

In the convex QP case, the primal has a strictly convex quadratic objective function, i.e. $Q\succ 0$. In that case it is natural to use the main constraints as the predicted set and to complete the quadratic slack dual variables.

```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & x \in \mathbb{R}^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{y} & - b^\top y - z^\top Q z
\\
& \;\;\text{s.t.} & A^\top y + Qz = c
\\
& & y \in \mathcal{C}^*,\; z \in \mathbb{R}^n
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{z} & - z^\top Q z - b^\top y
\\
& \;\;\text{s.t.} & Q z = c - A^\top y
\\
& & z \in \mathbb{R}^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + (c-A^\top y)^\top x
\\
& & x \in \mathbb{R}^n
\end{aligned}
\end{equation}
```

This model admits a closed form solution, $z = Q^{-1}(c - A^\top y)$. Furthermore, the closed form dual solution in this case is $x=z$.

### Decomposition detection

!!! warning
    The decomposition detection is very basic, based on minimizing the number of predicted dual variables. In some cases, it may actually be preferred to to predict more dual variables, if they are for some reason easier to learn. In that case, users should manually specify the decomposition.

L2ODLL automatically detects the decomposition to use based on the model.
The current detection logic is as follows:

- If all variables have finite upper and lower bounds, use the bounded decomposition.
- If all variables have a quadratic objective term, use the convex QP decomposition.
- Otherwise, use the generic decomposition.

The order of preference is to maximize the number of completed dual variables among the closed-form solutions, and if no closed-form solution is available, to use the generic decomposition. Thus, the bounded decomposition is preferred over the convex QP decomposition, as illustrated by the following example.

#### Bounded with quadratic objective

In the case where bounded decomposition is used with a model that has a quadratic objective, L2ODLL prefers the bounded decomposition, leading to the following models.
```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & l \leq x \leq u
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{w,y,z_l,z_u} & - b^\top y - w^\top Q w - l^\top z_l - u^\top z_u
\\
& \;\;\text{s.t.} & A^\top y + Qw + z_l + z_u = c
\\
& & y \in \mathcal{C}^*,\; w \in \mathbb{R}^n,\; z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{z_l,z_u} & - l^\top z_l - u^\top z_u - b^\top y - w^\top Q w
\\
& \;\;\text{s.t.} & z_l + z_u = c - A^\top y - Qw   
\\
& & z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \min\nolimits_{x} & (c-A^\top y - Qw)^\top x
\\
& & l \leq x \leq u
\end{aligned}
\end{equation}
```

This model admits a closed form solution, $z_l = |c-A^\top y-Qw|^+$ and $z_u = -|c-A^\top y-Qw|^-$. Furthermore, the $x$ that defines the (sub-)gradient is given element-wise by $l$ if $c-A^\top y-Qw > 0$, $u$ if $c-A^\top y-Qw < 0$, and $x\in[l,u]$ otherwise. Note that the gradient of the objective with respect to $w$ is $\nabla_w = -Qw -Q^\top x$.

This completes $2n$ dual variables, leaving the neural network to predict $m+n$ dual variables.


Consider the convex QP decomposition in this case:
```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & l \leq x \leq u
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{w,y,z_l,z_u} & - b^\top y - w^\top Q w - l^\top z_l - u^\top z_u
\\
& \;\;\text{s.t.} & A^\top y + Qw + z_l + z_u = c
\\
& & y \in \mathcal{C}^*,\; w \in \mathbb{R}^n,\; z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{w} & - w^\top Q w - b^\top y - l^\top z_l - u^\top z_u 
\\
& \;\;\text{s.t.} & Qw = c - A^\top y - z_l - z_u   
\\
& & w \in \mathbb{R}^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + (c-A^\top y - z_l - z_u)^\top x
\\
& & x \in \mathbb{R}^n
\end{aligned}
\end{equation}
```

This model admits a closed form solution, $w = Q^{-1}(c - A^\top y - z_l - z_u)$. Furthermore, the closed form dual solution in this case is $x=w$.

This completes $n$ dual variables, leaving the neural network to predict $m+2n$ dual variables.

Naturally, one may consider using both decompositions, i.e. to predict only the $y$ variables and to recover $w$, $z_l$, and $z_u$. Let us consider this case:
```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & l \leq x \leq u
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{w,y,z_l,z_u} & - b^\top y - w^\top Q w - l^\top z_l - u^\top z_u
\\
& \;\;\text{s.t.} & A^\top y + Qw + z_l + z_u = c
\\
& & y \in \mathcal{C}^*,\; w \in \mathbb{R}^n,\; z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{w,z_l,z_u} & - w^\top Q w - l^\top z_l - u^\top z_u - b^\top y
\\
& \;\;\text{s.t.} & Qw + z_l + z_u = c - A^\top y
\\
& & w \in \mathbb{R}^n,\; z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + (c-A^\top y)^\top x
\\
& & l \leq x \leq u
\end{aligned}
\end{equation}
```

This is an $n$-dimensional box-constrained convex QP, for which there is no closed form solution. Note that by using a custom generic decomposition, L2ODLL can still be used to set up this problem and solve it using a JuMP-compatible QP solver.