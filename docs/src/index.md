# L2ODLL.jl

L2ODLL.jl implements **Dual Lagrangian Learning** (DLL) of
[Tanneau and Van Hentenryck (2024)](https://arxiv.org/abs/2402.03086) for parametric
conic and convex quadratic problems modeled in [JuMP](https://jump.dev), with the
bounded-variable completions of
[Klamkin, Tanneau and Van Hentenryck](https://arxiv.org/abs/2402.02596).

A neural network (or any proxy) predicts some of the dual variables ``\hat y`` of a
primal problem from its parameters ``p``. L2ODLL projects the prediction onto the
dual cone and **completes** the remaining dual variables so that the result is dual
feasible. Its dual objective is therefore a **valid bound** on the optimal value for
*every* prediction (weak duality), and it is differentiable in ``\hat y``, so it is
also the self-supervised training loss: maximizing it needs no optimal solutions.

```mermaid
flowchart LR
    p["parameters p"] --> f["proxy f(p)"]
    f -->|"ŷ"| proj["Π onto dual cone"]
    proj -->|"y"| comp["dual completion z(y, p)"]
    comp --> bound["Φ(y, p) ≤ OPT(p)"]
    bound -.->|"∇ŷ Φ (training)"| f
```

L2ODLL takes the *primal* JuMP model, dualizes it with Dualization.jl, detects a
decomposition with a closed-form completion and builds the projection, completion,
bound and gradient functions automatically.

## Installation

L2ODLL is not registered yet. Its optional `L2OCore` interface
([`L2ODLL.DLLMethod`](@ref)) comes from the development version of
LearningToOptimize.jl; the registered LearningToOptimize 1.0.0 predates it. Install
the core subpackage first, then L2ODLL, on any Julia version from 1.10:

```julia
import Pkg
Pkg.add(; url = "https://github.com/LearningToOptimize/LearningToOptimize.jl", subdir = "lib/L2OCore")
Pkg.add(; url = "https://github.com/LearningToOptimize/L2ODLL.jl")
Pkg.add("JuMP")   # used by every example
```

The examples also use HiGHS and Clarabel: `Pkg.add(["HiGHS", "Clarabel"])`.

## Example

```julia
using JuMP, L2ODLL

model = Model()
@variable(model, 0 <= x[1:2] <= 1)
@variable(model, d in MOI.Parameter(1.0))          # parameter: the demand
@constraint(model, balance, x[1] + x[2] == d)
@objective(model, Min, x[1] + 3x[2])

decompose!(model)                                   # detects BoundDecomposition

dual_objective(model, [[2.0]], [1.5])               # 2.0 ≤ OPT(1.5) = 2.5, for any prediction
dual_objective(model, [[3.0]], [1.5])               # 2.5: the optimal price gives the optimum
dual_objective_gradient(model, [[2.0]], [1.5])      # training signal

Y = randn(sum(y_shape(model)), 64)                  # batch: predictions × instances
P = 1 .+ rand(1, 64)                                # parameters × instances
bounds, G = dual_objective_and_gradient(model, Y, P)
```

## Where to go next

- [Theory](theory.md): the optimization problems, the completions in closed form
  and their gradients, with equation numbers of the papers.
- [Package guide](guide/concepts.md): getting started, concepts, custom
  decompositions, projections and layers.
- [L2OCore interface](guide/l2ocore.md): using L2ODLL as a method of the
  LearningToOptimize pipeline; AD and GPU notes.
- Examples: [Markowitz portfolio](examples/markowitz.md),
  [economic dispatch](examples/economic_dispatch.md) (training a proxy).

## The LearningToOptimize ecosystem

| Package | What it provides |
|:--|:--|
| [LearningToOptimize.jl](https://github.com/LearningToOptimize/LearningToOptimize.jl) | the penalty method, Flux training and solver-labeled dataset generation; hosts the two subpackages below |
| ↳ [L2OCore](https://github.com/LearningToOptimize/LearningToOptimize.jl/tree/main/lib/L2OCore) | the shared interface: method families, problem wrapper, batched evaluators, datasets |
| ↳ [L2OSobolev](https://github.com/LearningToOptimize/LearningToOptimize.jl/tree/main/lib/L2OSobolev) | derivative (Sobolev) supervision added to any method |
| [L2ODLL.jl](https://github.com/LearningToOptimize/L2ODLL.jl) | Dual Lagrangian Learning: dual proxies that give valid dual bounds for conic problems |
| [L2ODC3.jl](https://github.com/LearningToOptimize/L2ODC3.jl) | DC3: equality completion and inequality correction for primal proxies |
| [L2OALM.jl](https://github.com/LearningToOptimize/L2OALM.jl) | primal-dual learning with an augmented Lagrangian |
