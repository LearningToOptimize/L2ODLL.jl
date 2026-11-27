<div align="center">

# L2ODLL.jl

<p><sub><b>DUAL PROXIES · DUAL COMPLETION · LAGRANGIAN BOUNDS</b></sub></p>

[![CI](https://github.com/LearningToOptimize/L2ODLL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LearningToOptimize/L2ODLL.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://learningtooptimize.github.io/L2ODLL.jl/dev/) [![Codecov](https://codecov.io/gh/LearningToOptimize/L2ODLL.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/LearningToOptimize/L2ODLL.jl) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

</div>

<p align="center">
  <picture>
    <source media="(max-width: 640px)" srcset="docs/src/assets/diagrams/l2odll--dual-completion-compact.svg">
    <img src="docs/src/assets/diagrams/l2odll--dual-completion.svg" width="760" alt="Dual Lagrangian Learning. Forward: the proxy f_theta maps the parameters p to a predicted dual y-hat; it is projected onto the dual cone, y-bar equals Pi_C-star of y-hat; the completion z of y-bar and p solves the completion problem (3), Phi of y-bar and p equals the maximum over z of minus h of p transpose z minus b of p transpose y-bar, subject to H transpose z equals c of p minus A transpose y-bar and z in the dual cone K-star; closed forms exist for bounded variables (4) and strictly convex quadratic objectives (6); by weak duality Phi is at most OPT of p for minimization. Training (7): maximize the expected Phi over theta, with the gradient of Phi with respect to y-hat returned to theta; no solved instances are needed. decompose! builds the projection, the completion and Phi once from the primal JuMP model.">
  </picture>
</p>
<p align="center"><i>The proxy predicts some dual variables; projection and completion make them dual feasible, so their dual objective is a bound on the optimal value and, maximized, the training loss.</i></p>

L2ODLL implements [Dual Lagrangian Learning for Conic Optimization](https://arxiv.org/abs/2402.03086) (DLL) for parametric problems written in [JuMP](https://github.com/jump-dev/JuMP.jl). A proxy (e.g. a neural network) predicts some dual variables of a parametric conic or convex quadratic problem; L2ODLL projects the prediction onto the dual cone and completes the remaining dual variables in closed form, so that the result is dual feasible. Its dual objective is then a **bound on the optimal value for every prediction**, and it is differentiable, so it doubles as a self-supervised training loss that needs no optimal solutions. You write only the *primal* JuMP model: L2ODLL dualizes it ([Dualization.jl](https://github.com/jump-dev/Dualization.jl)), detects a decomposition with a closed-form completion and builds the projection, completion, bound and gradient functions.

## Installation

L2ODLL is not registered yet. Its optional `L2OCore` interface (`L2ODLL.DLLMethod()`)
comes from the development version of LearningToOptimize.jl; the registered
LearningToOptimize 1.0.0 predates it. Install the core subpackage first, then L2ODLL,
on any Julia version from 1.10:

```julia
import Pkg
Pkg.add(; url = "https://github.com/LearningToOptimize/LearningToOptimize.jl", subdir = "lib/L2OCore")
Pkg.add(; url = "https://github.com/LearningToOptimize/L2ODLL.jl")
```

The example below also needs JuMP: `Pkg.add("JuMP")`.

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

See the [economic dispatch example](https://learningtooptimize.github.io/L2ODLL.jl/dev/examples/economic_dispatch/)
for training a proxy (it also uses HiGHS and L2OCore).

## What you need to provide

- A JuMP model whose problem data that changes between instances are **parameters**
  (`@variable(model, p in MOI.Parameter(value))`), entering the constraint constants
  and the linear objective coefficients.
- A problem with a closed-form completion: **every variable bounded** with an
  objective affine in the variables (any cones in the constraints), or a **strictly
  convex quadratic objective** (positive definite Hessian; negative definite for
  `Max`). Other
  problems work with a hand-built `GenericDecomposition` and a solver for the
  completion.
- A proxy producing `sum(y_shape(model))` outputs per instance, in any Julia ML
  framework (gradients work with ForwardDiff and Zygote).

## The dual bound

For these problems the completion is always dual feasible, so by weak duality
`dual_objective` is a bound on the optimal value for every prediction, and completing
the optimal duals recovers the optimal value (Theorems 1–2 of the DLL paper). The
[Theory page](https://learningtooptimize.github.io/L2ODLL.jl/dev/theory/) writes out
the problems and the closed-form completions.

## LearningToOptimize ecosystem

L2ODLL implements the method interface of `L2OCore` as a dual method
(`import L2OCore` loads it; `L2ODLL.DLLMethod()`), so it can be trained, compared and
evaluated in the same pipeline as the other methods of the ecosystem.

| Package | What it provides |
|:--|:--|
| [LearningToOptimize.jl](https://github.com/LearningToOptimize/LearningToOptimize.jl) | the penalty method, Flux training and solver-labeled dataset generation; hosts the two subpackages below |
| ↳ [L2OCore](https://github.com/LearningToOptimize/LearningToOptimize.jl/tree/main/lib/L2OCore) | the shared interface: method families, problem wrapper, batched evaluators, datasets |
| ↳ [L2OSobolev](https://github.com/LearningToOptimize/LearningToOptimize.jl/tree/main/lib/L2OSobolev) | derivative (Sobolev) supervision added to any method |
| [L2ODLL.jl](https://github.com/LearningToOptimize/L2ODLL.jl) | Dual Lagrangian Learning: dual proxies that give valid dual bounds for conic problems |
| [L2ODC3.jl](https://github.com/LearningToOptimize/L2ODC3.jl) | DC3: equality completion and inequality correction for primal proxies |
| [L2OALM.jl](https://github.com/LearningToOptimize/L2OALM.jl) | primal-dual learning with an augmented Lagrangian |

## Citation

If you use L2ODLL, please cite the DLL paper (and the interior-point follow-up if you
use the smoothed bounded completion); see also [`CITATION.cff`](CITATION.cff).

```bibtex
@inproceedings{tanneau2024dual,
  title     = {Dual Lagrangian Learning for Conic Optimization},
  author    = {Tanneau, Mathieu and Van Hentenryck, Pascal},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024},
  note      = {arXiv:2402.03086}
}
@article{klamkin2024dual,
  title   = {Dual Interior Point Optimization Learning},
  author  = {Klamkin, Michael and Tanneau, Mathieu and Van Hentenryck, Pascal},
  journal = {arXiv preprint arXiv:2402.02596},
  year    = {2024}
}
```
