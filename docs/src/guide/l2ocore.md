# L2OCore interface and design notes

```@meta
CurrentModule = L2ODLL
```

[L2OCore](https://github.com/LearningToOptimize/LearningToOptimize.jl) (in
`lib/L2OCore` of LearningToOptimize.jl) defines a small, method-agnostic pipeline
shared by the Learning-to-Optimize packages: a `ParametricProblem`, methods that
declare which problem classes they `supports`, and the steps
`decompose → generate_data → loss → complete → evaluate`.
L2ODLL implements it as a **dual method**.

| L2OCore function | DLL implementation | output |
|:--|:--|:--|
| `supports` | `LinearConic`, `ConvexQuadratic` | `true` for these classes |
| `decompose` | [`decompose!`](@ref) (auto-detected, or `DLLMethod(decomposition = BoundDecomposition)`) | a `DLLDecomposition` (the decomposition and an index map) |
| `output_dimension` | `sum(y_shape(model))` | proxy outputs per instance (`sum(y_shape) × B` per batch) |
| `complete` | projection + completion; full dual in `dual_layout` order with `JuMP.dual` signs | `Prediction(dual)` |
| `loss` | self-supervised (Theory eq. 7): ``-\text{mean}\,\Phi`` (min) / ``+\text{mean}\,\Phi`` (max) | a scalar to minimize |
| `generate_data` | inputs only; with `optimizer`, also optimal objectives and duals (for gaps) | an `L2ODataset` |
| `evaluate` | the L2OCore evaluators on the completed dual | dual bound and dual infeasibility per instance |

```julia
import L2OCore, L2ODLL, Clarabel
problem = L2OCore.ParametricProblem(model)
pipe = L2OCore.L2OPipeline(problem, L2ODLL.DLLMethod())
data = L2OCore.generate_data(pipe; sampler = L2OCore.BoxSampler(lo, hi), n = 100,
                             optimizer = Clarabel.Optimizer)       # solver optima for gaps
Y = proxy(data.inputs)                                              # sum(y_shape) × 100
ℓ = L2OCore.loss(pipe, data.inputs, Y)                              # minimize this
pred = L2OCore.complete(pipe, data.inputs, Y)
report = L2OCore.evaluate(problem, data.inputs, pred)             # dual bound, dual infeasibility
gap = (data.objective .- report.dual_bound) ./ max.(1, abs.(data.objective))
```

The [economic dispatch example](../examples/economic_dispatch.md) runs this end to end.

## Why a package extension

L2OCore adds no new transitive dependencies (it uses JuMP, MathOptInterface and
standard libraries, all already dependencies of L2ODLL), so weight is not the
concern. It is, however, not registered yet and is installed from a branch. Making
it a hard dependency would make every installation and every CI run of L2ODLL
depend on that branch. As a weak dependency:

- `using L2ODLL` works and is tested without L2OCore;
- `import L2OCore` loads the extension `L2ODLLL2OCoreExt`, which defines the method
  type; [`DLLMethod`](@ref) is its constructor (calling it without L2OCore loaded
  prints a hint);
- once L2OCore is registered, the `[sources]` entry and the CI step for Julia 1.10
  can be dropped; the extension can stay.

## Automatic differentiation

DLL's loss needs only **first-order** derivatives of ``\Phi`` with respect to the
proxy output (then the chain rule through the proxy), so it does not need nested AD.

| backend | through `dual_objective` / `L2OCore.loss` | tested against |
|:--|:--|:--|
| ForwardDiff | yes (default `ad_type`) | central finite differences |
| Zygote | yes | ForwardDiff, to ``10^{-10}`` |
| FiniteDifferences / other DI backends | yes, via `ad_type` | — |
| solver-based layers (`jump_builder`) | no AD; gradients would need the dual of the completion (issue #5) | — |

Consequently any Julia ML framework whose training loop uses Zygote (Flux, Lux)
can train a DLL proxy: the proxy is just a callable mapping parameters to the
`sum(y_shape) × B` output.

## GPU and batching

For the closed-form decompositions, [`decompose!`](@ref) also builds a **batched
layer** ([`batched_layer`](@ref)). It evaluates a whole batch (one instance per
column) with matrix products and broadcasts only, so it

- is used automatically by the matrix forms `dual_objective(model, Y, P)` and
  `dual_objective_and_gradient(model, Y, P)` and by the L2OCore `complete`/`loss`;
- differentiates with Zygote (one pullback for the batch) and with ForwardDiff
  (one pushforward per row of `Y`, independent of the batch size);
- is written for accelerator arrays: no scalar indexing, no mutation.

```julia
using Adapt, CUDA                       # or any GPU array package
layer = Adapt.adapt(CuArray, L2ODLL.densify(batched_layer(model)))
Y, P = CuArray(Y), CuArray(P)           # sum(y_shape) × B, num_parameters × B
bounds = layer(Y, P)                    # CuVector of valid bounds
∇ = Zygote.gradient(Y -> sum(layer(Y, P)), Y)[1]
c = batched_completion(layer, Y, P)     # (y, z, w) as device matrices
```

The test suite runs this path with JLArrays (a reference GPU array type that
forbids scalar indexing) and compares values, Zygote gradients and completions with
the CPU ones. CI has no GPU, so the CUDA snippet above is not run there. Projections onto cones other than scalar sets,
orthants and second-order cones fall back to a per-column CPU projection.
ExaModels and BatchNLPKernels-based evaluation in L2OCore is complementary: it
evaluates the primal problem, while this layer evaluates the dual bound.
