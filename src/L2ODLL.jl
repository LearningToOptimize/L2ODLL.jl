module L2ODLL

import Adapt
import Dualization
import JuMP
import LinearAlgebra
import MathOptSetDistances
import SparseArrays
import DifferentiationInterface
import ForwardDiff

const MOI = JuMP.MOI
const MOIB = JuMP.MOIB
const MOIU = JuMP.MOIU
const MOSD = MathOptSetDistances
const DI = DifferentiationInterface
const ADTypes = DI.ADTypes

abstract type AbstractDecomposition end  # must have p_ref and y_ref and implement can_decompose

include("batch.jl")
include("layers/generic.jl")
include("layers/bounded.jl")
include("layers/convex_qp.jl")
include("projection.jl")

"""
    DLLCache

Everything [`decompose!`](@ref) builds for a model: the projection `y_proj`, the
DLL layer `dll_layer(y_pred, param_value) -> dual objective`, the
`completion(y_pred, param_value) -> (y, z, w)` (or `nothing` when a custom
`dll_layer_builder` is given without a `completion_builder`), the dual model and
the decomposition.
"""
struct DLLCache
    y_proj::Function
    dll_layer::Function
    dual_model::JuMP.Model
    decomposition::AbstractDecomposition
    completion::Union{Nothing,Function}
end
DLLCache(y_proj, dll_layer, dual_model, decomposition) =
    DLLCache(y_proj, dll_layer, dual_model, decomposition, nothing)

const _DECOMPOSITIONS = [ # in order of preference for auto-detection
    BoundDecomposition,
    ConvexQP,
    GenericDecomposition
]

"""
    DLLMethod(; decomposition = nothing, optimizer = nothing)

Dual Lagrangian Learning as a method of the LearningToOptimize `L2OCore`
interface (an `L2OCore.AbstractDualMethod` with guarantee
`L2OCore.ValidDualBound`). Available after `import L2OCore` (package extension).

  - `decomposition`: `nothing` to auto-detect (as in [`decompose!`](@ref)), or a
    function `model -> AbstractDecomposition`, e.g. `L2ODLL.BoundDecomposition`.
  - `optimizer`: used when the decomposition has no closed-form completion.

```julia
import L2OCore, L2ODLL
pipe = L2OCore.L2OPipeline(L2OCore.ParametricProblem(model), L2ODLL.DLLMethod())
pred = L2OCore.complete(pipe, P, Y)        # completed duals, dual_layout order
L2OCore.check(L2OCore.ValidDualBound(), pipe.problem, P, pred; reference = opt)
```
"""
function DLLMethod end

"""
    decompose!(model::JuMP.Model)

Detect the best decomposition and build the DLL functions.
"""
function decompose!(model::JuMP.Model)
    for decomp in _DECOMPOSITIONS
        if can_decompose(model, decomp)
            return decompose!(model, decomp(model))
        end
    end
    error("Could not detect decomposition that guarantees completion feasibility.")
end

"""
    decompose!(model::JuMP.Model, decomposition::AbstractDecomposition)

Build the DLL functions using the given decomposition.
"""
function decompose!(model::JuMP.Model, decomposition::AbstractDecomposition;
    optimizer=nothing, proj_fn=nothing, dll_layer_builder=nothing, completion_builder=nothing
)
    return build_cache(model, decomposition; optimizer, proj_fn, dll_layer_builder, completion_builder)
end

"""
    dual_objective(model::JuMP.Model, y_predicted, param_value)

Evaluate the dual objective function (projection and completion).
"""
function dual_objective(model::JuMP.Model, y_predicted, param_value)
    cache = get_cache(model)
    @assert length.(y_predicted) == L2ODLL.y_shape(cache)
    return cache.dll_layer(y_predicted, param_value)
end

"""
    dual_objective_gradient(model::JuMP.Model, y_predicted, param_value; ad_type::ADTypes.AbstractADType=DI.AutoForwardDiff())

Evaluate the gradient of the dual objective function with respect to the predicted dual variables.
    This includes both the projection and the completion steps.
"""
function dual_objective_gradient(model::JuMP.Model, y_predicted, param_value; ad_type::ADTypes.AbstractADType=DI.AutoForwardDiff())
    cache = get_cache(model)
    y_shape = L2ODLL.y_shape(cache)
    @assert length.(y_predicted) == y_shape
    dobj_wrt_y = DI.gradient(
        (y,p) -> dual_objective(model, L2ODLL.unflatten_y(y, y_shape), p),
        ad_type,
        L2ODLL.flatten_y(y_predicted), DI.Constant(param_value)
    )
    return L2ODLL.unflatten_y(dobj_wrt_y, y_shape)
end

"""
    dual_objective_and_gradient(model::JuMP.Model, y_predicted, param_value; ad_type=DI.AutoForwardDiff())

Return `(dual_objective(model, y_predicted, param_value), dual_objective_gradient(model, y_predicted, param_value))`
computed in one differentiation pass.
"""
function dual_objective_and_gradient(model::JuMP.Model, y_predicted, param_value; ad_type::ADTypes.AbstractADType=DI.AutoForwardDiff())
    cache = get_cache(model)
    y_shape = L2ODLL.y_shape(cache)
    @assert length.(y_predicted) == y_shape
    val, g = DI.value_and_gradient(
        (y,p) -> dual_objective(model, L2ODLL.unflatten_y(y, y_shape), p),
        ad_type,
        L2ODLL.flatten_y(y_predicted), DI.Constant(param_value)
    )
    return val, L2ODLL.unflatten_y(g, y_shape)
end

# ---------------------------------------------------------------------------
# Batched (matrix) forms: the batch is the last dimension.
#   Y :: (sum(y_shape(model)) × B) flattened predictions, P :: (num_parameters × B)
# ---------------------------------------------------------------------------

function _check_batch(model::JuMP.Model, Y::AbstractMatrix, P::AbstractMatrix)
    shape = y_shape(model)
    size(Y, 1) == (isempty(shape) ? 0 : sum(shape)) ||
        throw(DimensionMismatch("Y has $(size(Y, 1)) rows, expected sum(y_shape(model)) = $((isempty(shape) ? 0 : sum(shape)))"))
    size(Y, 2) == size(P, 2) ||
        throw(DimensionMismatch("Y has $(size(Y, 2)) columns but P has $(size(P, 2))"))
    return shape
end

"""
    dual_objective(model::JuMP.Model, Y::AbstractMatrix, P::AbstractMatrix) -> Vector

Batched [`dual_objective`](@ref): column `k` of `Y` is a flattened prediction
(see [`flatten_y`](@ref)) and column `k` of `P` the matching parameter values.
"""
function dual_objective(model::JuMP.Model, Y::AbstractMatrix, P::AbstractMatrix)
    shape = _check_batch(model, Y, P)
    return [dual_objective(model, unflatten_y(collect(Y[:, k]), shape), collect(P[:, k])) for k in axes(Y, 2)]
end

"""
    dual_objective_gradient(model::JuMP.Model, Y::AbstractMatrix, P::AbstractMatrix; ad_type=DI.AutoForwardDiff()) -> Matrix

Batched [`dual_objective_gradient`](@ref): column `k` of the result is the
flattened gradient for column `k` of `Y`.
"""
function dual_objective_gradient(model::JuMP.Model, Y::AbstractMatrix, P::AbstractMatrix; ad_type::ADTypes.AbstractADType=DI.AutoForwardDiff())
    return last(dual_objective_and_gradient(model, Y, P; ad_type))
end

"""
    dual_objective_and_gradient(model::JuMP.Model, Y::AbstractMatrix, P::AbstractMatrix; ad_type=DI.AutoForwardDiff()) -> (Vector, Matrix)

Batched [`dual_objective_and_gradient`](@ref).
"""
function dual_objective_and_gradient(model::JuMP.Model, Y::AbstractMatrix, P::AbstractMatrix; ad_type::ADTypes.AbstractADType=DI.AutoForwardDiff())
    shape = _check_batch(model, Y, P)
    T = float(eltype(Y))
    vals = Vector{T}(undef, size(Y, 2))
    G = similar(Y, T)
    for k in axes(Y, 2)
        v, g = dual_objective_and_gradient(model, unflatten_y(collect(Y[:, k]), shape), collect(P[:, k]); ad_type)
        vals[k] = v
        G[:, k] .= flatten_y(g)
    end
    return vals, G
end

"""
    build_cache(model::JuMP.Model, decomposition::AbstractDecomposition;
        optimizer=nothing, proj_fn=nothing, dll_layer_builder=nothing, completion_builder=nothing
    )

Build the DLLCache for the given model and decomposition.
    In this lower-level function (compared to `decompose!`), users can set
    custom projection functions via `proj_fn`, custom DLL layer builders
    via `dll_layer_builder` and custom completion builders via
    `completion_builder` (same signature, returning `(y_pred, p) -> (y, z, w)`).
"""
function build_cache(model::JuMP.Model, decomposition::AbstractDecomposition;
    optimizer=nothing, proj_fn=nothing, dll_layer_builder=nothing, completion_builder=nothing
)
    dual_model = Dualization.dualize(model, consider_constrained_variables=false)

    proj_fn = !isnothing(proj_fn) ? proj_fn : make_proj_fn(decomposition, dual_model)

    layers = if !isnothing(dll_layer_builder)
            nothing  # custom layer: do not build (possibly unsupported) defaults
        elseif decomposition isa BoundDecomposition
            _bounded_layers(decomposition, proj_fn, dual_model)
        elseif decomposition isa ConvexQP
            _convex_qp_layers(decomposition, proj_fn, dual_model)
        elseif !isnothing(optimizer)
            _jump_layers(decomposition, proj_fn, dual_model, optimizer)
        else
            nothing
        end

    dll_layer = if !isnothing(dll_layer_builder)
            dll_layer_builder(decomposition, proj_fn, dual_model)
        elseif !isnothing(layers)
            layers.objective
        else
            error("A $(nameof(typeof(decomposition))) has no closed-form completion; pass `optimizer` or `dll_layer_builder`.")
        end

    completion = if !isnothing(completion_builder)
            completion_builder(decomposition, proj_fn, dual_model)
        elseif !isnothing(layers)
            layers.completion
        else
            nothing  # a custom layer without a matching completion
        end

    cache = DLLCache(proj_fn, dll_layer, dual_model, decomposition, completion)
    model.ext[:_L2ODLL_cache] = cache
    return cache
end

"""
    get_cache(model::JuMP.Model)

Get the DLLCache for the model. Must have called `decompose!` first.
"""
function get_cache(model::JuMP.Model)
    if !haskey(model.ext, :_L2ODLL_cache)
        error("No decomposition found. Please run L2ODLL.decompose! first.")
    end
    return model.ext[:_L2ODLL_cache]
end

"""
    dual_completion(model::JuMP.Model, y_predicted, param_value) -> (y, z, w)

Project the predicted `y` onto its dual cone and complete the remaining dual
variables, returning a `NamedTuple`:

  - `y`: projected predictions, one vector per constraint in [`get_y`](@ref);
  - `z`: completed dual variables, one vector per constraint in [`get_z`](@ref);
  - `w`: quadratic slack variables of the dual (empty for affine objectives).

Signs follow `JuMP.dual`. **Guarantee** (Theorem 1 of Tanneau and Van Hentenryck,
2024): when the decomposition's completion is feasible, `(y, z, w)` is dual
feasible, so [`dual_objective`](@ref) is a valid bound on the optimal value.
"""
function dual_completion(model::JuMP.Model, y_predicted, param_value)
    cache = get_cache(model)
    @assert length.(y_predicted) == L2ODLL.y_shape(cache)
    isnothing(cache.completion) && error(
        "No completion available: the cache was built with a custom `dll_layer_builder`; " *
        "also pass `completion_builder` to `decompose!`.",
    )
    return cache.completion(y_predicted, param_value)
end

"""
    get_z(model::JuMP.Model)

Get the primal constraints whose dual variables (`z`) are completed rather than predicted.
"""
function get_z(model::JuMP.Model)
    return completed_constraints(get_cache(model).decomposition)
end

"""
    make_completion_model(cache::DLLCache)

    Create a JuMP model for the dual completion step.
"""
function make_completion_model(cache::DLLCache)
    return make_completion_model(cache.decomposition, cache.dual_model)
end

"""
    get_y(model::JuMP.Model)

Get the primal constraints corresponding to the `y` variables in the decomposition.
"""
function get_y(model::JuMP.Model)
    return get_cache(model).decomposition.y_ref
end

"""
    get_y_dual(model::JuMP.Model)

Get the dual variables corresponding to the `y` variables in the decomposition.
    These are VariableRefs belonging to the dual model, not the passed-in `model`.
"""
function get_y_dual(model::JuMP.Model)
    return get_y_dual(get_cache(model))
end
function get_y_dual(cache::DLLCache)
    return get_y_dual(cache.dual_model, cache.decomposition)
end

"""
    y_shape(model::JuMP.Model)

Get the shape of the `y` variables in the decomposition.
    This is a Vector{Int} where each entry is the number of dual variables for that constraint.
"""
function y_shape(model::JuMP.Model)
    return y_shape(get_cache(model))
end
function y_shape(cache::DLLCache)
    return length.(get_y_dual(cache.dual_model, cache.decomposition))
end
function y_shape(dual_model::JuMP.Model, decomposition::AbstractDecomposition)
    return length.(get_y_dual(dual_model, decomposition))
end

"""
    flatten_y(y)

Flatten a vector of `y` variables into a single vector, i.e. Vector{Vector{Float64}} -> Vector{Float64}.
"""
function flatten_y(y)
    return reduce(vcat, y)
end

"""
    unflatten_y(y::Vector{T}, y_shape::Vector{Int}) where T

Unflatten a vector of flattened `y` variables into a vector of vectors, i.e. Vector{Float64} -> Vector{Vector{Float64}}.
"""
function unflatten_y(y::Vector{T}, y_shape::Vector{Int}) where T
    # Non-mutating so that reverse-mode AD (e.g. Zygote) can differentiate through it.
    # (`cumsum` is avoided as well: its reverse rule fails on integer arguments.)
    return map(eachindex(y_shape)) do i
        stop = sum(view(y_shape, 1:i))
        y[(stop - y_shape[i] + 1):stop]
    end
end

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
        if exc.f === DLLMethod
            print(io, "\n`L2ODLL.DLLMethod` is defined by a package extension; run `import L2OCore` first.")
        end
    end
    return nothing
end

end  # module