module L2ODLL

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

include("layers/generic.jl")
include("layers/bounded.jl")
include("layers/convex_qp.jl")
include("projection.jl")

struct DLLCache
    y_proj::Function
    dll_layer::Function
    dual_model::JuMP.Model
    decomposition::AbstractDecomposition
end

const _DECOMPOSITIONS = [ # in order of preference for auto-detection
    BoundDecomposition,
    ConvexQP,
    GenericDecomposition
]

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
    optimizer=nothing, proj_fn=nothing, dll_layer_builder=nothing
)
    return build_cache(model, decomposition; optimizer, proj_fn, dll_layer_builder)
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
    build_cache(model::JuMP.Model, decomposition::AbstractDecomposition;
        optimizer=nothing, proj_fn=nothing, dll_layer_builder=nothing
    )

Build the DLLCache for the given model and decomposition.
    In this lower-level function (compared to `decompose!`), users can set
    custom projection functions via `proj_fn` and custom DLL layer builders
    via `dll_layer_builder`.
"""
function build_cache(model::JuMP.Model, decomposition::AbstractDecomposition;
    optimizer=nothing, proj_fn=nothing, dll_layer_builder=nothing
)
    dual_model = Dualization.dualize(model, consider_constrained_variables=false)

    proj_fn = !isnothing(proj_fn) ? proj_fn : make_proj_fn(decomposition, dual_model)

    dll_layer = if !isnothing(dll_layer_builder)
            dll_layer_builder(decomposition, proj_fn, dual_model)
        elseif decomposition isa BoundDecomposition
            bounded_builder(decomposition, proj_fn, dual_model)
        elseif decomposition isa ConvexQP
            convex_qp_builder(decomposition, proj_fn, dual_model)
        else
            jump_builder(decomposition, proj_fn, dual_model, optimizer)
        end

    cache = DLLCache(proj_fn, dll_layer, dual_model, decomposition)
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

"""
    flatten_y(y::AbstractVector)

Flatten a vector of `y` variables into a single vector, i.e. Vector{Vector{Float64}} -> Vector{Float64}.
"""
function flatten_y(y::AbstractVector)
    return reduce(vcat, y)
end

"""
    unflatten_y(y::AbstractVector, y_shape::AbstractVector{Int})

Unflatten a vector of flattened `y` variables into a vector of vectors, i.e. Vector{Float64} -> Vector{Vector{Float64}}.
"""
function unflatten_y(y::AbstractVector, y_shape::AbstractVector{Int})
    return [y[start_idx:start_idx + shape - 1] for (start_idx, shape) in enumerate(y_shape)]
end

end  # module