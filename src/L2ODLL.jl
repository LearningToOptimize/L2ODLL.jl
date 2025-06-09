module L2ODLL

import Dualization
import JuMP
import LinearAlgebra
import MathOptSetDistances
import SparseArrays

const MOI = JuMP.MOI
const MOIB = JuMP.MOIB
const MOIU = JuMP.MOIU
const MOSD = MathOptSetDistances

abstract type AbstractDecomposition end  # must have p_ref and y_ref

include("MOI_wrapper.jl")
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
            poi_builder(decomposition, proj_fn, dual_model, optimizer)
        end

    return DLLCache(proj_fn, dll_layer, dual_model, decomposition)
end

function make_completion_model(cache::DLLCache)
    return make_completion_model(cache.decomposition, cache.dual_model)
end
function make_completion_data(cache::DLLCache; M=SparseArrays.SparseMatrixCSC{Float64,Int}, V=Vector{Float64}, T=Float64)
    completion_model, (p_ref, y_ref, ref_map) = make_completion_model(cache)
    return model_to_data(completion_model, M=M, V=V, T=T), (p_ref, y_ref, ref_map)
end
function get_y(cache::DLLCache)
    return get_y(cache.dual_model, cache.decomposition)
end

function y_shape(cache::DLLCache)
    return length.(get_y(cache.dual_model, cache.decomposition))
end

function flatten_y(y::AbstractVector)
    return reduce(vcat, y)
end

function unflatten_y(y::AbstractVector, y_shape::AbstractVector{Int})
    return [y[start_idx:start_idx + shape - 1] for (start_idx, shape) in enumerate(y_shape)]
end

end  # module