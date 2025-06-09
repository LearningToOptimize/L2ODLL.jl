module L2ODLL

import Dualization
import JuMP
import LinearAlgebra
import MathOptSetDistances
import ParametricOptInterface
import SparseArrays

const MOI = JuMP.MOI
const MOIB = JuMP.MOIB
const MOIU = JuMP.MOIU
const POI = ParametricOptInterface
const MOSD = MathOptSetDistances

abstract type AbstractDecomposition end  # must have p_ref and y_ref

include("MOI_wrapper.jl")
include("projection.jl")
include("layers/generic.jl")
include("layers/bounded_lp.jl")
include("layers/convex_qp.jl")

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
        elseif decomposition isa BoundDecomposition && _is_plp(model)
            bounded_lp_builder(decomposition, proj_fn, dual_model) # default to completion=:exact
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
end  # module