module L2ODLL

using Dualization
using JuMP
using LinearAlgebra
using MathOptSetDistances
using ParametricOptInterface

abstract type AbstractDecomposition end  # must have p_ref and y_ref

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
    dual_model = dualize(model, dual_names=DualNames("λ", "x", "θ", "q"), consider_constrained_variables=false)

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

end  # module