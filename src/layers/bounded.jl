struct BoundDecomposition <: AbstractDecomposition
    p_ref::Vector{JuMP.VariableRef}
    y_ref::Vector{JuMP.ConstraintRef}
    zl_ref::Vector{JuMP.ConstraintRef}
    zu_ref::Vector{JuMP.ConstraintRef}
end
function BoundDecomposition(model::JuMP.Model)
    p_ref = filter(JuMP.is_parameter, JuMP.all_variables(model))
    x_ref = filter(!JuMP.is_parameter, JuMP.all_variables(model))
    y_ref = JuMP.all_constraints(model, include_variable_in_set_constraints=false)
    zl_ref = JuMP.LowerBoundRef.(x_ref)
    zu_ref = JuMP.UpperBoundRef.(x_ref)
    return BoundDecomposition(p_ref, y_ref, zl_ref, zu_ref)
end

function bounded_builder(decomposition::BoundDecomposition, proj_fn, dual_model::JuMP.Model; completion=:exact, μ=1.0)
    p_vars = Dualization._get_dual_parameter.(dual_model, decomposition.p_ref)
    y_vars = Dualization._get_dual_variables.(dual_model, decomposition.y_ref)
    zl_vars = only.(Dualization._get_dual_variables.(dual_model, decomposition.zl_ref))
    zu_vars = only.(Dualization._get_dual_variables.(dual_model, decomposition.zu_ref))
    types = filter(t -> !(t[1] <: JuMP.VariableRef || t[1] <: Vector{JuMP.VariableRef}), JuMP.list_of_constraint_types(dual_model))

    zl_plus_zu = zeros(JuMP.AffExpr, length(zl_vars))
    idx_left = Set(1:length(zl_vars))
    for (F, S) in types
        @assert F <: JuMP.GenericAffExpr "Unsupported constraint function in bounded_lp_builder: $F"
        if S <: MOI.EqualTo
            constraints = JuMP.all_constraints(dual_model, F, S)
            for cr in constraints
                # figure out which index of zl/zu this constraint is
                co = JuMP.constraint_object(cr)
                zl_idx = findfirst(zl -> zl in keys(co.func.terms), zl_vars)
                zu_idx = findfirst(zu -> zu in keys(co.func.terms), zu_vars)
                # should be due to BoundDecomposition constructor
                @assert !isnothing(zl_idx) && !isnothing(zu_idx)
                @assert zl_idx == zu_idx
                # should be true with how Dualization.jl is used
                @assert co.func.terms[zl_vars[zl_idx]] == 1 && co.func.terms[zu_vars[zu_idx]] == 1

                # from A'y + zl + zu = c to zl + zu = c - A'y
                zl_plus_zu[zl_idx] = co.set.value - JuMP.value(vr -> (vr ∈ zl_vars || vr ∈ zu_vars) ? 0 : vr, cr)
                delete!(idx_left, zl_idx)
            end
        else
            throw(ArgumentError("Unsupported constraint set in bounded_lp_builder: $S"))
        end
    end
    @assert isempty(idx_left) "Some zl/zu were not found in the model"

    obj_func = JuMP.objective_function(dual_model)

    completer = if completion == :exact
        ExactBoundedLPCompletion()
    elseif completion == :log
        l = getfield.(getfield.(JuMP.constraint_object.(decomposition.zl_ref), :set), :lower)
        u = getfield.(getfield.(JuMP.constraint_object.(decomposition.zu_ref), :set), :upper)
        LogBoundedLPCompletion(μ, l, u)
    else
        throw(ArgumentError("Invalid completion type: $completion. Must be :exact or :log."))
    end

    return (y_pred, param_value) -> begin
        y_pred_proj = proj_fn(y_pred)

        zl_plus_zu_val = JuMP.value.(vr -> _find_and_return_value(vr,
            [reduce(vcat, y_vars), p_vars],
            [reduce(vcat, y_pred_proj), param_value]),
            zl_plus_zu
        )

        zl, zu = complete_zlzu(completer, zl_plus_zu_val)

        JuMP.value.(vr -> _find_and_return_value(vr, 
            [reduce(vcat, y_vars), p_vars, zl_vars, zu_vars],
            [reduce(vcat, y_pred_proj), param_value, zl, zu]), 
            obj_func
        )
    end
end

function _find_and_return_value(vr, var_lists, values)
    for (vars, val) in zip(var_lists, values)
        idx = findfirst(_vr -> JuMP.index(_vr) == JuMP.index(vr), vars)
        !isnothing(idx) && return val[idx]
    end
    throw(ArgumentError("Variable $vr not found in any variable list"))
end


abstract type AbstractCompletion end
abstract type BoundedLPCompletion <: AbstractCompletion end
struct ExactBoundedLPCompletion <: BoundedLPCompletion end

function complete_zlzu(
    ::ExactBoundedLPCompletion,
    z
)
    return max.(z, zero(eltype(z))), -max.(-z, zero(eltype(z)))
end


struct LogBoundedLPCompletion{T<:Real} <: BoundedLPCompletion
    μ::T
    l::AbstractVector{T}
    u::AbstractVector{T}
end

function complete_zlzu(
    c::LogBoundedLPCompletion,
    z
)
    v = c.μ ./ (c.u - c.l)
    w = eltype(z)(1//2) .* z
    sqrtv2w2 = hypot.(v, w)
    return (
        v + w + sqrtv2w2,
        -v - w + sqrtv2w2,
    )
end