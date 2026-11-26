struct BoundDecomposition <: AbstractDecomposition
    p_ref::Vector{JuMP.VariableRef}
    y_ref::Vector{JuMP.ConstraintRef}
    zl_ref::Vector{JuMP.ConstraintRef}
    zu_ref::Vector{JuMP.ConstraintRef}
end

"""
    BoundDecomposition(model::JuMP.Model)

Create a decomposition using `z` for bound constraints and `y` for all other constraints.
"""
function BoundDecomposition(model::JuMP.Model)
    p_ref = filter(JuMP.is_parameter, JuMP.all_variables(model))
    x_ref = filter(!JuMP.is_parameter, JuMP.all_variables(model))
    y_ref = JuMP.all_constraints(model; include_variable_in_set_constraints=false)
    zl_ref = JuMP.LowerBoundRef.(x_ref)
    zu_ref = JuMP.UpperBoundRef.(x_ref)
    return BoundDecomposition(p_ref, y_ref, zl_ref, zu_ref)
end
function can_decompose(model::JuMP.Model, ::Type{BoundDecomposition})
    x_ref = filter(!JuMP.is_parameter, JuMP.all_variables(model))
    length(x_ref) > 0 || return false
    # The closed-form bounded completion is implemented for affine objectives
    # (in the decision variables) only; quadratic objectives go to `ConvexQP`.
    _has_quadratic_decision_terms(JuMP.objective_function(model)) && return false
    if all(JuMP.has_lower_bound, x_ref) && all(JuMP.has_upper_bound, x_ref)
        return true
    end
    return false
end

_has_quadratic_decision_terms(::Any) = false
function _has_quadratic_decision_terms(f::JuMP.GenericQuadExpr)
    return any(((_, a, b),) -> !JuMP.is_parameter(a) && !JuMP.is_parameter(b), JuMP.quad_terms(f))
end

"""
    bounded_builder(decomposition::BoundDecomposition, proj_fn, dual_model; completion=:exact, μ=1.0, backend=nothing)

Closed-form DLL layer `(y_pred, param_value) -> dual objective` for the bounded
decomposition (Example 1 of Tanneau and Van Hentenryck, 2024). See
[`bounded_completion_builder`](@ref) for the completed dual variables.
"""
function bounded_builder(
    decomposition::BoundDecomposition, proj_fn, dual_model::JuMP.Model; kwargs...
)
    return _bounded_layers(decomposition, proj_fn, dual_model; kwargs...).objective
end

"""
    bounded_completion_builder(decomposition::BoundDecomposition, proj_fn, dual_model; completion=:exact, μ=1.0, backend=nothing)

Closed-form completion `(y_pred, param_value) -> (y, z, w)` for the bounded
decomposition: `y` is the projected prediction, `z = [[zl_1], …, [zl_n], [zu_1], …, [zu_n]]`
are the bound duals (in the order of [`completed_constraints`](@ref)) and `w` is empty.
"""
function bounded_completion_builder(
    decomposition::BoundDecomposition, proj_fn, dual_model::JuMP.Model; kwargs...
)
    return _bounded_layers(decomposition, proj_fn, dual_model; kwargs...).completion
end

function _bounded_layers(
    decomposition::BoundDecomposition,
    proj_fn,
    dual_model::JuMP.Model;
    completion=:exact,
    μ=1.0,
    backend=nothing,
    batch_projection=nothing,
)
    if !isempty(
        dual_model.ext[:_Dualization_jl_PrimalDualMap].primal_var_in_quad_obj_to_dual_slack_var
    )
        error(
            "The closed-form BoundDecomposition layer requires an objective that is affine " *
            "in the decision variables. For convex quadratic objectives use `ConvexQP`, or a " *
            "`GenericDecomposition` with an optimizer (see the decomposition detection docs).",
        )
    end
    p_vars = get_p(dual_model, decomposition)
    y_vars = get_y_dual(dual_model, decomposition)
    zl_vars = only.(get_zl(dual_model, decomposition))
    zu_vars = only.(get_zu(dual_model, decomposition))
    types = filter(
        t -> !(t[1] <: JuMP.VariableRef || t[1] <: Vector{JuMP.VariableRef}),
        JuMP.list_of_constraint_types(dual_model),
    )

    zl_plus_zu = [zero(JuMP.AffExpr) for _ in 1:length(zl_vars)]  # not `zeros`: it aliases one mutable AffExpr (#13)
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
                zl_plus_zu[zl_idx] =
                    co.set.value - JuMP.value(vr -> (vr ∈ zl_vars || vr ∈ zu_vars) ? 0 : vr, cr)
                delete!(idx_left, zl_idx)
            end
        else
            error("Unsupported constraint set in bounded_lp_builder: $S")
        end
    end
    @assert isempty(idx_left) "Some zl/zu were not found in the model"

    obj_func = JuMP.objective_function(dual_model)

    completer = if completion == :exact
        ExactBoundedCompletion()
    elseif completion == :log
        l = Float64[JuMP.constraint_object(c).set.lower for c in decomposition.zl_ref]
        u = Float64[JuMP.constraint_object(c).set.upper for c in decomposition.zu_ref]
        # eq. (7) of Klamkin, Tanneau and Van Hentenryck uses v = μ / (u - l): needs l < u, μ > 0
        all(u .> l) || error("The log-barrier completion requires l < u for every variable.")
        μ > 0 || error(
            "The log-barrier completion requires μ > 0; use completion = :exact for μ = 0."
        )
        LogBoundedCompletion(μ, l, u)
    else
        error("Invalid completion type: $completion. Must be :exact or :log.")
    end

    z_fn = VecAffExprMatrix(zl_plus_zu, [flatten_y(y_vars); p_vars]; backend=backend)
    obj_fn = QuadExprMatrix(
        obj_func, [flatten_y(y_vars); p_vars; zl_vars; zu_vars]; backend=backend
    )
    # z^l + z^u = c - Aᵀy (L2ODLL sign convention: z^u ≤ 0), then the closed form
    # of Example 1 in Tanneau and Van Hentenryck (2024) (`:exact`) or eq. (7) of
    # Klamkin, Tanneau and Van Hentenryck (2025) (`:log`).
    complete_fn =
        (y_pred, param_value) -> begin
            y_pred_proj = proj_fn(y_pred)
            zl_plus_zu_val = z_fn([flatten_y(y_pred_proj); param_value])
            zl, zu = complete_zlzu(completer, zl_plus_zu_val)
            return y_pred_proj, zl, zu
        end
    objective = (y_pred, param_value) -> begin
        y_pred_proj, zl, zu = complete_fn(y_pred, param_value)
        obj_fn([flatten_y(y_pred_proj); param_value; zl; zu])
    end
    completion_fn =
        (y_pred, param_value) -> begin
            y_pred_proj, zl, zu = complete_fn(y_pred, param_value)
            return (y=y_pred_proj, z=[[[v] for v in zl]; [[v] for v in zu]], w=zl[1:0])
        end
    batch = if isnothing(batch_projection)
        nothing
    else
        BoundedBatchedLayer(batch_projection, z_fn.A, z_fn.b, obj_fn.Q, obj_fn.c, obj_fn.c0, completer)
    end
    return (objective=objective, completion=completion_fn, batch=batch)
end

function _find_and_return_value(vr, var_lists, values)
    for (vars, val) in zip(var_lists, values)
        idx = findfirst(_vr -> _vr == vr, vars)
        !isnothing(idx) && return val[idx]
    end
    return error("Variable $vr not found in any variable list")
end

abstract type BoundedCompletion end
struct ExactBoundedCompletion <: BoundedCompletion end
function complete_zlzu(::ExactBoundedCompletion, zl_plus_zu)
    return max.(zl_plus_zu, zero(eltype(zl_plus_zu))), -max.(-zl_plus_zu, zero(eltype(zl_plus_zu)))
end

struct LogBoundedCompletion{V,T} <: BoundedCompletion
    μ::T
    l::V
    u::V
end
Adapt.@adapt_structure LogBoundedCompletion
function complete_zlzu(c::LogBoundedCompletion{V,T}, zl_plus_zu) where {V,T}
    v = c.μ ./ (c.u - c.l)
    w = T(1//2) .* zl_plus_zu
    sqrtv2w2 = hypot.(v, w)
    return (
        v .+ w .+ sqrtv2w2,      # broadcasts, so a matrix of instances also works
        .-v .+ w .- sqrtv2w2,
    )
end

function make_completion_model(
    decomposition::BoundDecomposition, dual_model::JuMP.Model; log_barrier=false, conic=true
)
    if log_barrier
        completion_model, (p_ref, y_ref, ref_map) = _make_completion_model(
            decomposition, dual_model
        )
        zl = getindex.(ref_map, get_zl(dual_model, decomposition))
        zu = getindex.(ref_map, get_zu(dual_model, decomposition))
        # TODO: add exponential cone constraints and objective terms
        # TODO: add log nonlinear objective terms
        # note objective sense
        return completion_model, (p_ref, y_ref, ref_map)
    else
        return _make_completion_model(decomposition, dual_model)
    end
end

function completed_constraints(decomposition::BoundDecomposition)
    return JuMP.ConstraintRef[decomposition.zl_ref; decomposition.zu_ref]
end

function get_zl(dual_model, decomposition::BoundDecomposition)
    return only.(Dualization._get_dual_variables.(dual_model, decomposition.zl_ref))
end

function get_zu(dual_model, decomposition::BoundDecomposition)
    return only.(Dualization._get_dual_variables.(dual_model, decomposition.zu_ref))
end
