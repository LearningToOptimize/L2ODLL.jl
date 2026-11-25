struct ConvexQP <: AbstractDecomposition
    p_ref::Vector{JuMP.VariableRef}
    y_ref::Vector{JuMP.ConstraintRef}
    x_ref::Vector{JuMP.VariableRef}
end

"""
    ConvexQP(model::JuMP.Model)

Create a decomposition using `z` for the quadratic slacks and `y` for all constraints.
"""
function ConvexQP(model::JuMP.Model)
    p_ref = filter(JuMP.is_parameter, JuMP.all_variables(model))
    y_ref = filter(
        cr -> !(typeof(JuMP.index(cr)).parameters[2] <: MOI.Parameter),
        JuMP.all_constraints(model, include_variable_in_set_constraints=true)
    )
    x_ref = filter(!JuMP.is_parameter, JuMP.all_variables(model))
    return ConvexQP(p_ref, y_ref, x_ref)
end
function can_decompose(model::JuMP.Model, ::Type{ConvexQP})
    x_ref = JuMP.all_variables(model)
    length(x_ref) > 0 || return false
    obj_func = JuMP.objective_function(model)
    if !(obj_func isa JuMP.QuadExpr)
        return false
    end
    for x_i in filter(!JuMP.is_parameter, x_ref)
        if !(JuMP.UnorderedPair(x_i, x_i) in keys(obj_func.terms))
            return false
        end
    end
    return true
end

"""
    convex_qp_builder(decomposition::ConvexQP, proj_fn, dual_model; backend=nothing)

Closed-form DLL layer `(y_pred, param_value) -> dual objective` for a strictly
convex quadratic objective (Example 3 of Tanneau and Van Hentenryck, 2024). See
[`convex_qp_completion_builder`](@ref) for the completed dual variables.
"""
function convex_qp_builder(decomposition::ConvexQP, proj_fn, dual_model::JuMP.Model; kwargs...)
    return _convex_qp_layers(decomposition, proj_fn, dual_model; kwargs...).objective
end

"""
    convex_qp_completion_builder(decomposition::ConvexQP, proj_fn, dual_model; backend=nothing)

Closed-form completion `(y_pred, param_value) -> (y, z, w)`: `y` is the projected
prediction (all constraints are predicted, so `z` is empty) and `w` are the
quadratic slack variables of the dual, obtained by solving the dual stationarity
equations for `w` given `y`.
"""
function convex_qp_completion_builder(decomposition::ConvexQP, proj_fn, dual_model::JuMP.Model; kwargs...)
    return _convex_qp_layers(decomposition, proj_fn, dual_model; kwargs...).completion
end

function _convex_qp_layers(decomposition::ConvexQP, proj_fn, dual_model::JuMP.Model;
    backend=nothing
    )
    p_vars = get_p(dual_model, decomposition)
    y_vars = get_y_dual(dual_model, decomposition)
    x_vars = get_x(decomposition)
    if !all(x -> has_quadslack(dual_model, x), x_vars)
        @warn "Some primal variables do not have a quadratic objective term, " *
               "so they do not have a quadratic slack variable in the dual." *
               "There may not be enough flexibility in the completion model to guarantee feasibility."
    end
    z_vars = get_quadslack(dual_model, decomposition)
    types = filter(
        t -> !(t[1] <: JuMP.VariableRef || t[1] <: Vector{JuMP.VariableRef}),
        JuMP.list_of_constraint_types(dual_model)
    )

    Q = zeros(length(z_vars), length(z_vars))
    Qz = [zero(JuMP.AffExpr) for _ in 1:length(z_vars)]  # not `zeros`: it aliases one mutable AffExpr (#13)
    idx_left = collect(reverse(1:length(z_vars)))
    for (F, S) in types
        @assert F <: JuMP.GenericAffExpr "Unsupported constraint function in convex_qp_builder: $F with set $S"
        if F <: JuMP.GenericAffExpr && S <: MOI.EqualTo
            constraints = JuMP.all_constraints(dual_model, F, S)
            for cr in constraints
                co = JuMP.constraint_object(cr)
                isnothing(findfirst(z -> z in keys(co.func.terms), z_vars)) && continue
                row_idx = pop!(idx_left)
                for (z_idx, z) in enumerate(z_vars)
                    if z in keys(co.func.terms)
                        Q[row_idx, z_idx] = co.func.terms[z]
                    end
                end
                # from A'y + Qz = c to Qz = c - A'y
                Qz[row_idx] = co.set.value - JuMP.value(vr -> (vr ∈ z_vars) ? 0 : vr, co.func)
            end
        else
            error("Unsupported constraint set in convex_qp_builder: $S")
        end
    end
    @assert isempty(idx_left) "Some z were not found in the model"

    Qinv = inv(Q)

    Qz_fn = VecAffExprMatrix(
        Qz,
        [flatten_y(y_vars); p_vars];
        backend=backend
    )

    obj_fn = QuadExprMatrix(
        JuMP.objective_function(dual_model),
        [flatten_y(y_vars); p_vars; z_vars];
        backend=backend
    )

    # From Aᵀy + Qz = c (Dualization's form of the stationarity condition) to
    # z = Q⁻¹(c - Aᵀy), the closed form of Example 3 in Tanneau and Van Hentenryck (2024).
    complete_fn = (y_pred, param_value) -> begin
        y_pred_proj = proj_fn(y_pred)
        Qz_val = Qz_fn([flatten_y(y_pred_proj); param_value])
        return y_pred_proj, Qinv * Qz_val
    end
    objective = (y_pred, param_value) -> begin
        y_pred_proj, z = complete_fn(y_pred, param_value)
        obj_fn([flatten_y(y_pred_proj); param_value; z])
    end
    completion_fn = (y_pred, param_value) -> begin
        y_pred_proj, z = complete_fn(y_pred, param_value)
        return (y = y_pred_proj, z = Vector{typeof(z)}(), w = z)
    end
    return (objective = objective, completion = completion_fn)
end

completed_constraints(::ConvexQP) = JuMP.ConstraintRef[]

function get_quadslack(dual_model, decomposition::ConvexQP)
    return Dualization._get_dual_slack_variable.(dual_model,
        filter(x -> has_quadslack(dual_model, x), decomposition.x_ref)
    )
end

function has_quadslack(dual_model, x::JuMP.VariableRef)
    pdm = dual_model.ext[:_Dualization_jl_PrimalDualMap]
    return haskey(pdm.primal_var_in_quad_obj_to_dual_slack_var, JuMP.index(x))
end

function get_x(decomposition::ConvexQP)
    return decomposition.x_ref
end
