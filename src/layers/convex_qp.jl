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
        JuMP.all_constraints(model; include_variable_in_set_constraints=true),
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
    # Example 3 of Tanneau and Van Hentenryck (2024) needs Q ≻ 0 (Q ≺ 0 for Max):
    # a diagonal term for every variable is necessary but not sufficient.
    return is_strictly_convex_objective(model)
end

"""
    objective_hessian(model::JuMP.Model, x_ref = decision variables) -> Matrix{Float64}

Hessian of the objective with respect to the decision variables `x_ref` (terms
involving parameters are excluded): for `f(x) = ½ xᵀHx + …`, returns `H`.
"""
function objective_hessian(
    model::JuMP.Model,
    x_ref::AbstractVector{JuMP.VariableRef}=filter(!JuMP.is_parameter, JuMP.all_variables(model)),
)
    n = length(x_ref)
    H = zeros(n, n)
    f = JuMP.objective_function(model)
    f isa JuMP.GenericQuadExpr || return H
    idx = _vr_to_idx(x_ref)
    for (coeff, a, b) in JuMP.quad_terms(f)
        (haskey(idx, a) && haskey(idx, b)) || continue  # parameter × variable or parameter × parameter
        i, j = idx[a], idx[b]
        if i == j
            H[i, i] += 2coeff
        else
            H[i, j] += coeff
            H[j, i] += coeff
        end
    end
    return H
end

"""
    is_strictly_convex_objective(model::JuMP.Model) -> Bool

Whether the objective is strictly convex in the decision variables (for `Min`;
strictly concave for `Max`), i.e. its Hessian is positive (negative) definite.
This is the assumption of the `ConvexQP` closed-form completion.
"""
function is_strictly_convex_objective(model::JuMP.Model)
    H = objective_hessian(model)
    isempty(H) && return false
    s = JuMP.objective_sense(model) == MOI.MAX_SENSE ? -1.0 : 1.0
    # `isposdef` accepts exactly singular matrices (zero Cholesky pivot), so test
    # the smallest eigenvalue against a relative tolerance instead.
    λ = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(s .* H))
    return first(λ) > sqrt(eps(Float64)) * max(1.0, maximum(abs, λ))
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
equations for `w` given `y`. With Dualization's sign convention `w` equals the
minimizer of the Lagrangian, `x(y) = Q⁻¹(Aᵀy - c)`, so `w` is the primal optimum
when `y` is dual optimal.
"""
function convex_qp_completion_builder(
    decomposition::ConvexQP, proj_fn, dual_model::JuMP.Model; kwargs...
)
    return _convex_qp_layers(decomposition, proj_fn, dual_model; kwargs...).completion
end

function _convex_qp_layers(
    decomposition::ConvexQP, proj_fn, dual_model::JuMP.Model; backend=nothing
)
    if !isempty(decomposition.x_ref)
        primal = JuMP.owner_model(first(decomposition.x_ref))
        is_strictly_convex_objective(primal) || error(
            "The ConvexQP closed-form completion requires a strictly convex quadratic objective " *
            "(positive definite Hessian for Min, negative definite for Max; Example 3 of " *
            "Tanneau and Van Hentenryck, 2024). Otherwise the completed value is not a valid bound.",
        )
    end
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
        JuMP.list_of_constraint_types(dual_model),
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

    Qz_fn = VecAffExprMatrix(Qz, [flatten_y(y_vars); p_vars]; backend=backend)

    obj_fn = QuadExprMatrix(
        JuMP.objective_function(dual_model), [flatten_y(y_vars); p_vars; z_vars]; backend=backend
    )

    # Dualization writes the stationarity condition as Aᵀy + Q_d z = c, where Q_d is
    # the coefficient matrix of z collected above (Q_d = -Q for an objective ½xᵀQx).
    # Hence z = Q_d⁻¹(c - Aᵀy) = Q⁻¹(Aᵀy - c) = x(y), the Lagrangian minimizer: the
    # closed form of Example 3 in Tanneau and Van Hentenryck (2024).
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
        return (y=y_pred_proj, z=Vector{typeof(z)}(), w=z)
    end
    return (objective=objective, completion=completion_fn)
end

completed_constraints(::ConvexQP) = JuMP.ConstraintRef[]

function get_quadslack(dual_model, decomposition::ConvexQP)
    return Dualization._get_dual_slack_variable.(
        dual_model, filter(x -> has_quadslack(dual_model, x), decomposition.x_ref)
    )
end

function has_quadslack(dual_model, x::JuMP.VariableRef)
    pdm = dual_model.ext[:_Dualization_jl_PrimalDualMap]
    return haskey(pdm.primal_var_in_quad_obj_to_dual_slack_var, JuMP.index(x))
end

function get_x(decomposition::ConvexQP)
    return decomposition.x_ref
end
