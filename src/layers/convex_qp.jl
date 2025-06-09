struct ConvexQP{M} <: AbstractDecomposition
    p_ref::Vector{JuMP.VariableRef}
    y_ref::Vector{JuMP.ConstraintRef}
    Qinv::M
    x_ref::Vector{JuMP.VariableRef}
end

function ConvexQP(model::JuMP.Model)
    p_ref = filter(is_parameter, JuMP.all_variables(model))
    y_ref = filter(
        cr -> !(typeof(index(cr)).parameters[2] <: MOI.Parameter),
        JuMP.all_constraints(model, include_variable_in_set_constraints=true)
    )
    Qinv, map_to_idx = _compute_quadratic_objective_inverse(model, p_ref)
    return ConvexQP{typeof(Qinv)}(p_ref, y_ref, Qinv, map_to_idx)
end

function _compute_quadratic_objective_inverse(model::JuMP.Model, ignore_vars::Vector{JuMP.VariableRef})
    obj_func = JuMP.objective_function(model)
    if obj_func isa JuMP.QuadExpr
        return _compute_quadratic_objective_inverse(obj_func, ignore_vars)
    else
        return nothing
    end
end

function _compute_quadratic_objective_inverse(
    func::MOI.ScalarQuadraticFunction{T},
    ignore_vars::Vector{MOI.VariableIndex}
) where {T}
    # based on MathOptInterface.jl/src/Bridges/Constraint/bridges/QuadtoSOCBridge.jl
    # TODO: use process_objective from MOI_wrapper.jl
    Q, index_to_variable_map = MOIB.Constraint._matrix_from_quadratic_terms(func.quadratic_terms)
    # figure out which columns/rows to drop
    drop_idx = []
    for (i, v) in enumerate(index_to_variable_map)
        if v in ignore_vars
            push!(drop_idx, i)
        end
    end
    Q = Q[setdiff(1:size(Q, 1), drop_idx), setdiff(1:size(Q, 2), drop_idx)]
    Qinv = inv(Matrix{eltype(Q)}(Q))
    return Qinv, index_to_variable_map
end

function _compute_quadratic_objective_inverse(func::JuMP.QuadExpr, ignore_vars::Vector{JuMP.VariableRef})
    Finv, index_to_variable_map = _compute_quadratic_objective_inverse(moi_function(func), index.(ignore_vars))
    return Finv, [JuMP.VariableRef(JuMP.owner_model(func), vi) for vi in index_to_variable_map]
end

function convex_qp_builder(decomposition::ConvexQP, proj_fn, dual_model::JuMP.Model)
    p_vars = Dualization._get_dual_parameter.(dual_model, decomposition.p_ref)
    y_vars = Dualization._get_dual_variables.(dual_model, decomposition.y_ref)
    z_vars = Dualization._get_dual_slack_variable.(dual_model, decomposition.x_ref)
    types = filter(t -> !(t[1] <: JuMP.VariableRef || t[1] <: Vector{JuMP.VariableRef}), JuMP.list_of_constraint_types(dual_model))

    Fz = zeros(JuMP.AffExpr, length(z_vars))
    idx_left = Set(1:length(z_vars))
    for (F, S) in types
        @assert F <: JuMP.GenericAffExpr "Unsupported constraint function in convex_qp_builder: $F with set $S"
        if F <: JuMP.GenericAffExpr && S <: MOI.EqualTo
            constraints = JuMP.all_constraints(dual_model, F, S)
            for cr in constraints
                co = JuMP.constraint_object(cr)
                z_idx = findall(z -> z in keys(co.func.terms), z_vars)
                length(z_idx) == 0 && continue # if primal var doesn't have quadratic term
                @assert length(z_idx) == 1 "Multiple z variables found in constraint $cr"
                z_idx = only(z_idx)

                # from A'y + F'z = c to F'z = c - A'y
                Fz[z_idx] = co.set.value - JuMP.value(vr -> (vr ∈ z_vars) ? 0 : vr, co.func)
                delete!(idx_left, z_idx)
            end
        else
            throw(ArgumentError("Unsupported constraint set in convex_qp_builder: $S"))
        end
    end
    @assert isempty(idx_left) "Some z were not found in the model"

    obj_func = JuMP.objective_function(dual_model)

    Finv = decomposition.Qinv

    return (y_pred, param_value) -> begin
        y_pred_proj = proj_fn(y_pred)

        Fz_val = JuMP.value.(vr -> _find_and_return_value(vr,
            [reduce(vcat, y_vars), p_vars],
            [reduce(vcat, y_pred_proj), param_value]),
            Fz
        )

        z = Finv * Fz_val

        JuMP.value.(vr -> _find_and_return_value(vr,
            [reduce(vcat, y_vars), p_vars, z_vars],
            [reduce(vcat, y_pred_proj), param_value, z]),
            obj_func
        )
    end
end