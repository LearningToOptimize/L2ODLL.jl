struct ConvexQP <: AbstractDecomposition
    p_ref::Vector{JuMP.VariableRef}
    y_ref::Vector{JuMP.ConstraintRef}
end
function ConvexQP(model::JuMP.Model)
    p_ref = filter(JuMP.is_parameter, JuMP.all_variables(model))
    y_ref = filter(
        cr -> !(typeof(JuMP.index(cr)).parameters[2] <: MOI.Parameter),
        JuMP.all_constraints(model, include_variable_in_set_constraints=true)
    )
    return ConvexQP(p_ref, y_ref)
end


function convex_qp_builder(decomposition::ConvexQP, proj_fn, dual_model::JuMP.Model)
    p_vars = get_p(dual_model, decomposition)
    y_vars = get_y(dual_model, decomposition)
    z_vars = get_quadslack(dual_model)
    types = filter(
        t -> !(t[1] <: JuMP.VariableRef || t[1] <: Vector{JuMP.VariableRef}),
        JuMP.list_of_constraint_types(dual_model)
    )

    Q = zeros(length(z_vars), length(z_vars))
    Qz = zeros(JuMP.AffExpr, length(z_vars))
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
            throw(ArgumentError("Unsupported constraint set in convex_qp_builder: $S"))
        end
    end
    @assert isempty(idx_left) "Some z were not found in the model"

    obj_func = JuMP.objective_function(dual_model)

    Qinv = inv(Q)
    return (y_pred, param_value) -> begin
        y_pred_proj = proj_fn(y_pred)

        Qz_val = JuMP.value.(vr -> _find_and_return_value(vr,
            [reduce(vcat, y_vars), p_vars],
            [reduce(vcat, y_pred_proj), param_value]),
            Qz
        )

        z = Qinv * Qz_val

        JuMP.value.(vr -> _find_and_return_value(vr,
            [reduce(vcat, y_vars), p_vars, z_vars],
            [reduce(vcat, y_pred_proj), param_value, z]),
            obj_func
        )
    end
end

function get_quadslack(dual_model)
    pdm = dual_model.ext[:_Dualization_jl_PrimalDualMap]
    return [JuMP.VariableRef(dual_model, vi) for vi in collect(values(pdm.primal_var_in_quad_obj_to_dual_slack_var))]
end