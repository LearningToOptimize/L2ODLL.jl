struct GenericDecomposition <: AbstractDecomposition
    p_ref::Vector{JuMP.VariableRef}
    y_ref::Vector{JuMP.ConstraintRef}
    z_ref::Vector{JuMP.ConstraintRef}
end
GenericDecomposition(model::JuMP.Model) = begin
    p_ref = filter(is_parameter, JuMP.all_variables(model))
    y_ref = JuMP.all_constraints(model, include_variable_in_set_constraints=false)
    all_cr = JuMP.all_constraints(model, include_variable_in_set_constraints=true)
    z_ref = [cr for cr in all_cr if !(cr in y_ref) && !(typeof(cr.index).parameters[2] <: MOI.Parameter)]
    return GenericDecomposition(p_ref, y_ref, z_ref)
end

function poi_builder(decomposition::AbstractDecomposition, proj_fn::Function, dual_model::JuMP.Model, optimizer)
    completion_model, ref_map = JuMP.copy_model(dual_model)
    p_ref = getindex.(ref_map, Dualization._get_dual_parameter.(dual_model, decomposition.p_ref))
    @constraint(completion_model, p_ref .∈ MOI.Parameter.(zeros(length(p_ref))))

    y_vars = Dualization._get_dual_variables.(dual_model, decomposition.y_ref)
    y_ref = getindex.(ref_map, reduce(vcat, y_vars))

    delete_lower_bound.(filter(has_lower_bound, y_ref))
    delete_upper_bound.(filter(has_upper_bound, y_ref))
    @constraint(completion_model, y_ref .∈ MOI.Parameter.(zeros(length(y_ref))))

    set_optimizer(completion_model, () -> ParametricOptInterface.Optimizer(optimizer()))
    
    completion_model.ext[:🔒] = ReentrantLock()
    # TODO: use DiffOpt to define frule/rrule
    return (y_pred, param_value) -> begin
        lock(completion_model.ext[:🔒])
        try
            JuMP.set_parameter_value.(p_ref, param_value)
            JuMP.set_parameter_value.(reduce(vcat, y_ref), reduce(vcat, proj_fn(y_pred)))

            JuMP.optimize!(completion_model)
            JuMP.assert_is_solved_and_feasible(completion_model)

            JuMP.value.(JuMP.objective_function(completion_model))
        finally
            unlock(completion_model.ext[:🔒])
        end
    end
end
