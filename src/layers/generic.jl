struct GenericDecomposition <: AbstractDecomposition
    p_ref::Vector{JuMP.VariableRef}
    y_ref::Vector{JuMP.ConstraintRef}
    z_ref::Vector{JuMP.ConstraintRef}
end
GenericDecomposition(model::JuMP.Model) = begin
    p_ref = filter(JuMP.is_parameter, JuMP.all_variables(model))
    y_ref = JuMP.all_constraints(model, include_variable_in_set_constraints=false)
    all_cr = JuMP.all_constraints(model, include_variable_in_set_constraints=true)
    z_ref = [cr for cr in all_cr if !(cr in y_ref) && !(typeof(cr.index).parameters[2] <: MOI.Parameter)]
    return GenericDecomposition(p_ref, y_ref, z_ref)
end

function poi_builder(decomposition::AbstractDecomposition, proj_fn::Function, dual_model::JuMP.Model, optimizer)
    completion_model, (p_ref, y_ref, ref_map) = make_completion_model(decomposition, dual_model)
    JuMP.set_optimizer(completion_model, optimizer) # () -> ParametricOptInterface.Optimizer(optimizer())
    completion_model.ext[:🔒] = ReentrantLock()
    # TODO: use DiffOpt to define frule/rrule
    # TODO: handle infeasibility?
    return (y_pred, param_value) -> begin
        lock(completion_model.ext[:🔒])
        try
            JuMP.set_parameter_value.(p_ref, param_value)
            JuMP.set_parameter_value.(reduce(vcat, y_ref), reduce(vcat, proj_fn(y_pred)))

            JuMP.optimize!(completion_model)
            JuMP.assert_is_solved_and_feasible(completion_model)

            JuMP.objective_value(completion_model)
        finally
            unlock(completion_model.ext[:🔒])
        end
    end
end

function make_completion_model(decomposition::AbstractDecomposition, dual_model::JuMP.Model)
    completion_model, ref_map = JuMP.copy_model(dual_model)
    p_ref = getindex.(ref_map, Dualization._get_dual_parameter.(dual_model, decomposition.p_ref))

    y_vars = Dualization._get_dual_variables.(dual_model, decomposition.y_ref)
    y_ref = Vector{JuMP.VariableRef}[]
    for y in y_vars
        push!(y_ref, getindex.(ref_map, y))
    end

    # remove dual cone constraints from y variables
    JuMP.delete.(completion_model, getindex.(ref_map, filter(!isnothing, Dualization._get_dual_constraint.(dual_model, decomposition.y_ref))))

    # mark y and p as parameters (optimizing over z only)
    y_ref_flat = reduce(vcat, y_ref)
    JuMP.@constraint(completion_model, y_ref_flat .∈ MOI.Parameter.(zeros(length(y_ref_flat))))
    JuMP.@constraint(completion_model, p_ref .∈ MOI.Parameter.(zeros(length(p_ref))))
    
    return completion_model, (p_ref, y_ref, ref_map)
end

function make_completion_data(decomposition::AbstractDecomposition, dual_model::JuMP.Model; M=SparseArrays.SparseMatrixCSC{Float64,Int}, V=Vector{Float64}, T=Float64)
    completion_model, (p_ref, y_ref, ref_map) = make_completion_model(decomposition, dual_model)
    return model_to_data(completion_model, M=M, V=V, T=T), (p_ref, y_ref, ref_map)
end