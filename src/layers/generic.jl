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

function poi_builder(decomposition::AbstractDecomposition, proj_fn::Function, dual_model::JuMP.Model, optimizer; silent=true)
    completion_model, (p_ref, y_ref, ref_map) = make_completion_model(decomposition, dual_model)
    JuMP.set_optimizer(completion_model, optimizer)
    silent && JuMP.set_silent(completion_model)
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
    return _make_completion_model(decomposition, dual_model)
end
function _make_completion_model(decomposition::AbstractDecomposition, dual_model::JuMP.Model)
    completion_model, ref_map = JuMP.copy_model(dual_model)
    p_ref = getindex.(ref_map, get_p(dual_model, decomposition))
    y_ref = getindex.(ref_map, get_y(dual_model, decomposition))

    # remove dual cone constraints from y variables
    JuMP.delete.(completion_model, getindex.(ref_map, filter(!isnothing, get_y_constraint(dual_model, decomposition))))

    # mark y and p as parameters (optimizing over z only)
    y_ref_flat = reduce(vcat, y_ref)
    JuMP.@constraint(completion_model, y_ref_flat .∈ MOI.Parameter.(zeros(length(y_ref_flat))))
    JuMP.@constraint(completion_model, p_ref .∈ MOI.Parameter.(zeros(length(p_ref))))
    
    return completion_model, (p_ref, y_ref, ref_map)
end

function make_vector_data(decomposition::AbstractDecomposition, dual_model::JuMP.Model; M=SparseArrays.SparseMatrixCSC{Float64,Int}, V=Vector{Float64}, T=Float64)
    completion_model, (p_ref, y_ref, ref_map) = make_completion_model(decomposition, dual_model)
    y_sets = get_y_sets(dual_model, decomposition)
    data = convert(VectorStandardFormData{M,V,T}, model_to_data(completion_model))
    return data, y_sets, (p_ref, y_ref, ref_map)
end

function get_y(dual_model, decomposition::AbstractDecomposition)
    return Dualization._get_dual_variables.(dual_model, decomposition.y_ref)
end

function get_p(dual_model, decomposition::AbstractDecomposition)
    return Dualization._get_dual_parameter.(dual_model, decomposition.p_ref)
end

function get_y_constraint(dual_model, decomposition::AbstractDecomposition)
    return Dualization._get_dual_constraint.(dual_model, decomposition.y_ref)
end
