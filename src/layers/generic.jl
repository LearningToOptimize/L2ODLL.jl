struct GenericDecomposition <: AbstractDecomposition
    p_ref::Vector{JuMP.VariableRef}
    y_ref::Vector{JuMP.ConstraintRef}
    z_ref::Vector{JuMP.ConstraintRef}
end

"""
    GenericDecomposition(model::JuMP.Model, y_ref::AbstractVector{<:JuMP.ConstraintRef})

Create a decomposition using `z` for all constraints except `y_ref`.
"""
GenericDecomposition(model::JuMP.Model, y_ref::AbstractVector{<:JuMP.ConstraintRef}) = begin
    p_ref = filter(JuMP.is_parameter, JuMP.all_variables(model))
    all_cr = JuMP.all_constraints(model; include_variable_in_set_constraints=true)
    z_ref = JuMP.ConstraintRef[
        cr for cr in all_cr if !(cr in y_ref) && !(typeof(cr.index).parameters[2] <: MOI.Parameter)
    ]
    return GenericDecomposition(p_ref, convert(Vector{JuMP.ConstraintRef}, y_ref), z_ref)
end

function can_decompose(::JuMP.Model, ::Type{GenericDecomposition})
    return false  # GenericDecomposition needs a manual constructor
end

"""
    jump_builder(decomposition::AbstractDecomposition, proj_fn::Function, dual_model::JuMP.Model, optimizer; silent=true)

Build the DLL layer `(y_pred, param_value) -> dual objective` by solving the
completion model (problem (9) of Tanneau and Van Hentenryck, 2024) with JuMP.
"""
function jump_builder(
    decomposition::AbstractDecomposition,
    proj_fn::Function,
    dual_model::JuMP.Model,
    optimizer;
    silent=true,
)
    return _jump_layers(decomposition, proj_fn, dual_model, optimizer; silent).objective
end

"""
    jump_completion_builder(decomposition::AbstractDecomposition, proj_fn::Function, dual_model::JuMP.Model, optimizer; silent=true)

Build the completion `(y_pred, param_value) -> (y, z, w)` by solving the
completion model with JuMP; `z` follows [`completed_constraints`](@ref) and `w`
are the quadratic slack variables of the dual (if any).
"""
function jump_completion_builder(
    decomposition::AbstractDecomposition,
    proj_fn::Function,
    dual_model::JuMP.Model,
    optimizer;
    silent=true,
)
    return _jump_layers(decomposition, proj_fn, dual_model, optimizer; silent).completion
end

function _jump_layers(
    decomposition::AbstractDecomposition,
    proj_fn::Function,
    dual_model::JuMP.Model,
    optimizer;
    silent=true,
)
    completion_model, (p_ref, y_ref, ref_map) = make_completion_model(decomposition, dual_model)
    JuMP.set_optimizer(completion_model, optimizer)
    silent && JuMP.set_silent(completion_model)
    completion_model.ext[:🔒] = ReentrantLock()
    # TODO: define frule/rrule using b-Ax
    # TODO: handle infeasibility?
    z_vars = [getindex.(Ref(ref_map), zv) for zv in get_z_dual(dual_model, decomposition)]
    w_vars = getindex.(Ref(ref_map), _quadslack_vars(dual_model))
    solve! =
        (y_pred, param_value, extract) -> begin
            lock(completion_model.ext[:🔒])
            try
                y_proj = proj_fn(y_pred)
                JuMP.set_parameter_value.(p_ref, param_value)
                JuMP.set_parameter_value.(flatten_y(y_ref), flatten_y(y_proj))

                JuMP.optimize!(completion_model)
                JuMP.assert_is_solved_and_feasible(completion_model)

                extract(y_proj)
            finally
                unlock(completion_model.ext[:🔒])
            end
        end
    objective =
        (y_pred, param_value) ->
            solve!(y_pred, param_value, _ -> JuMP.objective_value(completion_model))
    completion =
        (y_pred, param_value) -> solve!(
            y_pred,
            param_value,
            y_proj -> (y=y_proj, z=[JuMP.value.(zv) for zv in z_vars], w=JuMP.value.(w_vars)),
        )
    return (objective=objective, completion=completion, batch=nothing)
end

# Quadratic slack variables of the dual, ordered by the primal variable index.
function _quadslack_vars(dual_model::JuMP.Model)
    pdm = dual_model.ext[:_Dualization_jl_PrimalDualMap]
    qs = pdm.primal_var_in_quad_obj_to_dual_slack_var
    return [JuMP.VariableRef(dual_model, qs[k]) for k in sort!(collect(keys(qs)); by=k -> k.value)]
end

function make_completion_model(decomposition::AbstractDecomposition, dual_model::JuMP.Model)
    return _make_completion_model(decomposition, dual_model)
end
function _make_completion_model(decomposition::AbstractDecomposition, dual_model::JuMP.Model)
    # make model without y cone constraints
    y_cone_constraints = filter(!isnothing, get_y_constraint(dual_model, decomposition))
    completion_model, ref_map = JuMP.copy_model(
        dual_model; filter_constraints=c -> !(c in y_cone_constraints)
    )

    # mark y as parameters (optimizing over z only)
    p_ref = getindex.(ref_map, get_p(dual_model, decomposition))
    y_ref = getindex.(ref_map, get_y_dual(dual_model, decomposition))
    y_ref_flat = flatten_y(y_ref)
    JuMP.@constraint(completion_model, y_ref_flat .∈ MOI.Parameter.(zeros(length(y_ref_flat))))

    return completion_model, (p_ref, y_ref, ref_map)
end

function get_y_dual(dual_model, decomposition::AbstractDecomposition)
    return Dualization._get_dual_variables.(dual_model, decomposition.y_ref)
end

function get_p(dual_model, decomposition::AbstractDecomposition)
    return Dualization._get_dual_parameter.(dual_model, decomposition.p_ref)
end

"""
    completed_constraints(decomposition::AbstractDecomposition) -> Vector{JuMP.ConstraintRef}

Primal constraints whose dual variables (`z`) are recovered by the completion
step rather than predicted. The fallback is the field `decomposition.z_ref`.
"""
completed_constraints(decomposition::AbstractDecomposition) = decomposition.z_ref

function get_z_dual(dual_model, decomposition::AbstractDecomposition)
    return [
        Dualization._get_dual_variables(dual_model, cr) for
        cr in completed_constraints(decomposition)
    ]
end

function get_y_constraint(dual_model, decomposition::AbstractDecomposition)
    return Dualization._get_dual_constraint.(dual_model, decomposition.y_ref)
end
