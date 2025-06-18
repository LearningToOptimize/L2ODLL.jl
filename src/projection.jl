
"""
    make_proj_fn(decomposition::AbstractDecomposition, dual_model::JuMP.Model)

Create a function that projects the raw dual variable predictions onto their dual cone constraints.
"""
function make_proj_fn(decomposition::AbstractDecomposition, dual_model::JuMP.Model)
    sets = get_y_sets(dual_model, decomposition)
    # TODO: detect if there are any constraints in only y and p that we aren't projecting on to
    return (y_prediction) -> _projection_fn.(sets, y_prediction)
end

function _projection_fn(set, y)
    if isnothing(set)
        return y
    elseif length(y) == 1 && (
        set isa MOI.EqualTo ||
        set isa MOI.LessThan ||
        set isa MOI.GreaterThan
        # TODO: add other sets with one dual variable
    )
        # vector type here?
        return [MOSD.projection_on_set(MOSD.DefaultDistance(), only(y), set)]
    else
        return MOSD.projection_on_set(MOSD.DefaultDistance(), y, set)
    end
end

function get_y_sets(dual_model, decomposition)
    return [
        isnothing(set) ? nothing : MOI.get(dual_model, MOI.ConstraintSet(), set)
        for set in get_y_constraint(dual_model, decomposition)
    ]
end

function make_jump_proj_fn(decomposition::AbstractDecomposition, dual_model::JuMP.Model, optimizer)
    sets = get_y_sets(dual_model, decomposition)
    shapes = y_shape(dual_model, decomposition)

    proj_model = JuMP.Model(optimizer)

    idxs = [(i, ji) for (i,j) in enumerate(shapes) for ji in 1:j]
    JuMP.@variable(proj_model, y[idxs])

    for (i, set) in enumerate(sets)
        isnothing(set) && continue
        y_vars = filter(ij->first(ij)==i, idxs)
        if length(y_vars) == 1
            JuMP.@constraint(proj_model, y[only(y_vars)] ∈ set)
        else
            JuMP.@constraint(proj_model, y[y_vars] ∈ set)
        end
    end

    return (y_prediction) -> begin
        JuMP.set_objective_function(proj_model, sum((y .- reduce(vcat, y_prediction)).^2))
        # TODO: ensure reduce(vcat) and idxs are same order
        JuMP.set_objective_sense(proj_model, MOI.MIN_SENSE)
        JuMP.optimize!(proj_model)
        return value.(y)
    end
end
    
    