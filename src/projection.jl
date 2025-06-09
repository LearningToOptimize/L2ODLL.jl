function make_proj_fn(decomposition::AbstractDecomposition, dual_model::JuMP.Model)
    sets = [
        isnothing(set) ? nothing : MOI.get(dual_model, MOI.ConstraintSet(), set)
        for set in get_y_constraint(dual_model, decomposition)
    ]

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