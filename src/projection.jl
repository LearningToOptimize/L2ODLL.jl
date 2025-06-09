function make_proj_fn(decomposition::AbstractDecomposition, dual_model::JuMP.Model)
    sets = [
        isnothing(set) ? nothing : MOI.get(dual_model, MOI.ConstraintSet(), set)
        for set in Dualization._get_dual_constraint.(dual_model, decomposition.y_ref)
    ]

    # TODO: detect if there are any constraints in only y and p that we aren't projecting on to
    return (y_prediction) -> [
        if isnothing(set)
            y
        elseif length(y) == 1 && (
            set isa MOI.EqualTo ||
            set isa MOI.LessThan ||
            set isa MOI.GreaterThan
            # TODO: add other sets with one dual variable
        )
            [MOSD.projection_on_set(MOSD.DefaultDistance(), only(y), set)]
        else
            MOSD.projection_on_set(MOSD.DefaultDistance(), y, set)
        end
        for (set, y) in zip(sets, y_prediction)
    ]
end