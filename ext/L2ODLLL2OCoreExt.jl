module L2ODLLL2OCoreExt

# Dual Lagrangian Learning as an `L2OCore.AbstractDualMethod`.
#
# Mapping onto the L2OCore pipeline (see docs/src/guide/l2ocore.md):
#
#   decompose  -> L2ODLL.decompose! (the y/z split) + index maps between L2ODLL's
#                 (y, z) ordering and L2OCore's `dual_layout`
#   complete   -> projection of y onto its dual cone + completion of z
#                 (Theorem 1 of Tanneau and Van Hentenryck, 2024)
#   loss       -> self-supervised: minus the mean dual objective, eq. (27a) of
#                 Tanneau and Van Hentenryck (2024), written for a minimization loss
#   guarantees -> ValidDualBound (weak duality for the completed, dual-feasible point)

import JuMP
import L2OCore
import L2ODLL
import Random

const MOI = JuMP.MOI

"""
    DualLagrangianLearning(decomposition, optimizer) <: L2OCore.AbstractDualMethod

Dual Lagrangian Learning (Tanneau and Van Hentenryck, 2024). Construct it with
[`L2ODLL.DLLMethod`](@ref).
"""
struct DualLagrangianLearning{D,O} <: L2OCore.AbstractDualMethod
    decomposition::D  # nothing (auto-detect) or `model -> L2ODLL.AbstractDecomposition`
    optimizer::O      # optimizer for completions without a closed form, or nothing
end

function L2ODLL.DLLMethod(; decomposition = nothing, optimizer = nothing)
    return DualLagrangianLearning(decomposition, optimizer)
end

# Assumptions: convex conic (or convex quadratic) instances, so that the
# Lagrangian dual of the completed point is a valid bound. Whether a closed-form
# completion exists is decided per model by `decompose` (it errors otherwise).
L2OCore.supports(::DualLagrangianLearning, ::L2OCore.LinearConic) = true
L2OCore.supports(::DualLagrangianLearning, ::L2OCore.ConvexQuadratic) = true

L2OCore.guarantees(::DualLagrangianLearning) = (L2OCore.ValidDualBound(),)

"""
    DLLDecomposition <: L2OCore.AbstractDecomposition

The L2OCore view of an [`L2ODLL.DLLCache`](@ref): the proxy predicts the
flattened `y` (`output_dimension = sum(L2ODLL.y_shape(model))` rows) and the
completion fills the remaining duals. `perm` maps the stacked vector
`[flatten_y(y); flatten_y(z)]` to L2OCore's `dual_layout` order.
"""
struct DLLDecomposition{C} <: L2OCore.AbstractDecomposition
    model::JuMP.Model
    cache::C
    y_shape::Vector{Int}
    perm::Vector{Int}
end

L2OCore.output_dimension(d::DLLDecomposition) = sum(d.y_shape; init = 0)

function L2OCore.decompose(problem::L2OCore.ParametricProblem, m::DualLagrangianLearning)
    model = problem.model
    cache = if m.decomposition === nothing
        L2ODLL.decompose!(model)
    else
        L2ODLL.decompose!(model, m.decomposition(model); optimizer = m.optimizer)
    end
    cache.completion === nothing &&
        error("the DLL decomposition has no completion; pass `optimizer` to `DLLMethod`")
    cache.decomposition.p_ref == L2OCore.parameters(problem) ||
        error("parameter order of the DLL decomposition differs from the ParametricProblem")
    shape = L2ODLL.y_shape(cache)
    return DLLDecomposition(model, cache, shape, _dual_permutation(problem, cache))
end

# Rows of `[flatten_y(y); flatten_y(z)]` for each constraint of `problem`, in
# `dual_layout(problem)` order.
function _dual_permutation(problem::L2OCore.ParametricProblem, cache::L2ODLL.DLLCache)
    rows = Dict{JuMP.ConstraintRef,UnitRange{Int}}()
    offset = 0
    dual_model = cache.dual_model
    for refs in (cache.decomposition.y_ref, L2ODLL.completed_constraints(cache.decomposition))
        for cr in refs
            n = length(L2ODLL.Dualization._get_dual_variables(dual_model, cr))
            rows[cr] = (offset+1):(offset+n)
            offset += n
        end
    end
    perm = Int[]
    for (cr, layout) in zip(L2OCore.constraint_refs(problem), L2OCore.dual_layout(problem))
        haskey(rows, cr) || error(
            "constraint $cr is neither predicted nor completed by the DLL decomposition; " *
            "use a decomposition that covers every constraint",
        )
        r = rows[cr]
        length(r) == length(layout) || error("dual dimension mismatch for constraint $cr")
        append!(perm, r)
    end
    return perm
end

_asbatch(x::AbstractMatrix) = x
_asbatch(x::AbstractVector) = reshape(x, :, 1)

function _check_output(d::DLLDecomposition, Z, P)
    size(Z, 1) == L2OCore.output_dimension(d) || throw(
        DimensionMismatch("output has $(size(Z, 1)) rows, expected $(L2OCore.output_dimension(d))"),
    )
    size(Z, 2) == size(P, 2) ||
        throw(DimensionMismatch("output has $(size(Z, 2)) columns, inputs have $(size(P, 2))"))
    return nothing
end

function _complete_column(d::DLLDecomposition, z::AbstractVector, p::AbstractVector)
    c = L2ODLL.dual_completion(d.model, L2ODLL.unflatten_y(collect(z), d.y_shape), collect(p))
    v = isempty(c.z) ? L2ODLL.flatten_y(c.y) : vcat(L2ODLL.flatten_y(c.y), L2ODLL.flatten_y(c.z))
    return v[d.perm]
end

"""
    complete(::DualLagrangianLearning, pipe, inputs, output) -> Prediction

Project each predicted `y` onto its dual cone and complete the remaining duals
(closed form for `BoundDecomposition` / `ConvexQP`, a solver otherwise). The
returned `dual` is `num_duals × B` in `dual_layout` order with `JuMP.dual` signs.

**Guarantee** `ValidDualBound` (Theorems 1-2 of Tanneau and Van Hentenryck, 2024):
the completed point is dual feasible, so its Lagrangian value bounds the optimum.
"""
function L2OCore.complete(::DualLagrangianLearning, pipe::L2OCore.L2OPipeline, inputs, output)
    d = pipe.decomposition::DLLDecomposition
    P, Z = _asbatch(inputs), _asbatch(output)
    _check_output(d, Z, P)
    cols = [_complete_column(d, view(Z, :, k), view(P, :, k)) for k in axes(Z, 2)]
    return L2OCore.Prediction(dual = reduce(hcat, cols))
end

# Sign that turns the dual objective into a quantity to *minimize*: the dual
# of a minimization is a maximization (and vice versa).
_loss_sign(model::JuMP.Model) = JuMP.objective_sense(model) == MOI.MAX_SENSE ? 1 : -1

"""
    loss(::DualLagrangianLearning, pipe, inputs, output; kwargs...) -> Real

Self-supervised DLL loss, eq. (27a) of Tanneau and Van Hentenryck (2024): the
mean dual objective of the completed predictions, negated for minimization
problems so that the loss is always minimized. Needs no labels.
"""
function L2OCore.loss(::DualLagrangianLearning, pipe::L2OCore.L2OPipeline, inputs, output; kwargs...)
    d = pipe.decomposition::DLLDecomposition
    P, Z = _asbatch(inputs), _asbatch(output)
    _check_output(d, Z, P)
    vals = [
        L2ODLL.dual_objective(d.model, L2ODLL.unflatten_y(collect(Z[:, k]), d.y_shape), collect(P[:, k]))
        for k in axes(Z, 2)
    ]
    return _loss_sign(d.model) * sum(vals) / length(vals)
end

"""
    generate_data(::DualLagrangianLearning, pipe; sampler, n, rng, optimizer = nothing, kwargs...)

DLL is self-supervised: without `optimizer` the dataset holds only the sampled
inputs. With `optimizer`, every instance is solved and the optimal objective
values and duals are recorded (useful as `reference` for checking
`ValidDualBound` and for reporting gaps).
"""
function L2OCore.generate_data(
    ::DualLagrangianLearning,
    pipe::L2OCore.L2OPipeline;
    sampler,
    n::Integer,
    rng::Random.AbstractRNG = Random.default_rng(),
    optimizer = nothing,
    kwargs...,
)
    P = L2OCore.sample_parameters(sampler, rng, n)
    optimizer === nothing && return L2OCore.L2ODataset(P)
    return L2OCore.solve_instances(pipe.problem, P; optimizer, primal = false, dual = true, kwargs...)
end

end # module
