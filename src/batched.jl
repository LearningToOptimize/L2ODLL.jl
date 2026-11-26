# Batched (matrix) closed-form DLL layers.
#
# A batch is a matrix with one instance per column: predictions `Y` are
# `sum(y_shape) × B`, parameters `P` are `num_parameters × B`. The layers below
# evaluate the projection, the closed-form completion and the dual objective for
# the whole batch with matrix products and broadcasts only: no loop over columns,
# no scalar indexing and no mutation. They can therefore be differentiated with
# reverse-mode AD (Zygote) and moved to accelerator arrays with
# `Adapt.adapt(ArrayType, layer)`.

# ---------------------------------------------------------------------------
# Projection onto the dual cones, block by block
# ---------------------------------------------------------------------------

"""
    ProjectionBlock

Rows `rows` of a prediction matrix that belong to one constraint, and how to
project them onto the dual cone `set`: `kind` is one of `:free`, `:ge`, `:le`,
`:eq` (scalar sets with constant `a`), `:nonneg`, `:nonpos`, `:zeros`, `:soc`,
`:soc_radial` or `:generic` (per-column MathOptSetDistances fallback, CPU only).
"""
struct ProjectionBlock
    kind::Symbol
    rows::UnitRange{Int}
    a::Float64
    set::Any
end

"""
    BatchedProjection(decomposition, dual_model; projection = :euclidean)

Projection of a prediction matrix onto the dual cones of the predicted
constraints. `projection = :euclidean` matches [`make_proj_fn`](@ref);
`projection = :radial` uses the radial projection of Tanneau and Van Hentenryck
(2024, eq. 26) for second-order cones, `(t, v) ↦ (max(t, ‖v‖), v)` (their eq. 39),
which never maps a prediction to zero and so avoids vanishing gradients; for
scalar sets both coincide.
"""
struct BatchedProjection
    blocks::Vector{ProjectionBlock}
end

function BatchedProjection(
    decomposition::AbstractDecomposition, dual_model::JuMP.Model; projection::Symbol=:euclidean
)
    projection in (:euclidean, :radial) ||
        throw(ArgumentError("projection must be :euclidean or :radial"))
    sets = get_y_sets(dual_model, decomposition)
    shapes = y_shape(dual_model, decomposition)
    blocks = ProjectionBlock[]
    offset = 0
    for (set, n) in zip(sets, shapes)
        rows = (offset + 1):(offset + n)
        offset += n
        push!(blocks, _projection_block(set, rows, projection))
    end
    return BatchedProjection(blocks)
end

_projection_block(::Nothing, rows, _) = ProjectionBlock(:free, rows, 0.0, nothing)
_projection_block(s::MOI.GreaterThan, rows, _) = ProjectionBlock(:ge, rows, s.lower, s)
_projection_block(s::MOI.LessThan, rows, _) = ProjectionBlock(:le, rows, s.upper, s)
_projection_block(s::MOI.EqualTo, rows, _) = ProjectionBlock(:eq, rows, s.value, s)
_projection_block(s::MOI.Nonnegatives, rows, _) = ProjectionBlock(:nonneg, rows, 0.0, s)
_projection_block(s::MOI.Nonpositives, rows, _) = ProjectionBlock(:nonpos, rows, 0.0, s)
_projection_block(s::MOI.Zeros, rows, _) = ProjectionBlock(:zeros, rows, 0.0, s)
_projection_block(s::MOI.Reals, rows, _) = ProjectionBlock(:free, rows, 0.0, s)
function _projection_block(s::MOI.SecondOrderCone, rows, p)
    return ProjectionBlock(p == :radial ? :soc_radial : :soc, rows, 0.0, s)
end
_projection_block(s, rows, _) = ProjectionBlock(:generic, rows, 0.0, s)

"""
    (proj::BatchedProjection)(Y::AbstractMatrix) -> AbstractMatrix

Project every column of `Y` onto the product of dual cones.
"""
function (proj::BatchedProjection)(Y::AbstractMatrix)
    isempty(proj.blocks) && return Y
    return reduce(vcat, [_project_block(b, Y[b.rows, :]) for b in proj.blocks])
end

function _project_block(b::ProjectionBlock, Yb::AbstractMatrix)
    k = b.kind
    if k === :free
        return Yb
    elseif k === :ge
        return max.(Yb, b.a)
    elseif k === :le
        return min.(Yb, b.a)
    elseif k === :eq
        return zero(Yb) .+ b.a
    elseif k === :nonneg
        return max.(Yb, 0)
    elseif k === :nonpos
        return min.(Yb, 0)
    elseif k === :zeros
        return zero(Yb)
    elseif k === :soc
        return _project_soc(Yb)
    elseif k === :soc_radial
        return _project_soc_radial(Yb)
    else
        cols = [_projection_fn(b.set, collect(Yb[:, j])) for j in axes(Yb, 2)]
        return reduce(hcat, cols)
    end
end

# Euclidean projection onto {(t, v) : ‖v‖ ≤ t}, column-wise:
#   ‖v‖ ≤ t      → (t, v)
#   ‖v‖ ≤ -t     → 0
#   otherwise    → α (1, v/‖v‖) with α = (t + ‖v‖)/2.
function _project_soc(Yb::AbstractMatrix)
    t = Yb[1:1, :]
    v = Yb[2:end, :]
    # `floatmin` keeps the derivative of sqrt finite at v = 0 (then a branch
    # without the division is selected, so the value is unaffected).
    nv = sqrt.(sum(abs2, v; dims=1) .+ floatmin(eltype(Yb)))
    inside = nv .<= t
    origin = nv .<= -t
    α = (t .+ nv) ./ 2
    tp = ifelse.(inside, t, ifelse.(origin, zero(t), α))
    vp = ifelse.(inside, v, ifelse.(origin, zero(v), (α ./ nv) .* v))
    return vcat(tp, vp)
end

# Radial projection along ρ = (1, 0, …, 0) (eq. 39 of Tanneau and Van Hentenryck, 2024).
function _project_soc_radial(Yb::AbstractMatrix)
    t = Yb[1:1, :]
    v = Yb[2:end, :]
    nv = sqrt.(sum(abs2, v; dims=1) .+ floatmin(eltype(Yb)))
    return vcat(max.(t, nv), v)
end

# ---------------------------------------------------------------------------
# Batched layers
# ---------------------------------------------------------------------------

"""
    AbstractBatchedLayer

A closed-form DLL layer evaluated on a whole batch. Call it as `layer(Y, P)` to
get the vector of dual objectives, and use [`batched_completion`](@ref) for the
completed duals. Obtain one with [`batched_layer`](@ref).
"""
abstract type AbstractBatchedLayer end

"""
    BoundedBatchedLayer

Batched form of [`bounded_builder`](@ref): `R = A_z [Y; P] .+ b_z` (`= c - Aᵀy`),
`z_l, z_u` by the exact or log-barrier closed form, then the dual objective.
"""
struct BoundedBatchedLayer{MZ,VZ,MQ,VC,T,C<:BoundedCompletion} <: AbstractBatchedLayer
    proj::BatchedProjection
    Az::MZ
    bz::VZ
    Q::MQ
    c::VC
    c0::T
    completer::C
end

"""
    ConvexQPBatchedLayer

Batched form of [`convex_qp_builder`](@ref): `W = Q_d⁻¹ (A_z [Y; P] .+ b_z)`
(the Lagrangian minimizer), then the dual objective.
"""
struct ConvexQPBatchedLayer{MZ,VZ,MI,MQ,VC,T} <: AbstractBatchedLayer
    proj::BatchedProjection
    Az::MZ
    bz::VZ
    Qinv::MI
    Q::MQ
    c::VC
    c0::T
end

Adapt.@adapt_structure BoundedBatchedLayer
Adapt.@adapt_structure ConvexQPBatchedLayer
# The projection holds only index ranges and set metadata.
Adapt.adapt_structure(_, p::BatchedProjection) = p

# Column-wise xᵀQx + cᵀx + c0.
function _batch_quadform(Q, c, c0, X::AbstractMatrix)
    return vec(sum(X .* (Q * X); dims=1)) .+ vec(c' * X) .+ c0
end

function _bounded_parts(layer::BoundedBatchedLayer, Y::AbstractMatrix, P::AbstractMatrix)
    Yp = layer.proj(Y)
    V = vcat(Yp, P)
    R = layer.Az * V .+ layer.bz
    Zl, Zu = complete_zlzu(layer.completer, R)
    return Yp, V, Zl, Zu
end

function (layer::BoundedBatchedLayer)(Y::AbstractMatrix, P::AbstractMatrix)
    _, V, Zl, Zu = _bounded_parts(layer, Y, P)
    return _batch_quadform(layer.Q, layer.c, layer.c0, vcat(V, Zl, Zu))
end

function _convex_qp_parts(layer::ConvexQPBatchedLayer, Y::AbstractMatrix, P::AbstractMatrix)
    Yp = layer.proj(Y)
    V = vcat(Yp, P)
    W = layer.Qinv * (layer.Az * V .+ layer.bz)
    return Yp, V, W
end

function (layer::ConvexQPBatchedLayer)(Y::AbstractMatrix, P::AbstractMatrix)
    _, V, W = _convex_qp_parts(layer, Y, P)
    return _batch_quadform(layer.Q, layer.c, layer.c0, vcat(V, W))
end

"""
    batched_completion(layer::AbstractBatchedLayer, Y, P) -> (y, z, w)

Batched [`dual_completion`](@ref): matrices with one column per instance. `y` is
the projected prediction, `z` the completed constraint duals (rows in the order
of [`completed_constraints`](@ref), i.e. `[z_l; z_u]` for the bounded
decomposition, empty for `ConvexQP`) and `w` the quadratic slacks.
"""
function batched_completion(layer::BoundedBatchedLayer, Y::AbstractMatrix, P::AbstractMatrix)
    Yp, _, Zl, Zu = _bounded_parts(layer, Y, P)
    return (y=Yp, z=vcat(Zl, Zu), w=Zl[1:0, :])
end

function batched_completion(layer::ConvexQPBatchedLayer, Y::AbstractMatrix, P::AbstractMatrix)
    Yp, _, W = _convex_qp_parts(layer, Y, P)
    return (y=Yp, z=W[1:0, :], w=W)
end

"""
    batched_layer(model::JuMP.Model) -> Union{AbstractBatchedLayer,Nothing}

The batched closed-form layer built by [`decompose!`](@ref) (or `nothing` for
solver-based or custom layers). `layer(Y, P)` returns the dual objectives of a
batch; [`dual_objective`](@ref) and friends use it automatically for matrix
inputs. To run on an accelerator, move it with `Adapt.adapt(ArrayType, layer)`
and pass `Y`, `P` of that array type; sparse matrices are converted by the
array package's `adapt` rules (or call [`densify`](@ref) first).
"""
batched_layer(model::JuMP.Model) = get_cache(model).batch

"""
    densify(layer::AbstractBatchedLayer) -> AbstractBatchedLayer

Copy of `layer` with dense matrices (convenient before `Adapt.adapt` to array
types without sparse support).
"""
densify(l::BoundedBatchedLayer) =
    BoundedBatchedLayer(l.proj, Matrix(l.Az), l.bz, Matrix(l.Q), l.c, l.c0, l.completer)
function densify(l::ConvexQPBatchedLayer)
    return ConvexQPBatchedLayer(l.proj, Matrix(l.Az), l.bz, Matrix(l.Qinv), Matrix(l.Q), l.c, l.c0)
end

# A per-instance projection (vector of vectors) built from a batched one.
function _single_instance_projection(proj::BatchedProjection, shape::Vector{Int})
    return y -> unflatten_y(vec(proj(reshape(flatten_y(y), :, 1))), shape)
end
