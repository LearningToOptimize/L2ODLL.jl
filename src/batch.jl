abstract type AbstractExprMatrix end

struct AffExprMatrix{V,T} <: AbstractExprMatrix
    c::V
    c0::T
end
struct QuadExprMatrix{M,V,T} <: AbstractExprMatrix
    Q::M
    c::V
    c0::T
end
struct VecAffExprMatrix{M,V} <: AbstractExprMatrix
    A::M
    b::V
end


Adapt.@adapt_structure AffExprMatrix
Adapt.@adapt_structure QuadExprMatrix
Adapt.@adapt_structure VecAffExprMatrix

(aem::AffExprMatrix)(x) = aem.c'*x + aem.c0
(qem::QuadExprMatrix)(x) = x'*qem.Q*x + qem.c'*x + qem.c0
(vaem::VecAffExprMatrix)(x) = vaem.A*x + vaem.b


function AffExprMatrix(
    aff::Vector{JuMP.GenericAffExpr{T,V}},
    v::Vector{V};
    backend = nothing
) where {T,V}
    n = length(v)

    c = zeros(T, n)

    for (coeff, vr) in JuMP.linear_terms(aff)
        c[vr_to_idx[vr]] = coeff
    end

    c = _backend_vector(backend)(c)
    return AffExprMatrix(c, c0)
end


function QuadExprMatrix(
    qexpr::JuMP.GenericQuadExpr{T,V},
    v::Vector{V};
    backend = nothing
) where {T,V}
    quad_terms = JuMP.quad_terms(qexpr)
    nq = length(quad_terms)
    n = length(v)

    vr_to_idx = _vr_to_idx(v)

    Qi = Int[]
    sizehint!(Qi, nq)
    Qj = Int[]
    sizehint!(Qj, nq)
    Qv = T[]
    sizehint!(Qv, nq)
    c = zeros(T, n)
    c0 = qexpr.aff.constant

    for (coeff, vr1, vr2) in quad_terms
        push!(Qi, vr_to_idx[vr1])
        push!(Qj, vr_to_idx[vr2])
        push!(Qv, coeff)
    end
    for (coeff, vr) in JuMP.linear_terms(qexpr)
        c[vr_to_idx[vr]] = coeff
    end

    Q = _backend_matrix(backend)(Qi, Qj, Qv, n, n)
    c = _backend_vector(backend)(c)
    return QuadExprMatrix(Q, c, c0)
end


function VecAffExprMatrix(
    vaff::Vector{JuMP.GenericAffExpr{T,V}},
    v::Vector{V};
    backend = nothing
) where {T,V}
    m = length(vaff)
    n = length(v)

    linear_terms = [JuMP.linear_terms(jaff) for jaff in vaff]
    nlinear = sum(length.(linear_terms))

    vr_to_idx = _vr_to_idx(v)

    Ai = Int[]
    sizehint!(Ai, nlinear)
    Aj = Int[]
    sizehint!(Aj, nlinear)
    Av = T[]
    sizehint!(Av, nlinear)
    b = zeros(T, m)

    for (i, jaff) in enumerate(vaff)
        for (coeff, vr) in linear_terms[i]
            if vr ∈ v
                push!(Ai, i)
                push!(Aj, vr_to_idx[vr])
                push!(Av, coeff)
            else
                error("Variable $vr from function $i not found")
            end
        end
        b[i] = jaff.constant
    end

    A = _backend_matrix(backend)(Ai, Aj, Av, m, n)
    b = _backend_vector(backend)(b)
    return VecAffExprMatrix(A, b)
end

function _vr_to_idx(v::V) where {V}
    vr_to_idx = Dict{eltype(V), Int}()
    for (i, vr) in enumerate(v)
        vr_to_idx[vr] = i
    end
    return vr_to_idx
end

function _backend_matrix(::Nothing)
    return SparseArrays.sparse
end

function _backend_vector(::Nothing)
    return Vector
end
