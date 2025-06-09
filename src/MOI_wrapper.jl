include("bridges.jl")

struct VectorStandardFormData{M, V, T}
    At::M
    Ht::M
    c::V
    Q::M
    q::V
    q0::T
end

function model_to_data(model; M=SparseArrays.SparseMatrixCSC{Float64,Int}, V=Vector{Float64}, T=Float64)
    sfm = StandardFormModel{Float64}()
    index_map = MOI.copy_to(VectorizeExceptVariableIndex{Float64}(sfm), model)
    data = cache_to_data(sfm; M, V, T)
    return data, index_map
end

MOIU.@product_of_sets(
    RHS,
    MOI.Zeros,
)

const StandardFormModel{T<:Real} = MOIU.GenericModel{T,
    MOIU.ObjectiveContainer{T},
    MOIU.VariablesContainer{T},
    MOIU.MatrixOfConstraints{T,
        MOIU.MutableSparseMatrixCSC{T,Int,MOIU.OneBasedIndexing},
        Vector{T}, RHS{T},
    },
}

function split_variables_parameters(vc::MOIU.VariablesContainer{T}) where {T<:Real}
    vars = MOI.VariableIndex[]
    params = MOI.VariableIndex[]
    for i in 1:length(vc.set_mask)
        if vc.set_mask[i] == MOIU._single_variable_flag(MOI.Parameter)
            push!(params, MOI.VariableIndex(i))
        else
            push!(vars, MOI.VariableIndex(i))
        end
    end
    return vars, params
end


function cache_to_data(
    cache::StandardFormModel;
    M::Type{<:AbstractMatrix}=SparseArrays.SparseMatrixCSC{Float64,Int},
    V::Type{<:AbstractVector}=Vector{Float64},
    T::Type{<:Real}=Float64,
)
    v_idx, p_idx = begin
        v, p = split_variables_parameters(cache.variables)
        getfield.(v, :value), getfield.(p, :value)
    end

    l = cache.variables.lower[v_idx]
    u = cache.variables.upper[v_idx]
    p = cache.variables.lower[p_idx]
    @assert p == cache.variables.upper[p_idx]

    AtHt = convert(
        SparseArrays.SparseMatrixCSC{Float64,Int},
        cache.constraints.coefficients,
    )
    c = -cache.constraints.constants

    At = AtHt[:, p_idx]
    Ht = AtHt[:, v_idx]

    obj_type = MOI.get(cache, MOI.ObjectiveFunctionType())
    @assert obj_type ∈ [
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarAffineFunction{Float64},
    ] "Unsupported objective function type $obj_type"

    n = MOI.get(cache, MOI.NumberOfVariables())
    Q, q, q0 = process_objective(n, MOI.get(cache, MOI.ObjectiveFunction{obj_type}()))

    return VectorStandardFormData{M,V,T}(
        At, Ht, c, Q, q, q0
    )
end

function process_objective(n::Int, f::MOI.ScalarAffineFunction{T}) where {T <: AbstractFloat}
    Q = spzeros(T, n, n)

    q = zeros(T, n)
    processlinearterms!(q, f.terms)

    q0 = f.constant
    return Q, q, q0
end

function process_objective(n::Int, f::MOI.ScalarQuadraticFunction{T}) where {T <: AbstractFloat}
    I = [Int(term.variable_1.value) for term in f.quadratic_terms]
    J = [Int(term.variable_2.value) for term in f.quadratic_terms]
    V = [term.coefficient for term in f.quadratic_terms]
    Q = sparse(I, J, V, n, n)

    q = zeros(T, n)
    processlinearterms!(q, f.affine_terms)
    q0 = f.constant

    return Q, q, q0
end

function process_objective(n::Int, f::MOI.VariableIndex)
    return process_objective(n, MOI.ScalarAffineFunction{Float64}(f))
end

function processlinearterms!(q, terms::Vector{<:MOI.ScalarAffineTerm})
    for term in terms
        var = term.variable
        coeff = term.coefficient
        q[var.value] += coeff
    end
end

function number_of_constraints(model; include_variable_in_set=false)
    constraint_types = MOI.get(model, MOI.ListOfConstraintTypesPresent())
    n_total = 0
    for (i, (F, S)) in enumerate(constraint_types)
        if !include_variable_in_set && F == MOI.VariableIndex
            continue
        end
        m = MOI.get(model, MOI.NumberOfConstraints{F,S}())
        n_total += m
    end
    return n_total
end