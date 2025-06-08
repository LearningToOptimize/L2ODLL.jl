# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

mutable struct VectorizeExceptVariableIndexBridge{T,F,S,G} <: MOI.Bridges.Constraint.AbstractBridge
    vector_constraint::MOI.ConstraintIndex{F,S}
    set_constant::T # constant in scalar set
end

const VectorizeExceptVariableIndex{T,OT<:MOI.ModelLike} =
    MOI.Bridges.Constraint.SingleBridgeOptimizer{VectorizeExceptVariableIndexBridge{T},OT}

function MOI.Bridges.Constraint.bridge_constraint(
    ::Type{VectorizeExceptVariableIndexBridge{T,F,S,G}},
    model::MOI.ModelLike,
    scalar_f::G,
    set::MOI.Utilities.ScalarLinearSet{T},
) where {T,F,S,G}
    MOI.throw_if_scalar_and_constant_not_zero(scalar_f, typeof(set))
    set_const = MOI.constant(set)
    vector_f = MOI.Utilities.operate(
        vcat,
        T,
        MOI.Utilities.operate(-, T, scalar_f, set_const),
    )
    vector_constraint = MOI.add_constraint(model, vector_f, S(1))
    return VectorizeExceptVariableIndexBridge{T,F,S,G}(vector_constraint, set_const)
end

function MOI.supports_constraint(
    ::Type{VectorizeExceptVariableIndexBridge{T}},
    ::Type{F},
    ::Type{<:MOI.Utilities.ScalarLinearSet{T}},
) where {T,F<:MOI.AbstractScalarFunction}
    F <: MOI.VariableIndex && return false               # L2ODLL: only this line is changed
    return MOI.Utilities.is_coefficient_type(F, T)
end

function MOI.Bridges.added_constrained_variable_types(::Type{<:VectorizeExceptVariableIndexBridge})
    return Tuple{Type}[]
end

function MOI.Bridges.added_constraint_types(
    ::Type{<:VectorizeExceptVariableIndexBridge{T,F,S}},
) where {T,F,S}
    return Tuple{Type,Type}[(F, S)]
end

function MOI.Bridges.Constraint.concrete_bridge_type(
    ::Type{<:VectorizeExceptVariableIndexBridge{T}},
    G::Type{<:MOI.AbstractScalarFunction},
    S::Type{<:MOI.Utilities.ScalarLinearSet{T}},
) where {T}
    H = MOI.Utilities.promote_operation(-, T, G, T)
    F = MOI.Utilities.promote_operation(vcat, T, H)
    return VectorizeExceptVariableIndexBridge{T,F,MOI.Utilities.vector_set_type(S),G}
end

function MOI.get(
    ::VectorizeExceptVariableIndexBridge{T,F,S},
    ::MOI.NumberOfConstraints{F,S},
)::Int64 where {T,F,S}
    return 1
end

function MOI.get(
    bridge::VectorizeExceptVariableIndexBridge{T,F,S},
    ::MOI.ListOfConstraintIndices{F,S},
) where {T,F,S}
    return [bridge.vector_constraint]
end

function MOI.delete(model::MOI.ModelLike, bridge::VectorizeExceptVariableIndexBridge)
    MOI.delete(model, bridge.vector_constraint)
    return
end

function MOI.supports(
    model::MOI.ModelLike,
    attr::Union{MOI.ConstraintPrimalStart,MOI.ConstraintDualStart},
    ::Type{VectorizeExceptVariableIndexBridge{T,F,S,G}},
) where {T,F,S,G}
    return MOI.supports(model, attr, MOI.ConstraintIndex{F,S})
end

function MOI.get(
    model::MOI.ModelLike,
    attr::MOI.ConstraintPrimalStart,
    bridge::VectorizeExceptVariableIndexBridge,
)
    x = MOI.get(model, attr, bridge.vector_constraint)
    if x === nothing
        return nothing
    end
    return only(x) + bridge.set_constant
end

function MOI.get(
    model::MOI.ModelLike,
    attr::MOI.ConstraintPrimal,
    bridge::VectorizeExceptVariableIndexBridge,
)
    x = MOI.get(model, attr, bridge.vector_constraint)
    if MOI.Utilities.is_ray(MOI.get(model, MOI.PrimalStatus(attr.result_index)))
        # If it is an infeasibility certificate, it is a ray and satisfies the
        # homogenized problem, see https://github.com/jump-dev/MathOptInterface.jl/issues/433
        return only(x)
    else
        # Otherwise, we need to add the set constant since the ConstraintPrimal
        # is defined as the value of the function and the set_constant was
        # removed from the original function
        return only(x) + bridge.set_constant
    end
end

function MOI.set(
    model::MOI.ModelLike,
    attr::MOI.ConstraintPrimalStart,
    bridge::VectorizeExceptVariableIndexBridge,
    value,
)
    MOI.set(
        model,
        attr,
        bridge.vector_constraint,
        [value - bridge.set_constant],
    )
    return
end

function MOI.set(
    model::MOI.ModelLike,
    attr::MOI.ConstraintPrimalStart,
    bridge::VectorizeExceptVariableIndexBridge,
    ::Nothing,
)
    MOI.set(model, attr, bridge.vector_constraint, nothing)
    return
end

function MOI.get(
    model::MOI.ModelLike,
    attr::Union{MOI.ConstraintDual,MOI.ConstraintDualStart},
    bridge::VectorizeExceptVariableIndexBridge,
)
    x = MOI.get(model, attr, bridge.vector_constraint)
    if x === nothing
        return nothing
    end
    return only(x)
end

function MOI.set(
    model::MOI.ModelLike,
    attr::MOI.ConstraintDualStart,
    bridge::VectorizeExceptVariableIndexBridge,
    value,
)
    if value === nothing
        MOI.set(model, attr, bridge.vector_constraint, nothing)
    else
        MOI.set(model, attr, bridge.vector_constraint, [value])
    end
    return
end

function MOI.modify(
    model::MOI.ModelLike,
    bridge::VectorizeExceptVariableIndexBridge,
    change::MOI.ScalarCoefficientChange,
)
    MOI.modify(
        model,
        bridge.vector_constraint,
        MOI.MultirowChange(change.variable, [(1, change.new_coefficient)]),
    )
    return
end

function MOI.set(
    model::MOI.ModelLike,
    ::MOI.ConstraintSet,
    bridge::VectorizeExceptVariableIndexBridge,
    new_set::MOI.Utilities.ScalarLinearSet,
)
    bridge.set_constant = MOI.constant(new_set)
    MOI.modify(
        model,
        bridge.vector_constraint,
        MOI.VectorConstantChange([-bridge.set_constant]),
    )
    return
end

function MOI.get(
    model::MOI.ModelLike,
    attr::MOI.ConstraintFunction,
    bridge::VectorizeExceptVariableIndexBridge{T,F,S,G},
) where {T,F,S,G}
    f = MOI.Utilities.scalarize(
        MOI.get(model, attr, bridge.vector_constraint),
        true,
    )
    return convert(G, only(f))
end

function MOI.get(
    ::MOI.ModelLike,
    ::MOI.ConstraintSet,
    bridge::VectorizeExceptVariableIndexBridge{T,F,S},
) where {T,F,S}
    return MOI.Utilities.scalar_set_type(S, T)(bridge.set_constant)
end
