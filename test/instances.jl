# Random parametric instances of every problem class L2ODLL supports.
#
# Each generator returns `(model, sampler)` where `sampler(rng)` draws a feasible
# parameter vector. All data is drawn from `rng`, so tests are deterministic.

using JuMP: JuMP
using LinearAlgebra: LinearAlgebra
using Random: Random

"""
Bounded LP (Example 1 of Tanneau and Van Hentenryck, 2024):

    min/max (c + C p)ᵀx  s.t.  A x ≥ b + p,  E x == e + 0.1 p₁,  l ≤ x ≤ u.
"""
function bounded_lp(rng; n=5, m=3, sense=MOI.MIN_SENSE, equality=true)
    A = randn(rng, m, n)
    E = randn(rng, 1, n)
    x0 = rand(rng, n) .- 0.5
    b = A * x0 .- rand(rng, m)                   # x0 strictly satisfies A x ≥ b at p = 0
    e = E * x0
    c = randn(rng, n)
    C = 0.5 .* randn(rng, n, m)
    model = JuMP.Model()
    JuMP.@variable(model, -2 <= x[1:n] <= 2)
    JuMP.@variable(model, p[1:m] in MOI.Parameter.(zeros(m)))
    JuMP.@constraint(model, ineq[i=1:m], sum(A[i, j] * x[j] for j in 1:n) >= b[i] + p[i])
    if equality
        JuMP.@constraint(model, eq, sum(E[1, j] * x[j] for j in 1:n) == e[1] + 0.1 * p[1])
    end
    obj = sum((c[j] + sum(C[j, i] * p[i] for i in 1:m)) * x[j] for j in 1:n)
    JuMP.@objective(model, sense, obj)
    sampler = rng -> 0.2 .* (2 .* rand(rng, m) .- 1)
    data = (; A, b, c, C, E, e, l=fill(-2.0, n), u=fill(2.0, n))
    return model, sampler, data
end

"""
Bounded SOCP (portfolio with a risk budget; parameters in objective and cone):

    max μᵀx  s.t.  1ᵀx == 1,  (γ, F x) ∈ SOC,  0 ≤ x ≤ 1.
"""
function bounded_socp(rng; n=4, sense=MOI.MAX_SENSE)
    F = 0.1 .* randn(rng, n, n)
    model = JuMP.Model()
    JuMP.@variable(model, 0 <= x[1:n] <= 1)
    JuMP.@variable(model, μ[1:n] in MOI.Parameter.(zeros(n)))
    JuMP.@variable(model, γ in MOI.Parameter(1.0))
    JuMP.@constraint(model, budget, sum(x) == 1)
    JuMP.@constraint(model, risk, [γ; F * x] in JuMP.SecondOrderCone())
    if sense == MOI.MAX_SENSE
        JuMP.@objective(model, Max, LinearAlgebra.dot(μ, x))
    else
        JuMP.@objective(model, Min, -LinearAlgebra.dot(μ, x))
    end
    # γ ≥ ‖F‖₂ guarantees that every x in the simplex is feasible.
    γmin = LinearAlgebra.opnorm(F)
    sampler = rng -> [randn(rng, n); γmin * (1 + rand(rng))]
    return model, sampler, (; F)
end

"""
Strictly convex QP (Example 3 of Tanneau and Van Hentenryck, 2024):

    min ½ xᵀQx + (c + C p)ᵀx  s.t.  A x ≥ b + p,  E x == e,  x ≥ -3
    (or the equivalent max of the negated objective).
"""
function convex_qp(rng; n=4, m=2, sense=MOI.MIN_SENSE, bounds=:lower)
    M = randn(rng, n, n)
    Q = M * M' + LinearAlgebra.I
    A = randn(rng, m, n)
    E = randn(rng, 1, n)
    x0 = rand(rng, n) .- 0.5
    b = A * x0 .- rand(rng, m)
    e = E * x0
    c = randn(rng, n)
    C = 0.5 .* randn(rng, n, m)
    model = JuMP.Model()
    if bounds == :lower
        JuMP.@variable(model, x[1:n] >= -3)
    elseif bounds == :box
        JuMP.@variable(model, -3 <= x[1:n] <= 3)
    else
        JuMP.@variable(model, x[1:n])
    end
    JuMP.@variable(model, p[1:m] in MOI.Parameter.(zeros(m)))
    JuMP.@constraint(model, ineq[i=1:m], sum(A[i, j] * x[j] for j in 1:n) >= b[i] + p[i])
    JuMP.@constraint(model, eq, sum(E[1, j] * x[j] for j in 1:n) == e[1])
    lin = sum((c[j] + sum(C[j, i] * p[i] for i in 1:m)) * x[j] for j in 1:n)
    quad = 0.5 * sum(Q[i, j] * x[i] * x[j] for i in 1:n, j in 1:n)
    if sense == MOI.MIN_SENSE
        JuMP.@objective(model, Min, quad + lin)
    else
        JuMP.@objective(model, Max, -quad - lin)
    end
    sampler = rng -> 0.2 .* (2 .* rand(rng, m) .- 1)
    return model, sampler, (; Q, A, b, c, C, E, e)
end

const INSTANCE_CLASSES = [
    ("bounded LP, min", rng -> bounded_lp(rng; sense=MOI.MIN_SENSE)),
    ("bounded LP, max", rng -> bounded_lp(rng; sense=MOI.MAX_SENSE)),
    ("bounded SOCP, max", rng -> bounded_socp(rng; sense=MOI.MAX_SENSE)),
    ("bounded SOCP, min", rng -> bounded_socp(rng; sense=MOI.MIN_SENSE)),
    ("convex QP, min", rng -> convex_qp(rng; sense=MOI.MIN_SENSE)),
    ("convex QP, max", rng -> convex_qp(rng; sense=MOI.MAX_SENSE)),
    ("convex QP with box, min", rng -> convex_qp(rng; sense=MOI.MIN_SENSE, bounds=:box)),
    ("convex QP, free, max", rng -> convex_qp(rng; sense=MOI.MAX_SENSE, bounds=:free)),
]

"Solve `model` at parameter values `pv`; return the optimal objective (or `nothing`)."
function solve_at!(model, pv, optimizer)
    params = filter(JuMP.is_parameter, JuMP.all_variables(model))
    JuMP.set_parameter_value.(params, pv)
    JuMP.set_optimizer(model, optimizer)
    JuMP.set_silent(model)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model; dual=true) || return nothing
    return JuMP.objective_value(model)
end

"Sign `s` such that `s * (bound - opt) ≤ 0` is weak duality."
bound_sign(model) = JuMP.objective_sense(model) == MOI.MAX_SENSE ? -1 : 1

random_y(rng, model) = [randn(rng, length(v)) for v in L2ODLL.get_y_dual(model)]

"Central finite-difference gradient of `f` at `x`."
function central_fd(f, x; h=1e-6)
    g = similar(x)
    for i in eachindex(x)
        e = zero(x)
        e[i] = h
        g[i] = (f(x .+ e) - f(x .- e)) / (2h)
    end
    return g
end
