# Numerical checks of the properties of DLL stated on the Theory page
# (docs/src/theory.md), on every instance class of instances.jl.

using Clarabel: Clarabel
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using ParametricOptInterface: ParametricOptInterface

const NUM_INSTANCES = 4      # parameter draws per class
const NUM_PREDICTIONS = 5    # random predictions per parameter draw
const SOLVER = Clarabel.Optimizer
const POI_SOLVER = () -> ParametricOptInterface.Optimizer(HiGHS.Optimizer())  # completions are LPs/QPs

@testset "Method properties" begin
    for (name, make) in INSTANCE_CLASSES
        @testset "$name" begin
            rng = Random.Xoshiro(hash(name))
            model, sampler, _ = make(rng)
            L2ODLL.decompose!(model)
            s = bound_sign(model)
            nsolved = 0
            for _ in 1:NUM_INSTANCES
                pv = sampler(rng)
                opt = solve_at!(model, pv, SOLVER)
                opt === nothing && continue
                nsolved += 1
                scale = max(1, abs(opt))

                # Theorems 1-2 of Tanneau and Van Hentenryck (2024):
                # any prediction, once projected and completed, bounds the optimum.
                @testset "dual bound is valid for any prediction" begin
                    for _ in 1:NUM_PREDICTIONS
                        y = random_y(rng, model)
                        @test s * (L2ODLL.dual_objective(model, y, pv) - opt) <= 1e-7 * scale
                    end
                end

                # Theorem 2: completing the optimal y recovers
                # the optimal value (strong duality holds for these instances).
                @testset "bound is tight at the optimal duals" begin
                    ystar = JuMP.dual.(L2ODLL.get_y(model))
                    @test L2ODLL.dual_objective(model, ystar, pv) ≈ opt atol = 1e-5 * scale
                end

                # The dual cone projection is a projection.
                @testset "projection is idempotent" begin
                    proj = L2ODLL.get_cache(model).y_proj
                    y = proj(random_y(rng, model))
                    @test L2ODLL.flatten_y(proj(y)) ≈ L2ODLL.flatten_y(y) atol = 1e-10
                end

                # The AD gradient of the completed
                # dual objective equals central differences (at a random, hence
                # almost surely smooth, point).
                @testset "gradient matches finite differences" begin
                    y = random_y(rng, model)
                    shape = L2ODLL.y_shape(model)
                    f = yf -> L2ODLL.dual_objective(model, L2ODLL.unflatten_y(yf, shape), pv)
                    g = L2ODLL.flatten_y(L2ODLL.dual_objective_gradient(model, y, pv))
                    gfd = central_fd(f, L2ODLL.flatten_y(y))
                    @test g ≈ gfd rtol = 1e-5 atol = 1e-6
                    v, g2 = L2ODLL.dual_objective_and_gradient(model, y, pv)
                    @test v ≈ f(L2ODLL.flatten_y(y))
                    @test L2ODLL.flatten_y(g2) ≈ g
                end
            end
            @test nsolved >= NUM_INSTANCES - 1   # the generators produce feasible instances
        end
    end

    # The closed-form completion (Examples 1 and 3 of
    # Tanneau and Van Hentenryck, 2024) equals the optimal solution of the
    # completion problem (9) solved numerically.
    @testset "closed-form completion matches the solver: $name" for (name, make, D) in [
        ("bounded LP", rng -> bounded_lp(rng), L2ODLL.BoundDecomposition),
        ("bounded SOCP", rng -> bounded_socp(rng), L2ODLL.BoundDecomposition),
        ("convex QP", rng -> convex_qp(rng), L2ODLL.ConvexQP),
    ]
        rng = Random.Xoshiro(7)
        m1, sampler, _ = make(rng)
        m2, _, _ = make(Random.Xoshiro(7))
        L2ODLL.decompose!(m1)
        L2ODLL.decompose!(
            m2,
            D(m2);
            dll_layer_builder=(d, p, dm) -> L2ODLL.jump_builder(d, p, dm, POI_SOLVER),
            completion_builder=(d, p, dm) -> L2ODLL.jump_completion_builder(d, p, dm, POI_SOLVER),
        )
        for _ in 1:3
            pv = sampler(rng)
            y = random_y(rng, m1)
            @test L2ODLL.dual_objective(m1, y, pv) ≈ L2ODLL.dual_objective(m2, y, pv) rtol = 1e-6 atol =
                1e-6
            c1 = L2ODLL.dual_completion(m1, y, pv)
            c2 = L2ODLL.dual_completion(m2, y, pv)
            if D === L2ODLL.ConvexQP
                @test c1.w ≈ c2.w rtol = 1e-5 atol = 1e-6
            else
                # The LP completion is unique where c - Aᵀy ≠ 0 (almost surely).
                @test L2ODLL.flatten_y(c1.z) ≈ L2ODLL.flatten_y(c2.z) rtol = 1e-5 atol = 1e-6
            end
        end
    end

    # For the bounded LP min cᵀx s.t. Ax ≥ b, l ≤ x ≤ u,
    # the gradient of the completed dual objective is b - A x̂ with x̂ⱼ = lⱼ if
    # (c - Aᵀy)ⱼ > 0 and uⱼ if (c - Aᵀy)ⱼ < 0 (Theorems 3-4 of Klamkin, Tanneau
    # and Van Hentenryck, 2025). y > 0 lies in the interior of the dual cone,
    # where the projection is the identity.
    @testset "gradient is b - A x̂ (Theorem 3)" begin
        rng = Random.Xoshiro(3)
        model, sampler, data = bounded_lp(rng; equality=false)
        L2ODLL.decompose!(model)
        for _ in 1:5
            pv = sampler(rng)
            y = [[abs(randn(rng)) + 0.1] for _ in 1:length(pv)]
            yv = L2ODLL.flatten_y(y)
            c = data.c .+ data.C * pv
            r = c .- data.A' * yv
            x̂ = ifelse.(r .> 0, data.l, data.u)
            expected = (data.b .+ pv) .- data.A * x̂
            @test L2ODLL.flatten_y(L2ODLL.dual_objective_gradient(model, y, pv)) ≈ expected atol =
                1e-10
            # and the completed z matches Example 1: zˡ = |r|⁺, zᵘ = -|r|⁻
            z = L2ODLL.flatten_y(L2ODLL.dual_completion(model, y, pv).z)
            n = length(r)
            @test z[1:n] ≈ max.(r, 0)
            @test z[(n + 1):end] ≈ min.(r, 0)
        end
    end

    # The smoothed completion of eq. (7) in Klamkin,
    # Tanneau and Van Hentenryck (2025) satisfies zˡ + zᵘ = z with zˡ > 0 > zᵘ,
    # and tends to the exact completion as μ → 0.
    @testset "log-barrier completion is feasible and converges" begin
        rng = Random.Xoshiro(11)
        z = randn(rng, 20)
        l, u = -rand(rng, 20), rand(rng, 20)
        for μ in (1.0, 1e-2, 1e-6)
            zl, zu = L2ODLL.complete_zlzu(L2ODLL.LogBoundedCompletion(μ, l, u), z)
            @test all(zl .> 0)
            @test all(zu .< 0)
            @test zl .+ zu ≈ z
        end
        zl, zu = L2ODLL.complete_zlzu(L2ODLL.LogBoundedCompletion(1e-9, l, u), z)
        el, eu = L2ODLL.complete_zlzu(L2ODLL.ExactBoundedCompletion(), z)
        @test zl ≈ el atol = 1e-6
        @test zu ≈ eu atol = 1e-6
    end

    # The matrix API equals the per-instance API.
    @testset "matrix API matches the per-instance API" begin
        rng = Random.Xoshiro(5)
        model, sampler, _ = bounded_socp(rng)
        L2ODLL.decompose!(model)
        B = 4
        P = reduce(hcat, [sampler(rng) for _ in 1:B])
        ys = [random_y(rng, model) for _ in 1:B]
        Y = reduce(hcat, L2ODLL.flatten_y.(ys))
        vals = L2ODLL.dual_objective(model, Y, P)
        G = L2ODLL.dual_objective_gradient(model, Y, P)
        v2, G2 = L2ODLL.dual_objective_and_gradient(model, Y, P)
        for k in 1:B
            @test vals[k] ≈ L2ODLL.dual_objective(model, ys[k], P[:, k])
            @test G[:, k] ≈ L2ODLL.flatten_y(L2ODLL.dual_objective_gradient(model, ys[k], P[:, k]))
        end
        @test v2 ≈ vals
        @test G2 ≈ G
        @test_throws DimensionMismatch L2ODLL.dual_objective(model, Y[1:(end - 1), :], P)
        @test_throws DimensionMismatch L2ODLL.dual_objective(model, Y, P[:, 1:(end - 1)])
    end
end
