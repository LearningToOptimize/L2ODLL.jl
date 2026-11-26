# The hypotheses of the closed forms, the explicit formulas of the Theory page
# (docs/src/theory.md) computed by hand, and further results of the papers.

# Projected y-blocks of the `ineq` (ℝ₊ duals) and `eq` (free) constraints, looked
# up by constraint reference (the order of `get_y` follows the constraint types).
function split_y(model, y)
    refs = L2ODLL.get_y(model)
    pos(c) = findfirst(==(c), refs)
    yv = [max(only(y[pos(c)]), 0.0) for c in model[:ineq]]
    yeq = only(y[pos(model[:eq])])
    return yv, yeq
end

@testset "Closed forms and hypotheses" begin
    # Example 3 of Tanneau and Van Hentenryck (2024) needs
    # Q ≻ 0. With an indefinite Q the primal below is unbounded (OPT = -∞), so a
    # finite "bound" would be wrong; with a singular Q, Q⁻¹ does not exist.
    @testset "ConvexQP requires a strictly convex objective" begin
        function qp(Q)
            m = JuMP.Model()
            JuMP.@variable(m, x[1:2])
            JuMP.@variable(m, p in MOI.Parameter(1.0))
            JuMP.@constraint(m, x[1] + x[2] == p)
            JuMP.@objective(m, Min, sum(Q[i, j] * x[i] * x[j] for i in 1:2, j in 1:2) + x[1])
            return m
        end
        for Q in ([1.0 2.0; 2.0 1.0], [1.0 1.0; 1.0 1.0])     # indefinite, singular
            m = qp(Q)
            @test !L2ODLL.is_strictly_convex_objective(m)
            @test !L2ODLL.can_decompose(m, L2ODLL.ConvexQP)
            @test_throws ErrorException L2ODLL.decompose!(m)
            @test_throws ErrorException L2ODLL.decompose!(m, L2ODLL.ConvexQP(m))
        end
        m = qp([1.0 0.2; 0.2 1.0])
        @test L2ODLL.objective_hessian(m) ≈ [2.0 0.4; 0.4 2.0]
        @test L2ODLL.is_strictly_convex_objective(m)
        # Max of a concave objective is accepted; Max of a convex one is not.
        m = qp([1.0 0.0; 0.0 1.0])
        JuMP.set_objective_sense(m, MOI.MAX_SENSE)
        @test !L2ODLL.is_strictly_convex_objective(m)
        m = qp(-[1.0 0.0; 0.0 1.0])
        JuMP.set_objective_sense(m, MOI.MAX_SENSE)
        @test L2ODLL.is_strictly_convex_objective(m)
    end

    # Parametrization (Theory page): parameters only in the
    # constants b(p) and objective coefficients c(p). A parameter multiplying a
    # variable in a constraint is rejected rather than silently mis-handled.
    @testset "parameters multiplying variables are rejected" begin
        m = JuMP.Model()
        JuMP.@variable(m, 0 <= x[1:2] <= 1)
        JuMP.@variable(m, p in MOI.Parameter(0.5))
        JuMP.@constraint(m, p * x[1] + x[2] >= 0.5)
        JuMP.@objective(m, Min, x[1] + 2x[2])
        @test_throws Exception L2ODLL.decompose!(m)
    end

    # Constants and parameter-only terms of the primal
    # objective carry over to Φ (tightness at y⋆ would fail otherwise).
    @testset "objective constants enter the dual objective" begin
        m = JuMP.Model()
        JuMP.@variable(m, 0 <= x[1:2] <= 1)
        JuMP.@variable(m, p in MOI.Parameter(0.5))
        JuMP.@constraint(m, x[1] + x[2] >= p)
        JuMP.@objective(m, Min, x[1] + 2x[2] + 7 + 3p + p^2)
        L2ODLL.decompose!(m)
        for pv in (0.5, 1.5)
            opt = solve_at!(m, [pv], SOLVER)
            @test L2ODLL.dual_objective(m, JuMP.dual.(L2ODLL.get_y(m)), [pv]) ≈ opt atol = 1e-6
        end
    end

    # DIPOL Theorem 3 (concavity of the Lagrangian dual function): for
    # y₁, y₂ in the dual cone, Φ((y₁+y₂)/2) ≥ (Φ(y₁) + Φ(y₂))/2.
    @testset "dual objective is concave in y: $name" for (name, make) in INSTANCE_CLASSES
        rng = Random.Xoshiro(hash(name) + 2)
        model, sampler, _ = make(rng)
        L2ODLL.decompose!(model)
        proj = L2ODLL.get_cache(model).y_proj
        s = bound_sign(model)               # Φ is concave for Min, convex for Max
        for _ in 1:10
            pv = sampler(rng)
            y1, y2 = proj(random_y(rng, model)), proj(random_y(rng, model))
            ym = [(a .+ b) ./ 2 for (a, b) in zip(y1, y2)]
            Φ = y -> s * L2ODLL.dual_objective(model, y, pv)
            @test Φ(ym) >= (Φ(y1) + Φ(y2)) / 2 - 1e-9 * (1 + abs(Φ(ym)))
        end
    end

    # DLL Theorem 2: for fixed y the exact completion gives the
    # best bound; any other feasible completion, e.g. the log-barrier one, is worse.
    @testset "exact completion gives the best bound" begin
        rng = Random.Xoshiro(13)
        for (make, sense) in (
            (bounded_lp, MOI.MIN_SENSE), (bounded_lp, MOI.MAX_SENSE), (bounded_socp, MOI.MAX_SENSE)
        )
            model, sampler, _ = make(rng; sense)
            L2ODLL.decompose!(model)
            cache = L2ODLL.get_cache(model)
            d, dm = cache.decomposition, cache.dual_model
            s = bound_sign(model)
            for μ in (1e-2, 1.0)
                logΦ = L2ODLL.bounded_builder(d, cache.y_proj, dm; completion=:log, μ)
                for _ in 1:5
                    pv, y = sampler(rng), random_y(rng, model)
                    @test s * logΦ(y, pv) <= s * L2ODLL.dual_objective(model, y, pv) + 1e-10
                end
            end
        end
    end

    # Eq. (4): Φ = (b + p)ᵀy + (e + 0.1p₁) y_eq + lᵀz_l + uᵀz_u with
    # z_l = |r|⁺, z_u = -|r|⁻, r = c(p) - Aᵀy - Eᵀy_eq, computed by hand from the
    # data of `bounded_lp` (MOI form: the constants are -(b + p) and -(e + 0.1p₁)).
    @testset "bounded closed form (eq. 4) by hand" begin
        rng = Random.Xoshiro(17)
        model, sampler, data = bounded_lp(rng)
        L2ODLL.decompose!(model)
        m = length(data.b)
        for _ in 1:5
            pv = sampler(rng)
            y = random_y(rng, model)
            yv, yeq = split_y(model, y)
            r = data.c .+ data.C * pv .- data.A' * yv .- vec(data.E) .* yeq
            Φ =
                (data.b .+ pv)' * yv +
                (data.e[1] + 0.1pv[1]) * yeq +
                data.l' * max.(r, 0) +
                data.u' * min.(r, 0)
            @test L2ODLL.dual_objective(model, y, pv) ≈ Φ rtol = 1e-10
        end
    end

    # Eq. (6): Φ = (b + p)ᵀy + e y_eq - ½ rᵀQ⁻¹r with r = c(p) - Aᵀy - Eᵀy_eq
    # for the free strictly convex QP min ½xᵀQx + c(p)ᵀx s.t. Ax ≥ b + p, Ex = e.
    @testset "ConvexQP closed form (eq. 6) by hand" begin
        rng = Random.Xoshiro(19)
        model, sampler, data = convex_qp(rng; bounds=:free)
        L2ODLL.decompose!(model)
        m = length(data.b)
        for _ in 1:5
            pv = sampler(rng)
            y = random_y(rng, model)
            yv, yeq = split_y(model, y)
            r = data.c .+ data.C * pv .- data.A' * yv .- vec(data.E) .* yeq
            Φ = (data.b .+ pv)' * yv + data.e[1] * yeq - r' * (data.Q \ r) / 2
            @test L2ODLL.dual_objective(model, y, pv) ≈ Φ rtol = 1e-8
            # and w is the Lagrangian minimizer x(y) = Q⁻¹(Aᵀy - c) = -Q⁻¹r
            @test L2ODLL.dual_completion(model, y, pv).w ≈ -(data.Q \ r) rtol = 1e-8
        end
    end

    # DLL Example 3: w is the minimizer of the Lagrangian,
    # so completing the optimal y returns the (unique) primal optimum.
    qp_classes = INSTANCE_CLASSES[5:end]
    @testset "quadratic slack recovers the primal optimum: $name" for (name, make) in qp_classes
        rng = Random.Xoshiro(hash(name) + 3)
        model, sampler, _ = make(rng)
        L2ODLL.decompose!(model)
        x = filter(!JuMP.is_parameter, JuMP.all_variables(model))
        pv = sampler(rng)
        solve_at!(model, pv, SOLVER) === nothing && continue
        w = L2ODLL.dual_completion(model, JuMP.dual.(L2ODLL.get_y(model)), pv).w
        @test w ≈ JuMP.value.(x) atol = 1e-5
    end

    # DIPOL Theorems 3 and 5: the barrier-augmented
    # completion value Ξ_μ(r) = max {lᵀz_l + uᵀz_u + μ Σ(ln z_l + ln(-z_u)) : z_l + z_u = r}
    # has derivative x_μ(r) given in closed form by Theorem 5. Evaluating Ξ_μ at
    # L2ODLL's completion (eq. 7) and differentiating checks that eq. 7 is the
    # exact maximizer (envelope theorem) and the closed form of x_μ.
    @testset "log-barrier gradient matches Theorem 5" begin
        rng = Random.Xoshiro(23)
        n = 15
        l, u = -1 .- rand(rng, n), 1 .+ rand(rng, n)
        for μ in (1e-3, 0.1, 1.0)
            c = L2ODLL.LogBoundedCompletion(μ, l, u)
            Ξ = r -> begin
                zl, zu = L2ODLL.complete_zlzu(c, r)
                l' * zl + u' * zu + μ * sum(log.(zl) .+ log.(-zu))
            end
            r = 3 .* randn(rng, n)
            xμ = (u .+ l) ./ 2 .+ μ ./ r .- sign.(r) .* sqrt.((μ ./ r) .^ 2 .+ ((u .- l) ./ 2) .^ 2)
            @test ForwardDiff.gradient(Ξ, r) ≈ xμ rtol = 1e-8
            @test all(l .< xμ .< u)                      # interior, as μ > 0
        end
        # eq. (7) needs l < u and μ > 0
        m = JuMP.Model()
        JuMP.@variable(m, 0 <= x[1:2] <= 1)
        JuMP.set_lower_bound(x[2], 1.0)                  # l₂ = u₂
        JuMP.@variable(m, p in MOI.Parameter(0.5))
        JuMP.@constraint(m, x[1] + x[2] >= p)
        JuMP.@objective(m, Min, x[1] + 2x[2])
        cache = L2ODLL.decompose!(m)
        @test_throws ErrorException L2ODLL.bounded_builder(
            cache.decomposition, cache.y_proj, cache.dual_model; completion=:log, μ=0.1
        )
        model, _, _ = bounded_lp(rng)
        cache = L2ODLL.decompose!(model)
        @test_throws ErrorException L2ODLL.bounded_builder(
            cache.decomposition, cache.y_proj, cache.dual_model; completion=:log, μ=0.0
        )
    end
end
