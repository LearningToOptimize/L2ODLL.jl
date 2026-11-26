# Batched closed-form layers, projections and completions (src/batched.jl).

using Adapt: Adapt
using JLArrays: JLArrays
using L2OCore: L2OCore
using Zygote: Zygote

const _EXT = Base.get_extension(L2ODLL, :L2ODLLL2OCoreExt)

columns(Y, shape) = [L2ODLL.unflatten_y(Y[:, k], shape) for k in axes(Y, 2)]
_flatz(z) = isempty(z) ? Float64[] : L2ODLL.flatten_y(z)

@testset "Batched layers" begin
    # The batched layer, its gradients (forward-mode
    # pushforwards and reverse-mode) and its completion equal the per-instance ones.
    @testset "batched matches per-instance: $name" for (name, make) in INSTANCE_CLASSES
        rng = Random.Xoshiro(hash(name) + 5)
        model, sampler, _ = make(rng)
        L2ODLL.decompose!(model)
        layer = batched_layer(model)
        @test layer isa L2ODLL.AbstractBatchedLayer
        B = 7
        P = reduce(hcat, [sampler(rng) for _ in 1:B])
        shape = y_shape(model)
        Y = randn(rng, sum(shape), B)
        ys = columns(Y, shape)
        loop = [dual_objective(model, ys[k], P[:, k]) for k in 1:B]
        Gloop = reduce(
            hcat, [L2ODLL.flatten_y(dual_objective_gradient(model, ys[k], P[:, k])) for k in 1:B]
        )
        @test layer(Y, P) ≈ loop rtol = 1e-12 atol = 1e-12
        v, G = dual_objective_and_gradient(model, Y, P)               # ForwardDiff, batched
        @test v ≈ loop rtol = 1e-12 atol = 1e-12
        @test G ≈ Gloop rtol = 1e-10 atol = 1e-12
        _, Gz = dual_objective_and_gradient(
            model, Y, P; ad_type=DifferentiationInterface.AutoZygote()
        )
        @test Gz ≈ Gloop rtol = 1e-10 atol = 1e-12
        c = batched_completion(layer, Y, P)
        cs = [dual_completion(model, ys[k], P[:, k]) for k in 1:B]
        @test c.y ≈ reduce(hcat, [L2ODLL.flatten_y(ci.y) for ci in cs])
        @test c.z ≈ reduce(hcat, [_flatz(ci.z) for ci in cs]) atol = 1e-12
        @test size(c.w) == (length(first(cs).w), B)
        isempty(first(cs).w) || @test c.w ≈ reduce(hcat, [ci.w for ci in cs])

        # The L2OCore completion takes the batched path and agrees with the column path.
        pipe = L2OCore.L2OPipeline(L2OCore.ParametricProblem(model), DLLMethod())
        D = L2OCore.complete(pipe, P, Y).dual
        Dcols = reduce(
            hcat, [_EXT._complete_column(pipe.decomposition, Y[:, k], P[:, k]) for k in 1:B]
        )
        @test D ≈ Dcols atol = 1e-12
    end

    # The matrix projection equals the per-instance
    # MathOptSetDistances projection, including the generic fallback.
    @testset "batched projection matches MathOptSetDistances" begin
        rng = Random.Xoshiro(31)
        for make in (bounded_lp, bounded_socp, convex_qp)
            model, _, _ = make(rng)
            cache = L2ODLL.decompose!(model)
            bp = L2ODLL.BatchedProjection(cache.decomposition, cache.dual_model)
            shape = y_shape(model)
            Y = 3 .* randn(rng, sum(shape), 20)
            ref = reduce(hcat, [L2ODLL.flatten_y(cache.y_proj(y)) for y in columns(Y, shape)])
            @test bp(Y) ≈ ref atol = 1e-12
        end
        # second-order cone special cases: inside, polar (→ 0), outside, v = 0
        blk = L2ODLL.BatchedProjection([
            L2ODLL.ProjectionBlock(:soc, 1:3, 0.0, MOI.SecondOrderCone(3))
        ])
        Y = [2.0 -5.0 1.0 0.0 -1.0; 1.0 1.0 2.0 0.0 0.0; 0.5 1.0 -1.0 0.0 0.0]
        ref = reduce(hcat, [L2ODLL._projection_fn(MOI.SecondOrderCone(3), Y[:, k]) for k in 1:5])
        @test blk(Y) ≈ ref atol = 1e-12
        @test blk(Y)[:, 1] == Y[:, 1]
        @test all(iszero, blk(Y)[:, 2])
        # every closed-form block matches the column-wise reference projection
        for set in (
            MOI.GreaterThan(0.5),
            MOI.LessThan(-0.5),
            MOI.EqualTo(1.5),
            MOI.Nonnegatives(2),
            MOI.Nonpositives(2),
            MOI.Zeros(2),
            MOI.Reals(2),
        )
            n = MOI.dimension(set)
            bp = L2ODLL.BatchedProjection([L2ODLL._projection_block(set, 1:n, :euclidean)])
            Y = 2 .* randn(rng, n, 6)
            ref = reduce(hcat, [L2ODLL._projection_fn(set, Y[:, k]) for k in 1:6])
            @test bp(Y) ≈ ref atol = 1e-12
        end
        # generic fallback (rotated second-order cone)
        set = MOI.RotatedSecondOrderCone(3)
        gen = L2ODLL.BatchedProjection([L2ODLL.ProjectionBlock(:generic, 1:3, 0.0, set)])
        Y = randn(rng, 3, 4)
        @test gen(Y) ≈ reduce(hcat, [L2ODLL._projection_fn(set, Y[:, k]) for k in 1:4])
        # gradients through the SOC projection are finite at v = 0
        g = only(Zygote.gradient(Yv -> sum(blk(Yv)), [1.0 -1.0; 0.0 0.0; 0.0 0.0]))
        @test all(isfinite, g)
    end

    # Radial projection, eq. 26/39 of Tanneau and Van Hentenryck (2024): maps into
    # the cone, is the identity on the cone, and keeps a nonzero gradient where the
    # Euclidean projection is identically zero (the polar cone).
    @testset "radial projection" begin
        rad = L2ODLL.BatchedProjection([
            L2ODLL.ProjectionBlock(:soc_radial, 1:3, 0.0, MOI.SecondOrderCone(3))
        ])
        euc = L2ODLL.BatchedProjection([
            L2ODLL.ProjectionBlock(:soc, 1:3, 0.0, MOI.SecondOrderCone(3))
        ])
        rng = Random.Xoshiro(37)
        Y = 2 .* randn(rng, 3, 50)
        R = rad(Y)
        @test all(R[1, k] >= sqrt(R[2, k]^2 + R[3, k]^2) - 1e-12 for k in 1:50)   # in the cone
        @test R[2:3, :] == Y[2:3, :]                                                 # along ρ = e₁
        @test rad(euc(Y)) ≈ euc(Y)                                                    # identity on the cone
        polar = reshape([-3.0, 0.5, 0.5], 3, 1)
        @test all(iszero, only(Zygote.gradient(Yv -> sum(euc(Yv)), polar))[2:3])
        @test any(!iszero, only(Zygote.gradient(Yv -> sum(rad(Yv)), polar))[2:3])

        # Valid dual bounds with the radial projection, per instance and batched.
        for sense in (MOI.MAX_SENSE, MOI.MIN_SENSE)
            model, sampler, _ = bounded_socp(rng; sense)
            L2ODLL.decompose!(model; projection=:radial)
            s = bound_sign(model)
            for _ in 1:3
                pv = sampler(rng)
                opt = solve_at!(model, pv, SOLVER)
                opt === nothing && continue
                Y = randn(rng, sum(y_shape(model)), 6)
                P = repeat(pv, 1, 6)
                vals = dual_objective(model, Y, P)
                @test all(s .* (vals .- opt) .<= 1e-7 * max(1, abs(opt)))
                @test vals ≈ [dual_objective(model, y, pv) for y in columns(Y, y_shape(model))]
                @test dual_objective(model, JuMP.dual.(get_y(model)), pv) ≈ opt atol =
                    1e-5 * max(1, abs(opt))
            end
        end
        @test_throws ArgumentError L2ODLL.decompose!(bounded_socp(rng)[1]; projection=:other)
    end

    # `completion = :log` through decompose! and DLLMethod
    # gives valid bounds that are below the exact completion's (DLL Theorem 2).
    @testset "log-barrier completion through decompose! and DLLMethod" begin
        rng = Random.Xoshiro(41)
        model, sampler, _ = bounded_lp(rng)
        exact, _, _ = bounded_lp(Random.Xoshiro(41))
        L2ODLL.decompose!(model; completion=:log, μ=0.1)
        L2ODLL.decompose!(exact)
        pv = sampler(rng)
        opt = solve_at!(model, pv, SOLVER)
        Y = randn(rng, sum(y_shape(model)), 8)
        P = repeat(pv, 1, 8)
        vlog, vex = dual_objective(model, Y, P), dual_objective(exact, Y, P)
        @test all(vlog .<= vex .+ 1e-10)
        @test all(vex .<= opt + 1e-7 * max(1, abs(opt)))
        @test vlog ≈ [dual_objective(model, y, pv) for y in columns(Y, y_shape(model))]
        z = batched_completion(batched_layer(model), Y, P).z
        n = size(z, 1) ÷ 2
        @test all(z[1:n, :] .> 0) && all(z[(n + 1):end, :] .< 0)       # strictly interior

        problem = L2OCore.ParametricProblem(model)
        pipe = L2OCore.L2OPipeline(problem, DLLMethod(; completion=:log, μ=0.1))
        pred = L2OCore.complete(pipe, P, Y)
        report = L2OCore.evaluate(problem, P, pred)
        @test maximum(report.dual_infeasibility) <= 1e-6
        @test all(report.dual_bound .<= opt + 1e-6 * max(1, abs(opt)))
        qp, _, _ = convex_qp(rng)
        @test_throws ArgumentError L2ODLL.decompose!(qp; completion=:log)
    end

    # With JLArrays (a reference GPU array type that forbids scalar
    # indexing), the adapted layers give the CPU values and Zygote gradients.
    @testset "GPU arrays (JLArrays): $name" for (name, make) in INSTANCE_CLASSES[[1, 3, 5, 8]]
        JLArrays.allowscalar(false)
        rng = Random.Xoshiro(hash(name) + 7)
        model, sampler, _ = make(rng)
        L2ODLL.decompose!(model)
        layer = batched_layer(model)
        P = reduce(hcat, [sampler(rng) for _ in 1:5])
        Y = randn(rng, sum(y_shape(model)), 5)
        G = only(Zygote.gradient(Yv -> sum(layer(Yv, P)), Y))
        for l in (layer, L2ODLL.densify(layer))
            dev = Adapt.adapt(JLArrays.JLArray, l)
            Yd, Pd = JLArrays.JLArray(Y), JLArrays.JLArray(P)
            v = dev(Yd, Pd)
            @test v isa JLArrays.JLArray
            @test Array(v) ≈ layer(Y, P) rtol = 1e-12 atol = 1e-12
            Gd = only(Zygote.gradient(Yv -> sum(dev(Yv, Pd)), Yd))
            @test Gd isa JLArrays.JLArray
            @test Array(Gd) ≈ G rtol = 1e-10 atol = 1e-12
            c = batched_completion(dev, Yd, Pd)
            @test Array(c.z) ≈ batched_completion(layer, Y, P).z atol = 1e-12
        end
    end
end
