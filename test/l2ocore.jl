# The L2OCore interface (package extension L2ODLLL2OCoreExt).

using L2OCore: L2OCore

@testset "L2OCore interface" begin
    method = L2ODLL.DLLMethod()
    @test method isa L2OCore.AbstractDualMethod
    @test L2OCore.supports(method, L2OCore.LinearConic())
    @test L2OCore.supports(method, L2OCore.ConvexQuadratic())
    @test !L2OCore.supports(method, L2OCore.MixedInteger())
    @test !L2OCore.supports(method, L2OCore.SmoothNonlinear())

    for (name, make) in INSTANCE_CLASSES
        @testset "$name" begin
            rng = Random.Xoshiro(hash(name) + 1)
            model, sampler, _ = make(rng)
            problem = L2OCore.ParametricProblem(model)
            pipe = L2OCore.L2OPipeline(problem, method)
            @test L2OCore.output_dimension(pipe) == sum(L2ODLL.y_shape(model))
            @test L2OCore.input_dimension(pipe) == L2OCore.num_parameters(problem)

            B = 4
            P = reduce(hcat, [sampler(rng) for _ in 1:B])
            Z = randn(rng, L2OCore.output_dimension(pipe), B)
            pred = L2OCore.complete(pipe, P, Z)
            @test pred.primal === nothing
            @test size(pred.dual) == (L2OCore.num_duals(problem), B)

            # L2OCore's independent Lagrangian dual function of the
            # completed duals equals L2ODLL's dual objective, and the completed
            # point is dual feasible.
            bound, infeas = L2OCore.dual_bound(pipe.evaluator, pred.dual, P)
            @test maximum(infeas) <= 1e-8
            @test bound ≈ L2ODLL.dual_objective(model, Z, P) rtol = 1e-8 atol = 1e-8

            # The completed duals bound the solver optima (L2OCore's evaluation).
            data = L2OCore.solve_instances(problem, P; optimizer=SOLVER, primal=false, dual=true)
            ok = data.status .== MOI.OPTIMAL
            @test count(ok) >= B - 1
            report = L2OCore.evaluate(
                problem, P[:, ok], L2OCore.Prediction(; dual=pred.dual[:, ok]); atol=1e-6
            )
            opt = data.objective[ok]
            @test maximum(report.dual_infeasibility) <= 1e-6
            @test all(
                bound_sign(model) .* (report.dual_bound .- opt) .<= 1e-6 .* max.(1, abs.(opt))
            )

            # The loss is the (sign-adjusted) mean dual objective: minimizing it
            # maximizes the bound (eq. (27a) of Tanneau and Van Hentenryck, 2024).
            ℓ = L2OCore.loss(pipe, P, Z)
            @test ℓ ≈ -bound_sign(model) * sum(bound) / B

            # The bound is tight at the optimal duals, through the pipeline: extract
            # the predicted rows of the solver's duals (L2ODLL.get_y order), complete
            # them, and recover the optimal value.
            pos = Dict(
                c => r for
                (c, r) in zip(L2OCore.constraint_refs(problem), L2OCore.dual_layout(problem))
            )
            rows = reduce(vcat, [collect(pos[c]) for c in L2ODLL.get_y(model)])
            Ystar = data.dual[rows, ok]
            bstar, _ = L2OCore.dual_bound(
                pipe.evaluator, L2OCore.complete(pipe, P[:, ok], Ystar).dual, P[:, ok]; atol=1e-5
            )
            @test bstar ≈ data.objective[ok] rtol = 1e-5 atol = 1e-5

            # Self-supervised data: inputs only unless an optimizer is given.
            ds = L2OCore.generate_data(
                pipe; sampler=L2OCore.BoxSampler(P[:, 1] .- 1e-3, P[:, 1] .+ 1e-3), n=2, rng
            )
            @test ds.dual === nothing && ds.objective === nothing
            @test size(ds.inputs) == (L2OCore.num_parameters(problem), 2)

            @test_throws DimensionMismatch L2OCore.complete(pipe, P, Z[1:(end - 1), :])
        end
    end

    @testset "Explicit decomposition" begin
        model, sampler, _ = bounded_lp(Random.Xoshiro(1))
        pipe = L2OCore.L2OPipeline(
            L2OCore.ParametricProblem(model),
            L2ODLL.DLLMethod(; decomposition=L2ODLL.BoundDecomposition),
        )
        @test pipe.decomposition.cache.decomposition isa L2ODLL.BoundDecomposition
        P = reshape(sampler(Random.Xoshiro(2)), :, 1)
        pred = L2OCore.complete(
            pipe, P, randn(Random.Xoshiro(3), L2OCore.output_dimension(pipe), 1)
        )
        @test size(pred.dual, 2) == 1
    end
end

using Zygote: Zygote

# The DLL loss is differentiable with Zygote (the reverse-mode
# backend used by Flux and commonly with Lux), and agrees with ForwardDiff.
@testset "Zygote gradient matches ForwardDiff" begin
    for (name, make) in INSTANCE_CLASSES[[1, 3, 5]]
        rng = Random.Xoshiro(21)
        model, sampler, _ = make(rng)
        pipe = L2OCore.L2OPipeline(L2OCore.ParametricProblem(model), L2ODLL.DLLMethod())
        P = reduce(hcat, [sampler(rng) for _ in 1:3])
        Z = randn(rng, L2OCore.output_dimension(pipe), 3)
        gz = only(Zygote.gradient(z -> L2OCore.loss(pipe, P, z), Z))
        gf = ForwardDiff.gradient(z -> L2OCore.loss(pipe, P, z), Z)
        @test gz ≈ gf rtol = 1e-10 atol = 1e-12
    end
end
