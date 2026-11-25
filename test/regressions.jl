# Regression tests for fixed bugs.

import Clarabel

@testset "Regressions" begin
    function small_bounded_lp()
        m = JuMP.Model()
        JuMP.@variable(m, 0 <= x[1:2] <= 1)
        JuMP.@variable(m, p in MOI.Parameter(1.0))
        JuMP.@constraint(m, c1, x[1] + x[2] >= p)
        JuMP.@objective(m, Min, x[1] + 2x[2])
        return m, x, c1
    end

    @testset "GenericDecomposition(model, y_ref) with concrete refs" begin
        m, _, c1 = small_bounded_lp()
        d = L2ODLL.GenericDecomposition(m, [c1])
        @test d.y_ref == [c1]
        @test length(d.z_ref) == 4             # the four bounds
        @test length(d.p_ref) == 1
        @test !L2ODLL.can_decompose(m, L2ODLL.GenericDecomposition)
    end

    @testset "GenericDecomposition with an optimizer" begin
        m, _, c1 = small_bounded_lp()
        L2ODLL.decompose!(m, L2ODLL.GenericDecomposition(m, [c1]);
            optimizer = () -> ParametricOptInterface.Optimizer(HiGHS.Optimizer()))
        m2, _, _ = small_bounded_lp()
        L2ODLL.decompose!(m2)
        for y in ([[0.3]], [[2.5]], [[-1.0]])
            @test L2ODLL.dual_objective(m, y, [1.0]) ≈ L2ODLL.dual_objective(m2, y, [1.0]) atol = 1e-6
        end
        @test L2ODLL.get_z(m) == L2ODLL.completed_constraints(L2ODLL.get_cache(m).decomposition)
        @test length(L2ODLL.dual_completion(m, [[0.3]], [1.0]).z) == 4
        @test_throws ErrorException L2ODLL.decompose!(small_bounded_lp()[1],
            L2ODLL.GenericDecomposition(small_bounded_lp()[1], JuMP.ConstraintRef[]))
    end

    @testset "make_jump_proj_fn matches make_proj_fn" begin
        m, _, _ = small_bounded_lp()
        d = L2ODLL.BoundDecomposition(m)
        dm = L2ODLL.Dualization.dualize(m; consider_constrained_variables = false)
        jproj = L2ODLL.make_jump_proj_fn(d, dm, Clarabel.Optimizer)
        proj = L2ODLL.make_proj_fn(d, dm)
        for y in ([[0.3]], [[-0.3]])
            @test L2ODLL.flatten_y(jproj(y)) ≈ L2ODLL.flatten_y(proj(y)) atol = 1e-6
        end
    end

    @testset "AffExprMatrix from an affine expression" begin
        m, x, _ = small_bounded_lp()
        a = L2ODLL.AffExprMatrix(1.5 * x[1] - x[2] + 2, x)
        @test a([3.0, 5.0]) ≈ 1.5 * 3 - 5 + 2
    end

    @testset "Bounded convex QP is detected as ConvexQP" begin
        m = JuMP.Model()
        JuMP.@variable(m, -1 <= x[1:2] <= 1)
        JuMP.@variable(m, p in MOI.Parameter(0.5))
        JuMP.@constraint(m, x[1] + x[2] >= p)
        JuMP.@objective(m, Min, x[1]^2 + x[2]^2 + x[1])
        @test !L2ODLL.can_decompose(m, L2ODLL.BoundDecomposition)
        @test L2ODLL.decompose!(m).decomposition isa L2ODLL.ConvexQP
        @test_throws ErrorException L2ODLL.decompose!(m, L2ODLL.BoundDecomposition(m))
    end

    @testset "Custom layer without completion" begin
        m, _, _ = small_bounded_lp()
        L2ODLL.decompose!(m, L2ODLL.BoundDecomposition(m); dll_layer_builder = (d, p, dm) -> ((y, pv) -> 0.0))
        @test L2ODLL.dual_objective(m, [[1.0]], [1.0]) == 0.0
        @test_throws ErrorException L2ODLL.dual_completion(m, [[1.0]], [1.0])
    end
end
