using L2ODLL
using Test

import JuMP, Dualization, ParametricOptInterface
import Clarabel, HiGHS

import LinearAlgebra
import JSON, HTTP

function test_vectorize_bridge_matches_moi()
    url = "https://api.github.com/repos/jump-dev/MathOptInterface.jl/commits?path=src/Bridges/Constraint/bridges/VectorizeBridge.jl"
    response = HTTP.get(url)
    data = JSON.parse(String(response.body))
    @test data[1]["sha"] == "4e2630554afcde0b0b7c3e680e7fd3666e9e3825"
end
randn_like(vecofvecs) = [randn(length(v)) for v in vecofvecs];
SOLVER = () -> ParametricOptInterface.Optimizer(HiGHS.Optimizer());

@testset "L2ODLL.jl" begin
    test_vectorize_bridge_matches_moi()

    @testset "Markowitz Frontier" begin
        μ = [11.5; 9.5; 6] / 100
        Σ = [
            166 34 58
            34 64 4
            58 4 100
        ] / 100^2
        N = length(μ);

        m = JuMP.Model();
        JuMP.@variable(m, x[1:N] >=0);
        JuMP.@variable(m, γ in JuMP.MOI.Parameter(0.0));
        JuMP.@objective(m, Max, γ*LinearAlgebra.dot(μ,x) - x' * Σ * x);
        JuMP.@constraint(m, simplex, sum(x) == 1);
        param_value = [0.1];
        JuMP.set_parameter_value.([γ], param_value);

        cqp_cache = L2ODLL.build_cache(m, L2ODLL.ConvexQP(m));
        cqp_y_pred = randn_like(Dualization._get_dual_variables.(cqp_cache.dual_model, cqp_cache.decomposition.y_ref));
        dobj1 = cqp_cache.dll_layer(cqp_y_pred, param_value)
        dobj, dobj_wrt_y_flat = value_and_gradient(cqp_cache.dll_layer, AutoForwardDiff(), L2ODLL.flatten_y(cqp_y_pred), Constant(param_value))
        dobj_wrt_y = L2ODLL.unflatten_y(dobj_wrt_y_flat, L2ODLL.y_shape(cqp_cache))

        cqp_solver_cache = L2ODLL.build_cache(m, L2ODLL.ConvexQP(m), dll_layer_builder=(d,p,m) -> L2ODLL.poi_builder(d,p,m,SOLVER));
        dobj2 = cqp_solver_cache.dll_layer(cqp_y_pred, param_value)
        @test isapprox(dobj1, dobj2, atol=1e-6)

        JuMP.set_optimizer(m, Clarabel.Optimizer);
        JuMP.optimize!(m)
        cqp_y_true = JuMP.dual.(cqp_cache.decomposition.y_ref)
        dobj1 = cqp_cache.dll_layer(cqp_y_true, param_value)
        dobj2 = cqp_cache.dll_layer(cqp_y_true, param_value)
        @test isapprox(dobj1, dobj2, atol=1e-6) && isapprox(dobj1, JuMP.objective_value(m), atol=1e-6)
    end

    @testset "Markowitz SecondOrderCone" begin
        Σ = [
            166 34 58
            34 64 4
            58 4 100
        ] / 100^2
        N = size(Σ, 1);
        m = JuMP.Model();
        JuMP.@variable(m, x[1:N]);
        JuMP.set_lower_bound.(x, 0);
        JuMP.set_upper_bound.(x, 1); # need to add upper bounds for bound decomposition
        JuMP.@variable(m, μ[1:N] in JuMP.MOI.Parameter.(0.0));
        JuMP.@variable(m, γ in JuMP.MOI.Parameter(0.1));
        JuMP.@objective(m, Max, LinearAlgebra.dot(μ,x));
        JuMP.@constraint(m, simplex, sum(x) == 1);
        JuMP.@constraint(m, risk, [γ; LinearAlgebra.cholesky(Σ).L * x] in JuMP.SecondOrderCone());
        param_value = [randn(N); 0.1];
        JuMP.set_parameter_value.([μ; γ], param_value);

        blp_cache = L2ODLL.build_cache(m, L2ODLL.BoundDecomposition(m));
        blp_y_pred = randn_like(Dualization._get_dual_variables.(blp_cache.dual_model, blp_cache.decomposition.y_ref));
        dobj1 = blp_cache.dll_layer(blp_y_pred, param_value)
        dobj, dobj_wrt_y = value_and_gradient(
            (y,p) ->blp_cache.dll_layer(L2ODLL.unflatten_y(y, L2ODLL.y_shape(blp_cache)),p),
            AutoForwardDiff(),
            L2ODLL.flatten_y(blp_y_pred), Constant(param_value)
        )
        dobj_wrt_y = L2ODLL.unflatten_y(dobj_wrt_y, L2ODLL.y_shape(blp_cache))

        blp_solver_cache = L2ODLL.build_cache(m, L2ODLL.BoundDecomposition(m), dll_layer_builder=(d,p,m) -> L2ODLL.poi_builder(d,p,m,SOLVER));
        dobj2 = blp_solver_cache.dll_layer(blp_y_pred, param_value)
        @test isapprox(dobj1, dobj2, atol=1e-6)

        JuMP.set_optimizer(m, Clarabel.Optimizer);
        JuMP.optimize!(m)
        blp_y_true = JuMP.dual.(blp_cache.decomposition.y_ref)
        dobj1 = blp_cache.dll_layer(blp_y_true, param_value)
        dobj2 = blp_cache.dll_layer(blp_y_true, param_value)
        @test isapprox(dobj1, dobj2, atol=1e-6) && isapprox(dobj1, JuMP.objective_value(m), atol=1e-6)
    end
end