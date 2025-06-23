using L2ODLL
using Test

import JuMP, Dualization, ParametricOptInterface
import Clarabel, HiGHS
import DifferentiationInterface, ForwardDiff
import LinearAlgebra

randn_like(vecofvecs) = [randn(length(v)) for v in vecofvecs];
SOLVER = () -> ParametricOptInterface.Optimizer(HiGHS.Optimizer());

@testset "L2ODLL.jl" begin
    @testset "Markowitz Frontier (ConvexQP)" begin
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

        L2ODLL.decompose!(m);

        cqp_y_pred = randn_like(L2ODLL.get_y_dual(m));

        dobj = L2ODLL.dual_objective(m, cqp_y_pred, param_value)
        dobj_wrt_y = L2ODLL.dual_objective_gradient(m, cqp_y_pred, param_value)

        m2 = JuMP.Model();
        JuMP.@variable(m2, x[1:N] >=0);
        JuMP.@variable(m2, γ in JuMP.MOI.Parameter(0.0));
        JuMP.@objective(m2, Max, γ*LinearAlgebra.dot(μ,x) - x' * Σ * x);
        JuMP.@constraint(m2, simplex, sum(x) == 1);
        JuMP.set_parameter_value.([γ], param_value);
        L2ODLL.decompose!(m2, L2ODLL.ConvexQP(m2), dll_layer_builder=(d,p,m) -> L2ODLL.jump_builder(d,p,m,SOLVER));
        dobj2 = L2ODLL.dual_objective(m2, cqp_y_pred, param_value)

        JuMP.set_optimizer(m, Clarabel.Optimizer);
        JuMP.set_silent(m);
        JuMP.optimize!(m)
        cqp_y_true = JuMP.dual.(L2ODLL.get_y(m))
        dobj1 = L2ODLL.dual_objective(m, cqp_y_true, param_value)
        @test isapprox(dobj1, JuMP.objective_value(m), atol=1e-6)

        batch_size = 3
        param_values = [[0.1], [0.2], [0.3]]
        y_predicted_batch = [randn_like(L2ODLL.get_y_dual(m)) for _ in 1:batch_size]

        dobj_batch = L2ODLL.dual_objective.(m, y_predicted_batch, param_values)
        dobj_grad_batch = L2ODLL.dual_objective_gradient.(m, y_predicted_batch, param_values)
        
        @test length(dobj_batch) == batch_size
        @test length(dobj_grad_batch) == batch_size
        
        for i in 1:batch_size
            individual_dobj = L2ODLL.dual_objective(m, y_predicted_batch[i], param_values[i])
            individual_dobj_grad = L2ODLL.dual_objective_gradient(m, y_predicted_batch[i], param_values[i])
            @test dobj_batch[i] ≈ individual_dobj atol=1e-6
            @test all(isapprox.(dobj_grad_batch[i], individual_dobj_grad, atol=1e-10))
        end
    end

    @testset "Markowitz SecondOrderCone (BoundDecomposition)" begin
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
        JuMP.@constraint(m, simplex, sum(x) == 1);
        JuMP.@constraint(m, risk, [γ; LinearAlgebra.cholesky(Σ).L * x] in JuMP.SecondOrderCone());
        JuMP.@objective(m, Max, LinearAlgebra.dot(μ,x));
        param_value = [randn(N); 0.1];
        JuMP.set_parameter_value.([μ; γ], param_value);

        L2ODLL.decompose!(m);

        blp_y_pred = randn_like(L2ODLL.get_y_dual(m));

        dobj = L2ODLL.dual_objective(m, blp_y_pred, param_value)
        dobj_wrt_y = L2ODLL.dual_objective_gradient(m, blp_y_pred, param_value)

        m2 = JuMP.Model();
        JuMP.@variable(m2, x[1:N]);
        JuMP.set_lower_bound.(x, 0);
        JuMP.set_upper_bound.(x, 1);
        JuMP.@variable(m2, μ[1:N] in JuMP.MOI.Parameter.(0.0));
        JuMP.@variable(m2, γ in JuMP.MOI.Parameter(0.1));
        JuMP.@constraint(m2, simplex, sum(x) == 1);
        JuMP.@constraint(m2, risk, [γ; LinearAlgebra.cholesky(Σ).L * x] in JuMP.SecondOrderCone());
        JuMP.@objective(m2, Max, LinearAlgebra.dot(μ,x));
        JuMP.set_parameter_value.([μ; γ], param_value);
        
        L2ODLL.decompose!(m2, L2ODLL.BoundDecomposition(m2), dll_layer_builder=(d,p,m) -> L2ODLL.jump_builder(d,p,m,SOLVER));
        dobj2 = L2ODLL.dual_objective(m2, blp_y_pred, param_value)

        JuMP.set_optimizer(m, Clarabel.Optimizer);
        JuMP.set_silent(m);
        JuMP.optimize!(m)
        blp_y_true = JuMP.dual.(L2ODLL.get_y(m))

        dobj1 = L2ODLL.dual_objective(m, blp_y_true, param_value)
        @test isapprox(dobj1, JuMP.objective_value(m), atol=1e-6)

        batch_size = 3
        param_values = [[randn(N); rand()] for _ in 1:batch_size]
        y_predicted_batch = [randn_like(L2ODLL.get_y_dual(m)) for _ in 1:batch_size]

        dobj_batch = L2ODLL.dual_objective.(m, y_predicted_batch, param_values)
        dobj_grad_batch = L2ODLL.dual_objective_gradient.(m, y_predicted_batch, param_values)
        
        @test length(dobj_batch) == batch_size
        @test length(dobj_grad_batch) == batch_size
        
        for i in 1:batch_size
            individual_dobj = L2ODLL.dual_objective(m, y_predicted_batch[i], param_values[i])
            individual_dobj_grad = L2ODLL.dual_objective_gradient(m, y_predicted_batch[i], param_values[i])
            @test dobj_batch[i] ≈ individual_dobj atol=1e-6
            @test all(isapprox.(dobj_grad_batch[i], individual_dobj_grad, atol=1e-10))
        end
    end

    @testset "Utilities" begin
        @testset "flatten_y" begin
            y_vec = [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]]
            y_flat = L2ODLL.flatten_y(y_vec)
            @test y_flat == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            
            y_shape = [2, 3, 1]
            y_unflat = L2ODLL.unflatten_y(y_flat, y_shape)
            @test y_unflat == y_vec
            
            original = [[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]]
            flattened = L2ODLL.flatten_y(original)
            shape = length.(original)
            reconstructed = L2ODLL.unflatten_y(flattened, shape)
            @test reconstructed == original
            
            single_vec = [[1.0, 2.0, 3.0]]
            single_flat = L2ODLL.flatten_y(single_vec)
            @test single_flat == [1.0, 2.0, 3.0]
            single_unflat = L2ODLL.unflatten_y(single_flat, [3])
            @test single_unflat == single_vec
            
            empty_vec = Vector{Float64}[]
            @test_throws ArgumentError("reducing over an empty collection is not allowed; consider supplying `init` to the reducer") L2ODLL.flatten_y(empty_vec)
            
            empty_flat = Float64[]
            empty_unflat = L2ODLL.unflatten_y(empty_flat, Int[])
            @test empty_unflat == Vector{Float64}[]
            
            mixed_vec = [[1.0, 2.0], Float64[], [3.0]]
            mixed_flat = L2ODLL.flatten_y(mixed_vec)
            @test mixed_flat == [1.0, 2.0, 3.0]
            mixed_shape = [2, 0, 1]
            mixed_unflat = L2ODLL.unflatten_y(mixed_flat, mixed_shape)
            @test mixed_unflat == mixed_vec
            
            int_vec = [[1, 2], [3, 4, 5]]
            int_flat = L2ODLL.flatten_y(int_vec)
            @test int_flat == [1, 2, 3, 4, 5]
            int_unflat = L2ODLL.unflatten_y(int_flat, [2, 3])
            @test int_unflat == int_vec
            
            mismatched_flat = [1.0, 2.0, 3.0]  # length 3
            mismatched_shape = [2, 3]  # expects length 5
            @test_throws BoundsError L2ODLL.unflatten_y(mismatched_flat, mismatched_shape)
        end
    end
end