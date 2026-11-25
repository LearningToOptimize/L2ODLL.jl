# # Markowitz portfolio (second-order cone)
#
# A risk-constrained portfolio problem: maximize expected return subject to a
# budget and a second-order cone bound on the standard deviation,
#
# ```math
# \max_x\; \mu^\top x \quad \text{s.t.}\quad \mathbf{1}^\top x = 1,\;
# (\gamma, L^\top x) \in \mathcal{Q},\; 0 \le x \le 1,
# ```
#
# with the expected returns ``\mu`` and the risk budget ``\gamma`` as parameters
# (``\Sigma = L L^\top``). Every variable is bounded and the objective is linear,
# so L2ODLL uses the `BoundDecomposition` (Example 1 of Tanneau and Van
# Hentenryck, 2024): the proxy predicts the duals of the budget and of the cone,
# and the bound duals are completed in closed form.

using JuMP, LinearAlgebra, Random
using Clarabel: Clarabel
using L2ODLL

Σ = [166 34 58; 34 64 4; 58 4 100] / 100^2
N = size(Σ, 1)

model = Model()
@variable(model, 0 <= x[1:N] <= 1)
@variable(model, μ[1:N] in MOI.Parameter.(0.0))
@variable(model, γ in MOI.Parameter(0.1))
@constraint(model, budget, sum(x) == 1)
@constraint(model, risk, [γ; cholesky(Σ).L * x] in SecondOrderCone())
@objective(model, Max, dot(μ, x))

cache = decompose!(model)
cache.decomposition

# The predicted dual variables: one for the budget and four for the cone.

get_y(model), y_shape(model)

# ## Any prediction gives a valid bound
#
# For a maximization problem the dual objective is an **upper** bound. We draw
# parameters, solve the problem for reference, and evaluate random predictions.

rng = Xoshiro(42)
pv = [0.1 .* randn(rng, N); 0.1]
set_parameter_value.([μ; γ], pv)
set_optimizer(model, Clarabel.Optimizer);
set_silent(model);
optimize!(model)
opt = objective_value(model)

bounds = [dual_objective(model, [randn(rng, n) for n in y_shape(model)], pv) for _ in 1:5]
all(bounds .>= opt - 1e-8)

# The optimal dual solution recovers the optimum (strong duality):

ystar = dual.(get_y(model))
dual_objective(model, ystar, pv), opt

# The completed dual variables are available as well:

dual_completion(model, ystar, pv).z

# ## Gradients and batches
#
# The gradient with respect to the prediction (ForwardDiff by default) and the
# matrix forms, with the batch as the last dimension:

B = 8
P = reduce(hcat, [[0.1 .* randn(rng, N); 0.05 + 0.1rand(rng)] for _ in 1:B])
Y = randn(rng, sum(y_shape(model)), B)
vals, G = dual_objective_and_gradient(model, Y, P)
size(G)
