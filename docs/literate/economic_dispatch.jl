# # Economic dispatch: training a dual proxy
#
# Generators ``g = 1, \dots, G`` with linear costs ``c_g`` and capacities
# ``\overline{p}_g`` must meet a demand ``d`` and keep a spinning reserve ``r``:
#
# ```math
# \min_{p}\; c^\top p \quad\text{s.t.}\quad \mathbf{1}^\top p = d,\quad
# \mathbf{1}^\top(\overline{p} - p) \ge r,\quad 0 \le p \le \overline{p}.
# ```
#
# ``(d, r)`` are the parameters. The dual variables of the two coupling
# constraints are the energy and reserve prices. We train a small linear proxy
# ``\hat y = W \,[d; r] + w_0`` to predict them, **self-supervised**: the loss is
# the (negated) completed dual objective (eq. (27a) of Tanneau and Van Hentenryck,
# 2024), so no optimal solutions are needed for training, and every prediction
# along the way is a valid lower bound on the dispatch cost.

using JuMP, Random, Statistics
import HiGHS
using L2ODLL

rng = Xoshiro(1)
G = 10
cost = sort(10 .+ 40 .* rand(rng, G))
pmax = 50 .+ 100 .* rand(rng, G)

model = Model()
@variable(model, 0 <= p[g = 1:G] <= pmax[g])
@variable(model, d in MOI.Parameter(500.0))
@variable(model, r in MOI.Parameter(50.0))
@constraint(model, balance, sum(p) == d)
@constraint(model, reserve, sum(pmax[g] - p[g] for g in 1:G) >= r)
@objective(model, Min, cost' * p)
decompose!(model)
y_shape(model)

# Parameters: demand between 30% and 80% of capacity, reserve 5–10% of demand.

function sample_params(rng, n)
    d = (0.3 .+ 0.5 .* rand(rng, n)) .* sum(pmax)
    return permutedims(hcat(d, (0.05 .+ 0.05 .* rand(rng, n)) .* d))
end
Ptrain = sample_params(rng, 256)
Ptest = sample_params(rng, 64)

# Reference optima (only to report the gap on the test set):

set_optimizer(model, HiGHS.Optimizer); set_silent(model)
function optimum(model, pv)
    set_parameter_value(d, pv[1]); set_parameter_value(r, pv[2])
    optimize!(model)
    return objective_value(model)
end
opt_test = [optimum(model, Ptest[:, k]) for k in axes(Ptest, 2)]

# A linear proxy on scaled inputs and plain gradient ascent on the mean bound.
# `dual_objective_and_gradient(model, Y, P)` returns the bound and its gradient
# with respect to the prediction for every column.

scale = sum(pmax)
W = zeros(2, 2); w0 = zeros(2)
proxy(W, w0, P) = W * (P ./ scale) .+ w0
gap(W, w0) = mean((opt_test .- dual_objective(model, proxy(W, w0, Ptest), Ptest)) ./ opt_test)

history = Float64[]
for epoch in 1:300
    Y = proxy(W, w0, Ptrain)
    _, Gy = dual_objective_and_gradient(model, Y, Ptrain)    # ∂Φ/∂ŷ, 2 × 256
    global W += 0.5 .* Gy * (Ptrain ./ scale)' ./ size(Ptrain, 2) ./ scale
    global w0 += 0.5 .* vec(mean(Gy; dims = 2)) ./ scale
    push!(history, gap(W, w0))
end
history[[1, 10, 100, 300]]

# The relative gap on held-out instances decreases from about 100% to about 10%.
# The optimal prices are piecewise constant in ``(d, r)`` (they jump at the merit
# order), which a linear proxy cannot represent; a neural network trained with the
# same loss does better. Whatever the proxy, every bound is valid:

bounds = dual_objective(model, proxy(W, w0, Ptest), Ptest)
all(bounds .<= opt_test .+ 1e-6)

# ## The same through L2OCore
#
# With `import L2OCore` the method plugs into the shared LearningToOptimize
# pipeline, which checks the `ValidDualBound` guarantee independently (its own
# Lagrangian dual function of the completed duals, against the solver optima):

import L2OCore
problem = L2OCore.ParametricProblem(model)
pipe = L2OCore.L2OPipeline(problem, DLLMethod())
pred = L2OCore.complete(pipe, Ptest, proxy(W, w0, Ptest))
L2OCore.check(L2OCore.ValidDualBound(), problem, Ptest, pred; reference = opt_test)

# and its loss is the negated mean bound used above:

L2OCore.loss(pipe, Ptest, proxy(W, w0, Ptest)) ≈ -mean(bounds)
