using Revise

using L2ODLL, Dualization, ParametricOptInterface
using DifferentiationInterface, ForwardDiff

randn_like(vecofvecs) = [randn(length(v)) for v in vecofvecs];
zeros_like(vecofvecs) = [zeros(length(v)) for v in vecofvecs];

# POI helps a lot here since most of the variables become parameters
SOLVER = () -> ParametricOptInterface.Optimizer(HiGHS.Optimizer());

# get a bounded LP
using PGLearn, HiGHS, PowerModels, PGLib
opf = PGLearn.build_opf(
    PGLearn.FullPTDFEconomicDispatch,
    PGLearn.OPFData(make_basic_network(PowerModels.parse_file(
        first(PGLearn._get_case_info(Dict("pglib_case" => "5_pjm")))
    ))),
    HiGHS.Optimizer,
)
pd_value = opf.data.pd;

# general decomposition via solver
cache = L2ODLL.build_cache(opf.model, L2ODLL.GenericDecomposition(opf.model), optimizer=SOLVER);
y_prediction = reduce(vcat, randn_like(Dualization._get_dual_variables.(cache.dual_model, cache.decomposition.y_ref)));
dobj = cache.dll_layer(y_prediction, pd_value)
# need diffopt...
# dobj, dobj_wrt_y = value_and_gradient(cache.dll_layer, AutoForwardDiff(), y_prediction, Constant(opf.data.pd))


# bounded LP special case
blp_cache = L2ODLL.build_cache(opf.model, L2ODLL.BoundDecomposition(opf.model), optimizer=SOLVER);
dobj = blp_cache.dll_layer(y_prediction, pd_value)
# now we can diff through!
dobj, dobj_wrt_y = value_and_gradient(blp_cache.dll_layer, AutoForwardDiff(), y_prediction, Constant(pd_value))



# test convexqp decomposition -- predict everything except the quadratic slack variables
using Clarabel
opf = PGLearn.build_opf(
    PGLearn.SOCOPF,
    PGLearn.OPFData(make_basic_network(PowerModels.parse_file(
        first(PGLearn._get_case_info(Dict("pglib_case" => "5_pjm")))
    ))),
    Clarabel.Optimizer,
)
pd_value = opf.data.pd;
qd_value = opf.data.qd;
param_value = [pd_value; qd_value];

cqp_cache = L2ODLL.build_cache(opf.model, L2ODLL.ConvexQP(opf.model));
cqp_y_pred = zeros_like(Dualization._get_dual_variables.(cqp_cache.dual_model, cqp_cache.decomposition.y_ref));
dobj1 = cqp_cache.dll_layer(cqp_y_pred, param_value)
# vec{vec{T}} not supported by ForwardDiff
# dobj, dobj_wrt_y = value_and_gradient(cqp_cache.dll_layer, AutoForwardDiff(), cqp_y_pred, Constant(pd_value))

# can force use of solver
cqp_solver_cache = L2ODLL.build_cache(opf.model, L2ODLL.ConvexQP(opf.model), dll_layer_builder=(d,p,m) -> L2ODLL.poi_builder(d,p,m,SOLVER));
dobj2 = cqp_solver_cache.dll_layer(cqp_y_pred, param_value)
@assert dobj1 ≈ dobj2

# generic works too
gen_cache = L2ODLL.build_cache(opf.model, L2ODLL.GenericDecomposition(opf.model), optimizer=SOLVER);
gen_y_pred = zeros_like(Dualization._get_dual_variables.(gen_cache.dual_model, gen_cache.decomposition.y_ref));
dobj = gen_cache.dll_layer(gen_y_pred, param_value)
# diffopt needed
# dobj, dobj_wrt_y = value_and_gradient(gen_cache.dll_layer, AutoForwardDiff(), gen_y_pred, Constant(pd_value))


# generate problem data
μ = [11.5; 9.5; 6] / 100          #expected returns
Σ = [
    166 34 58              #covariance matrix
    34 64 4
    58 4 100
] / 100^2

m = Model()
N = length(μ)
@variable(m, x[1:N] >=0)
@variable(m, γ in MOI.Parameter(0.0))

@objective(m, Max, γ*dot(μ,x) - x' * Σ * x)
@constraint(m, simplex, sum(x) == 1)
param_value = 0.1

cqp_cache = L2ODLL.build_cache(m, L2ODLL.ConvexQP(m));
cqp_y_pred = randn_like(Dualization._get_dual_variables.(cqp_cache.dual_model, cqp_cache.decomposition.y_ref));
dobj1 = cqp_cache.dll_layer(cqp_y_pred, param_value)
# vec{vec{T}} not supported by ForwardDiff
# dobj, dobj_wrt_y = value_and_gradient(cqp_cache.dll_layer, AutoForwardDiff(), cqp_y_pred, Constant(pd_value))

# test with solver
cqp_solver_cache = L2ODLL.build_cache(m, L2ODLL.ConvexQP(m), dll_layer_builder=(d,p,m) -> L2ODLL.poi_builder(d,p,m,SOLVER));
dobj2 = cqp_solver_cache.dll_layer(cqp_y_pred, param_value)
@assert dobj1 ≈ dobj2

m = Model()
@variable(m, x[1:N])
set_lower_bound.(x, 0)
set_upper_bound.(x, 1) # need to add upper bounds for bound decomposition
@variable(m, μ[1:N] in MOI.Parameter.(0.0))
@variable(m, γ in MOI.Parameter(0.1))

@objective(m, Max, dot(μ,x))
@constraint(m, simplex, sum(x) == 1)
# @constraint(m, risk, x' * Σ * x <= γ)
L = cholesky(Σ).L
@constraint(m, risk, [γ; L * x] in SecondOrderCone())
blp_cache = L2ODLL.build_cache(m, L2ODLL.BoundDecomposition(m));
blp_y_pred = randn_like(Dualization._get_dual_variables.(blp_cache.dual_model, blp_cache.decomposition.y_ref));
dobj1 = blp_cache.dll_layer(blp_y_pred, param_value)
# vec{vec{T}} not supported by ForwardDiff
# dobj, dobj_wrt_y = value_and_gradient(blp_cache.dll_layer, AutoForwardDiff(), blp_y_pred, Constant(pd_value))

# test with solver
blp_solver_cache = L2ODLL.build_cache(m, L2ODLL.BoundDecomposition(m), dll_layer_builder=(d,p,m) -> L2ODLL.poi_builder(d,p,m,SOLVER));
dobj2 = blp_solver_cache.dll_layer(blp_y_pred, param_value)
@assert dobj1 ≈ dobj2
