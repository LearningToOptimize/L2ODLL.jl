using Revise

using L2ODLL, Dualization
using DifferentiationInterface, ForwardDiff

# get a bounded LP
using PGLearn, HiGHS, PowerModels, PGLib
opf = PGLearn.build_opf(
    PGLearn.FullPTDFEconomicDispatch,
    PGLearn.OPFData(make_basic_network(PowerModels.parse_file(
        first(PGLearn._get_case_info(Dict("pglib_case" => "5_pjm")))
    ))),
    HiGHS.Optimizer,
)

function randn_like(vecofvecs::Vector{Vector{T}}) where T
    # TODO: forwarddiff can't do wrt vec{vec{T}}, but need for conic support
    return [randn(length(v)) for v in vecofvecs]
end



# general decomposition via solver
decomposition = L2ODLL.GenericDecomposition(opf.model)
# pre-build dll_layer (POI) and y_proj (MOSD) functions
cache = L2ODLL.build_cache(opf.model, decomposition, optimizer=HiGHS.Optimizer)

y_prediction = reduce(vcat, randn_like(Dualization._get_dual_variables.(cache.dual_model, cache.decomposition.y_ref)))
pd_value = opf.data.pd
dobj = cache.dll_layer(y_prediction, pd_value)
# need diffopt...
# dobj, dobj_wrt_y = value_and_gradient(cache.dll_layer, AutoForwardDiff(), y_prediction, Constant(opf.data.pd))


# bounded LP special case
# pre-build dll_layer (closed form / no solver) and y_proj (MOSD) functions
blp_cache = L2ODLL.build_cache(opf.model, L2ODLL.BoundDecomposition(opf.model), optimizer=HiGHS.Optimizer)

y_prediction = reduce(vcat, randn_like(Dualization._get_dual_variables.(blp_cache.dual_model, blp_cache.decomposition.y_ref)))
pd_value = opf.data.pd
dobj = blp_cache.dll_layer(y_prediction, pd_value)
# now we can diff through!
dobj, dobj_wrt_y = value_and_gradient(blp_cache.dll_layer, AutoForwardDiff(), y_prediction, Constant(pd_value))




using Clarabel
opf = PGLearn.build_opf(
    PGLearn.SOCOPF,
    PGLearn.OPFData(make_basic_network(PowerModels.parse_file(
        first(PGLearn._get_case_info(Dict("pglib_case" => "5_pjm")))
    ))),
    Clarabel.Optimizer,
)

cqp_cache = L2ODLL.build_cache(opf.model, L2ODLL.ConvexQP(opf.model))

y_prediction = randn_like(Dualization._get_dual_variables.(cqp_cache.dual_model, cqp_cache.decomposition.y_ref))
pd_value = opf.data.pd
qd_value = opf.data.qd
param_value = [pd_value; qd_value]
dobj = cqp_cache.dll_layer(y_prediction, param_value)
# vec{vec{T}} not supported by ForwardDiff
# dobj, dobj_wrt_y = value_and_gradient(cqp_cache.dll_layer, AutoForwardDiff(), y_prediction, Constant(pd_value))

gen_cache = L2ODLL.build_cache(opf.model, L2ODLL.GenericDecomposition(opf.model), optimizer=Clarabel.Optimizer)

y_prediction = randn_like(Dualization._get_dual_variables.(gen_cache.dual_model, gen_cache.decomposition.y_ref))
pd_value = opf.data.pd
qd_value = opf.data.qd
param_value = [pd_value; qd_value]
dobj = gen_cache.dll_layer(y_prediction, param_value)
# diffopt needed
# dobj, dobj_wrt_y = value_and_gradient(gen_cache.dll_layer, AutoForwardDiff(), y_prediction, Constant(pd_value))
