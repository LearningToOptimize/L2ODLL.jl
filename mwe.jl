using Revise

using L2ODLL
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



# general decomposition via solver
decomposition = L2ODLL.GenericDecomposition(opf.model)
# pre-build dll_layer (POI) and y_proj (MOSD) functions
cache = L2ODLL.build_cache(opf.model, decomposition, optimizer=HiGHS.Optimizer)

y_prediction = randn(length(cache.decomposition.y_ref))
pd_value = opf.data.pd
dobj = cache.dll_layer(y_prediction, pd_value)
# need diffopt...
# dobj, dobj_wrt_y = value_and_gradient(cache.dll_layer, AutoForwardDiff(), y_prediction, Constant(opf.data.pd))


# bounded LP special case
# pre-build dll_layer (closed form / no solver) and y_proj (MOSD) functions
blp_cache = L2ODLL.build_cache(opf.model, L2ODLL.BoundDecomposition(opf.model))

y_prediction = randn(length(blp_cache.decomposition.y_ref))
pd_value = opf.data.pd
dobj = blp_cache.dll_layer(y_prediction, pd_value)
# now we can diff through!
dobj, dobj_wrt_y = value_and_gradient(blp_cache.dll_layer, AutoForwardDiff(), y_prediction, Constant(pd_value))

