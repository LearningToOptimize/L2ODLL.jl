using L2ODLL
using Test

using JuMP: JuMP
using Dualization: Dualization
using ParametricOptInterface: ParametricOptInterface
using Clarabel: Clarabel
using HiGHS: HiGHS
using DifferentiationInterface: DifferentiationInterface
using ForwardDiff: ForwardDiff
using LinearAlgebra: LinearAlgebra
using Random: Random

const MOI = JuMP.MOI

include("instances.jl")

@testset "L2ODLL.jl" begin
    include("markowitz.jl")
    include("utilities.jl")
    include("regressions.jl")
    include("guarantees.jl")
    include("audit.jl")
    include("l2ocore.jl")
    include("qa.jl")
end
