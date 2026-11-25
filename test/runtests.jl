using L2ODLL
using Test

import JuMP, Dualization, ParametricOptInterface
import Clarabel, HiGHS
import DifferentiationInterface, ForwardDiff
import LinearAlgebra
import Random

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
