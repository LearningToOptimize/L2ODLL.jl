using L2ODLL
using Test

using JSON, HTTP
function test_vectorize_bridge_matches_moi()
    url = "https://api.github.com/repos/jump-dev/MathOptInterface.jl/commits?path=src/Bridges/Constraint/bridges/VectorizeBridge.jl"
    response = HTTP.get(url)
    data = JSON.parse(String(response.body))
    @test data[1]["sha"] == "4e2630554afcde0b0b7c3e680e7fd3666e9e3825"
end

@testset "L2ODLL.jl" begin
    test_vectorize_bridge_matches_moi()
end