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
        @test_throws Union{ArgumentError,MethodError} L2ODLL.flatten_y(empty_vec)  # MethodError on Julia 1.10
        
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
