using BoundaryValueDiffEqCore: __device_sparse_linsolve
using SparseArrays, Test

@testset "Optional device linear solver fallback" begin
    A = sparse([2.0 1.0; 1.0 3.0])
    @test __device_sparse_linsolve(A) === nothing
end
