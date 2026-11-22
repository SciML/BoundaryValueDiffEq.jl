using BoundaryValueDiffEqCore, CUDA, Test

CUDA.functional() || error("The Core GPU test group requires a functional CUDA device.")
CUDA.allowscalar(false)

include("device_sparse_tests.jl")

@testset "Core CUDA sparse storage" begin
    @test Base.get_extension(BoundaryValueDiffEqCore, :BoundaryValueDiffEqCoreCUDAExt) !== nothing
    test_device_sparse_storage(CuArray)
    template = CUDA.zeros(Float64, 2, 3)
    storage = __device_sparse_matrix(template, sparse([1, 2], [2, 1], trues(2), 2, 3))
    @test CUDA.device(nonzeros(storage.matrix)) == CUDA.device(template)

    if length(CUDA.devices()) > 1
        other = first(filter(!=(CUDA.device(template)), collect(CUDA.devices())))
        CUDA.device!(other) do
            storage = __device_sparse_matrix(template, spzeros(2, 3))
            @test CUDA.device(nonzeros(storage.matrix)) == CUDA.device(template)
            @test CUDA.device() == other
        end
    end
end
