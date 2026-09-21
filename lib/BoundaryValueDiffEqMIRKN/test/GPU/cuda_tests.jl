using CUDA, CUDSS, Test
CUDA.functional() || error("MIRKN CUDA tests require a functional CUDA device.")
CUDA.allowscalar(false)
include("device_backend_tests.jl")
@testset "MIRKN CUDA" begin
    test_device_backend(CuArray, CUDABackend(); gpu = true)
    test_device_features(CuArray, CUDABackend(); gpu = true)
    test_hybrid(CUDABackend())
    include("regression_tests.jl")
end

include("resizing_tests.jl")
@testset "MIRKN CUDA flat buffer resizing" test_mirkn_flat_buffers(CuArray, CUDA.CUDABackend())
