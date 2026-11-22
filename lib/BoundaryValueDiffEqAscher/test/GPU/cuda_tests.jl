using CUDA, Test
CUDA.functional() || error("Ascher CUDA tests require a functional NVIDIA GPU.")
CUDA.allowscalar(false)

include("device_tests.jl")

test_ascher_device_jacobians(CuArray, CUDA.CUDABackend())
