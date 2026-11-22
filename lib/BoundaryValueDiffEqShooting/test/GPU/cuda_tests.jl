using CUDA, DiffEqGPU, Test
CUDA.functional() || error("CUDA tests require a functional CUDA device.")
CUDA.allowscalar(false)
include("device_tests.jl")
test_device_jacobians(CUDA.CUDABackend(), GPUTsit5())
include("kernel_algorithm_tests.jl")
kernel_algorithm_tests(CUDA.CUDABackend())
include("cudss_tests.jl")
