using CUDA, Test
CUDA.functional() || error("CUDA tests require a functional CUDA device.")
CUDA.allowscalar(false)
include("device_tests.jl")
device_shooting_tests(CUDA.CUDABackend(); gpu = true)
include("cudss_tests.jl")
