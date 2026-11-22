using DiffEqGPU, Logging
include("device_tests.jl")
include("kernel_algorithm_tests.jl")

# DiffEqGPU deliberately warns whenever its kernels run on CPU. Exercise the
# extension in ordinary CI too, without requiring a CUDA installation or device.
with_logger(NullLogger()) do
    test_device_jacobians(CPU(), GPUTsit5())
    kernel_algorithm_tests(CPU())
end
