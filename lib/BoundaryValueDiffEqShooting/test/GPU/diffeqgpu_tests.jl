using DiffEqGPU, Logging
include("device_tests.jl")
include("kernel_algorithm_tests.jl")

# DiffEqGPU deliberately warns whenever its kernels run on CPU. Exercise the
# extension in ordinary CI too, without requiring a CUDA installation or device.
with_logger(NullLogger()) do
    device_shooting_tests(CPU(); ode_alg = GPUTsit5())
    kernel_algorithm_tests(CPU())
end
