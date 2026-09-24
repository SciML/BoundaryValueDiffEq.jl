using SafeTestsets, Test
using SciMLTesting

run_tests(;
    env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    core = function ()
        return @time @safetestset "MIRKN Basic Tests" include("Core/mirkn_basic_tests.jl")
    end,
    groups = Dict(
        "DeviceKernels" => function ()
            @time @safetestset "MIRKN flat buffer resizing" begin
                include("GPU/resizing_tests.jl")
                test_mirkn_flat_buffers(identity, CPU())
            end
            @time @safetestset "MIRKN Packed Kernels and Sparse AD" include("GPU/device_kernel_tests.jl")
            return @time @safetestset "MIRKN Resident Pipeline on CPU" begin
                include("GPU/device_backend_tests.jl")
                test_device_backend(identity, CPU())
                test_device_features(identity, CPU())
            end
        end,
        "GPU" => (;
            env = joinpath(@__DIR__, "GPU"),
            body = function ()
                return @time @safetestset "MIRKN CUDA" include("GPU/cuda_tests.jl")
            end,
        ),
    ),
    qa = (;
        env = joinpath(@__DIR__, "qa"),
        body = function ()
            # QA (Aqua) runs on release + LTS Julia only; skip on prerelease.
            isempty(VERSION.prerelease) || return nothing
            return @time @safetestset "Quality Assurance" include("qa/qa.jl")
        end,
    ),
    all = ["Core", "DeviceKernels", "QA"],
)
