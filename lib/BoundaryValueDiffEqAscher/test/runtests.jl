using SafeTestsets, Test
using SciMLTesting

run_tests(;
    env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    core = function ()
        return @time @safetestset "Ascher Basic Tests" include("Core/ascher_basic_tests.jl")
    end,
    groups = Dict(
        "DeviceKernels" => function ()
            @time @safetestset "Ascher flat buffer resizing" begin
                include("GPU/resizing_tests.jl")
                test_ascher_flat_buffers(identity, CPU())
            end
            return @time @safetestset "Ascher device formulation" include("GPU/device_tests.jl")
        end,
        "GPU" => (;
            env = joinpath(@__DIR__, "GPU"),
            body = function ()
                return @time @safetestset "Ascher CUDA" include("GPU/cuda_tests.jl")
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
