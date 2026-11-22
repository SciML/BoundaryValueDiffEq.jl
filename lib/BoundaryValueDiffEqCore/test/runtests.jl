using SafeTestsets, Test
using SciMLTesting

run_tests(;
    env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    core = function ()
        @time @safetestset "Utility Tests" include("Core/util_tests.jl")
        return @time @safetestset "Internal Solver Selection" include("Core/internal_solver_tests.jl")
    end,
    groups = Dict(
        "GPU" => (;
            env = joinpath(@__DIR__, "GPU"),
            body = function ()
                @time @safetestset "Core CUDA Storage" include("GPU/cuda_tests.jl")
                return @time @safetestset "Core CUDSS Solvers" include("GPU/cudss_tests.jl")
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
    all = ["Core", "QA"],
)
