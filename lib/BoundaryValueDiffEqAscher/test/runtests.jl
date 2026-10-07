using SafeTestsets, Test
using SciMLTesting

run_tests(;
    env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    core = function ()
        @time @safetestset "Ascher Basic Tests" include("Core/ascher_basic_tests.jl")
        @time @safetestset "Ascher DAE Benchmarks" include("Core/dae_tests.jl")
        return @time @safetestset "Ascher Sparse Collocation" include("Core/performance_tests.jl")
    end,
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
