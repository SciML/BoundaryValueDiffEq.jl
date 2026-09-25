using SafeTestsets, Test
using SciMLTesting

run_tests(;
    env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    core = function ()
        @time @safetestset "Shooting Basic Problems Tests" include("Core/basic_problems_tests.jl")
        @time @safetestset "Shooting NLLS Tests" include("Core/nlls_tests.jl")
        return @time @safetestset "Shooting Orbital Tests" include("Core/orbital_tests.jl")
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
