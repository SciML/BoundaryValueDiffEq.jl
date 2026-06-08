using Pkg
using InteractiveUtils, SafeTestsets, Test

@info sprint(InteractiveUtils.versioninfo)

const TEST_GROUP = get(ENV, "BOUNDARYVALUEDIFFEQ_TEST_GROUP", "All")

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    # On Julia < 1.11, the [sources] section in Project.toml is not honored.
    # Manually Pkg.develop the local path dependencies so QA tests the PR branch code.
    if VERSION < v"1.11.0-DEV.0"
        Pkg.develop([
            Pkg.PackageSpec(path = joinpath(@__DIR__, "..")),
            Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "BoundaryValueDiffEqCore"))
        ])
    end
    return Pkg.instantiate()
end

@time begin
    if TEST_GROUP == "Core"
        # Representative light FIRK set covering both formulations' basic solves.
        # Targeted by the uniform "Core" downgrade value. The full EXPANDED/NESTED
        # groups below already include these basic tests, and "All" runs those
        # groups, so "Core" is kept separate from "All" to avoid double execution.
        @time @safetestset "FIRK Expanded Basic Tests" include("expanded/firk_basic_tests.jl")
        @time @safetestset "FIRK Nested Basic Tests" include("nested/firk_basic_tests.jl")
    end

    if TEST_GROUP == "EXPANDED" || TEST_GROUP == "All"
        @time @safetestset "FIRK Expanded Basic Tests" include("expanded/firk_basic_tests.jl")
        @time @safetestset "FIRK Expanded NLLS Tests" include("expanded/nlls_tests.jl")
        @time @safetestset "FIRK Expanded AD Tests" include("expanded/ad_tests.jl")
        @time @safetestset "FIRK Expanded Ensemble Tests" include("expanded/ensemble_tests.jl")
        @time @safetestset "FIRK Expanded Singular BVP Tests" include("expanded/singular_bvp_tests.jl")
        @time @safetestset "FIRK Expanded VectorOfVector Initials Tests" include("expanded/vectorofvector_initials_tests.jl")
    end

    if TEST_GROUP == "NESTED" || TEST_GROUP == "All"
        @time @safetestset "FIRK Nested Basic Tests" include("nested/firk_basic_tests.jl")
        @time @safetestset "FIRK Nested NLLS Tests" include("nested/nlls_tests.jl")
        @time @safetestset "FIRK Nested Ensemble Tests" include("nested/ensemble_tests.jl")
        @time @safetestset "FIRK Nested VectorOfVector Initials Tests" include("nested/vectorofvector_initials_tests.jl")
    end

    if (TEST_GROUP == "QA" || TEST_GROUP == "All") && isempty(VERSION.prerelease)
        activate_qa_env()
        @time @safetestset "Quality Assurance" include("qa/qa.jl")
    end
end
