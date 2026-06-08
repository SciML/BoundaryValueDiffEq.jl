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
    if TEST_GROUP == "Core" || TEST_GROUP == "All"
        @time @safetestset "MIRK Basic Tests" include("mirk_basic_tests.jl")
        @time @safetestset "MIRK NLLS Tests" include("nlls_tests.jl")
        @time @safetestset "MIRK Ensemble Tests" include("ensemble_tests.jl")
        @time @safetestset "MIRK AD Tests" include("ad_tests.jl")
        @time @safetestset "MIRK Singular BVP Tests" include("singular_bvp_tests.jl")
        @time @safetestset "MIRK VectorOfVector Initials Tests" include("vectorofvector_initials_tests.jl")
        @time @safetestset "MIRK Dynamic Optimization Tests" include("dynamic_optimization_tests.jl")
    end

    if (TEST_GROUP == "QA" || TEST_GROUP == "All") && isempty(VERSION.prerelease)
        activate_qa_env()
        @time @safetestset "Quality Assurance" include("qa/qa.jl")
    end
end
