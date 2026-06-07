using InteractiveUtils, SafeTestsets, Test

@info sprint(InteractiveUtils.versioninfo)

const TEST_GROUP = get(ENV, "BOUNDARYVALUEDIFFEQ_TEST_GROUP", "All")

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
        @time @safetestset "Quality Assurance" include("qa_tests.jl")
    end
end
