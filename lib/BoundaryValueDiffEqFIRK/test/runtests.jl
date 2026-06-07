using InteractiveUtils, SafeTestsets, Test

@info sprint(InteractiveUtils.versioninfo)

const TEST_GROUP = get(ENV, "BOUNDARYVALUEDIFFEQ_TEST_GROUP", "All")

@time begin
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
        @time @safetestset "Quality Assurance" include("qa_tests.jl")
    end
end
