using InteractiveUtils, SafeTestsets, Test

@info sprint(InteractiveUtils.versioninfo)

const TEST_GROUP = get(ENV, "BOUNDARYVALUEDIFFEQ_TEST_GROUP", "All")

@time begin
    if TEST_GROUP == "Core" || TEST_GROUP == "All"
        @time @safetestset "Shooting Basic Problems Tests" include("basic_problems_tests.jl")
        @time @safetestset "Shooting NLLS Tests" include("nlls_tests.jl")
        @time @safetestset "Shooting Orbital Tests" include("orbital_tests.jl")
    end

    if (TEST_GROUP == "QA" || TEST_GROUP == "All") && isempty(VERSION.prerelease)
        @time @safetestset "Quality Assurance" include("qa_tests.jl")
    end
end
