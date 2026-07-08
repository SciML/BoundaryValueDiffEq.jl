using SafeTestsets, Test
using SciMLTesting

run_tests(;
    env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    # Core: representative light FIRK set covering both formulations' basic solves.
    # Targeted by the uniform "Core" downgrade value. The full EXPANDED/NESTED groups
    # already include these basic tests, and "All" runs those groups, so "Core" is
    # kept out of "All" (see the `all` list below) to avoid double execution.
    core = function ()
        @time @safetestset "FIRK Expanded Basic Tests" include("expanded/firk_basic_tests.jl")
        return @time @safetestset "FIRK Nested Basic Tests" include("nested/firk_basic_tests.jl")
    end,
    groups = Dict(
        "EXPANDED" => function ()
            @time @safetestset "FIRK Expanded Basic Tests" include("expanded/firk_basic_tests.jl")
            @time @safetestset "FIRK Expanded NLLS Tests" include("expanded/nlls_tests.jl")
            @time @safetestset "FIRK Expanded Ensemble Tests" include("expanded/ensemble_tests.jl")
            @time @safetestset "FIRK Expanded Singular BVP Tests" include("expanded/singular_bvp_tests.jl")
            @time @safetestset "FIRK Expanded DAE Tests" include("expanded/dae_tests.jl")
            return @time @safetestset "FIRK Expanded VectorOfVector Initials Tests" include("expanded/vectorofvector_initials_tests.jl")
        end,
        "NESTED" => function ()
            @time @safetestset "FIRK Nested Basic Tests" include("nested/firk_basic_tests.jl")
            @time @safetestset "FIRK Nested NLLS Tests" include("nested/nlls_tests.jl")
            @time @safetestset "FIRK Nested Ensemble Tests" include("nested/ensemble_tests.jl")
            @time @safetestset "FIRK Nested DAE Tests" include("nested/dae_tests.jl")
            return @time @safetestset "FIRK Nested VectorOfVector Initials Tests" include("nested/vectorofvector_initials_tests.jl")
        end,
        # AD: the different-AD-backend compatibility tests. Enzyme and Mooncake are
        # heavy optional backends kept out of the main test environment (they force a
        # large joint at-floor resolve on the Downgrade lane); they live in this
        # group's own test/AD/Project.toml, auto-activated before the body runs.
        "AD" => (;
            env = joinpath(@__DIR__, "AD"),
            body = function ()
                return @time @safetestset "FIRK Expanded AD Tests" include("AD/ad_tests.jl")
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
    # "All" runs EXPANDED + NESTED + AD + QA. "Core" is intentionally excluded: its
    # basic tests are already covered inside EXPANDED and NESTED.
    all = ["EXPANDED", "NESTED", "AD", "QA"],
)
