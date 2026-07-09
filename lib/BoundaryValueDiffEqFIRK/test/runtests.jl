using SafeTestsets, Test
using SciMLTesting

expanded_basic() = @time @safetestset "FIRK Expanded Basic Tests" include("expanded/firk_basic_tests.jl")
expanded_nlls() = @time @safetestset "FIRK Expanded NLLS Tests" include("expanded/nlls_tests.jl")
expanded_ensemble() = @time @safetestset "FIRK Expanded Ensemble Tests" include("expanded/ensemble_tests.jl")
expanded_singular() = @time @safetestset "FIRK Expanded Singular BVP Tests" include("expanded/singular_bvp_tests.jl")
expanded_dae() = @time @safetestset "FIRK Expanded DAE Tests" include("expanded/dae_tests.jl")
expanded_vector_initials() = @time @safetestset "FIRK Expanded VectorOfVector Initials Tests" include("expanded/vectorofvector_initials_tests.jl")

function expanded_all()
    expanded_basic()
    expanded_nlls()
    expanded_ensemble()
    expanded_singular()
    expanded_dae()
    return expanded_vector_initials()
end

nested_basic() = @time @safetestset "FIRK Nested Basic Tests" include("nested/firk_basic_tests.jl")
nested_nlls() = @time @safetestset "FIRK Nested NLLS Tests" include("nested/nlls_tests.jl")
nested_ensemble() = @time @safetestset "FIRK Nested Ensemble Tests" include("nested/ensemble_tests.jl")
nested_dae() = @time @safetestset "FIRK Nested DAE Tests" include("nested/dae_tests.jl")
nested_vector_initials() = @time @safetestset "FIRK Nested VectorOfVector Initials Tests" include("nested/vectorofvector_initials_tests.jl")

function nested_all()
    nested_basic()
    nested_nlls()
    nested_ensemble()
    nested_dae()
    return nested_vector_initials()
end

run_tests(;
    env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    # Core: representative light FIRK set covering both formulations' basic solves.
    # Targeted by the uniform "Core" downgrade value. "All" runs the split groups,
    # so "Core" is kept out of "All" (see the `all` list below) to avoid double
    # execution.
    core = function ()
        expanded_basic()
        return nested_basic()
    end,
    groups = Dict(
        "EXPANDED" => expanded_all,
        "EXPANDED_BASIC" => expanded_basic,
        "EXPANDED_NLLS" => expanded_nlls,
        "EXPANDED_ENSEMBLE" => expanded_ensemble,
        "EXPANDED_SINGULAR" => expanded_singular,
        "EXPANDED_DAE" => expanded_dae,
        "EXPANDED_VECTOR_INITIALS" => expanded_vector_initials,
        "NESTED" => nested_all,
        "NESTED_BASIC" => nested_basic,
        "NESTED_NLLS" => nested_nlls,
        "NESTED_ENSEMBLE" => nested_ensemble,
        "NESTED_DAE" => nested_dae,
        "NESTED_VECTOR_INITIALS" => nested_vector_initials,
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
    # "All" runs the split expanded/nested groups + AD + QA. "Core" is intentionally
    # excluded because its basic tests are already covered by the split groups.
    all = [
        "EXPANDED_BASIC",
        "EXPANDED_NLLS",
        "EXPANDED_ENSEMBLE",
        "EXPANDED_SINGULAR",
        "EXPANDED_DAE",
        "EXPANDED_VECTOR_INITIALS",
        "NESTED_BASIC",
        "NESTED_NLLS",
        "NESTED_ENSEMBLE",
        "NESTED_DAE",
        "NESTED_VECTOR_INITIALS",
        "AD",
        "QA",
    ],
)
