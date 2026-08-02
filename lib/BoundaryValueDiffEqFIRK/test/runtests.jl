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
        # The expanded formulation is split across several matrix groups. Nearly all of
        # its wall time is Julia compilation: every (problem, solver) pair specializes
        # the whole FIRK -> NonlinearSolve -> AD stack afresh at ~20 s a piece, and the
        # solver sweeps run 16 solvers over 4 problems. Run as one group it exceeds two
        # hours (issue #548), so the two solver sweeps get groups of their own.
        "EXPANDED_BASIC" => function ()
            return @time @safetestset "FIRK Expanded Basic Tests" include("expanded/firk_basic_tests.jl")
        end,
        "EXPANDED_AFFINENESS" => function ()
            return @time @safetestset "FIRK Expanded Affineness Tests" include("expanded/firk_affineness_tests.jl")
        end,
        "EXPANDED_CONVERGENCE" => function ()
            return @time @safetestset "FIRK Expanded Convergence Tests" include("expanded/firk_convergence_tests.jl")
        end,
        "EXPANDED_NLLS" => function ()
            return @time @safetestset "FIRK Expanded NLLS Tests" include("expanded/nlls_tests.jl")
        end,
        "EXPANDED_MISC" => function ()
            @time @safetestset "FIRK Expanded Ensemble Tests" include("expanded/ensemble_tests.jl")
            @time @safetestset "FIRK Expanded Singular BVP Tests" include("expanded/singular_bvp_tests.jl")
            return @time @safetestset "FIRK Expanded VectorOfVector Initials Tests" include("expanded/vectorofvector_initials_tests.jl")
        end,
        # Aggregate of the five EXPANDED_* groups for running the whole formulation
        # locally in one process. Kept out of the CI matrix and out of `all` so it does
        # not duplicate them.
        "EXPANDED" => function ()
            @time @safetestset "FIRK Expanded Basic Tests" include("expanded/firk_basic_tests.jl")
            @time @safetestset "FIRK Expanded Affineness Tests" include("expanded/firk_affineness_tests.jl")
            @time @safetestset "FIRK Expanded Convergence Tests" include("expanded/firk_convergence_tests.jl")
            @time @safetestset "FIRK Expanded NLLS Tests" include("expanded/nlls_tests.jl")
            @time @safetestset "FIRK Expanded Ensemble Tests" include("expanded/ensemble_tests.jl")
            @time @safetestset "FIRK Expanded Singular BVP Tests" include("expanded/singular_bvp_tests.jl")
            return @time @safetestset "FIRK Expanded VectorOfVector Initials Tests" include("expanded/vectorofvector_initials_tests.jl")
        end,
        "NESTED" => function ()
            @time @safetestset "FIRK Nested Basic Tests" include("nested/firk_basic_tests.jl")
            @time @safetestset "FIRK Nested NLLS Tests" include("nested/nlls_tests.jl")
            @time @safetestset "FIRK Nested Ensemble Tests" include("nested/ensemble_tests.jl")
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
    # "All" runs the split EXPANDED_* groups + NESTED + AD + QA. "Core" and the
    # aggregate "EXPANDED" are intentionally excluded: both only re-run tests that the
    # listed groups already cover.
    all = [
        "EXPANDED_BASIC", "EXPANDED_AFFINENESS", "EXPANDED_CONVERGENCE",
        "EXPANDED_NLLS", "EXPANDED_MISC", "NESTED", "AD", "QA",
    ],
)
