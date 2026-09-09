using SafeTestsets, Test
using SciMLTesting

run_tests(;
    env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    core = function ()
        @time @safetestset "MIRK Basic Tests" include("Core/mirk_basic_tests.jl")
        @time @safetestset "MIRK NLLS Tests" include("Core/nlls_tests.jl")
        @time @safetestset "MIRK Ensemble Tests" include("Core/ensemble_tests.jl")
        @time @safetestset "MIRK Singular BVP Tests" include("Core/singular_bvp_tests.jl")
        @time @safetestset "MIRK VectorOfVector Initials Tests" include("Core/vectorofvector_initials_tests.jl")
        return @time @safetestset "MIRK Dynamic Optimization Tests" include("Core/dynamic_optimization_tests.jl")
    end,
    groups = Dict(
        # DAE: the index-1 DAE (singular mass matrix) tests. Split out of Core
        # rather than left in it: Core already runs close to its 120 min budget,
        # and a dedicated group also makes the DAE coverage runnable on its own.
        "DAE" => function ()
            return @time @safetestset "MIRK DAE Tests" include("Core/dae_tests.jl")
        end,
        # AD: the different-AD-backend compatibility tests. Enzyme and Mooncake are
        # heavy optional backends kept out of the main test environment (they force a
        # large joint at-floor resolve on the Downgrade lane); they live in this
        # group's own test/AD/Project.toml, auto-activated before the body runs.
        "AD" => (;
            env = joinpath(@__DIR__, "AD"),
            body = function ()
                return @time @safetestset "MIRK AD Tests" include("AD/ad_tests.jl")
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
    all = ["Core", "DAE", "AD", "QA"],
)
