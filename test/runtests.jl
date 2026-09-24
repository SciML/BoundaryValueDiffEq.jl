using SafeTestsets, Test
using SciMLTesting

const LIB_DIR = joinpath(dirname(@__DIR__), "lib")

# Centralized sublibrary CI (SciML/.github sublibrary-project-tests.yml@v1) tests
# each lib/<name> via the project model and never routes through this file. This
# dispatcher only matters when the root suite is invoked with a GROUP that names a
# sublibrary (e.g. local `GROUP=BoundaryValueDiffEqMIRK julia test/runtests.jl`):
# the bare sublibrary name selects that sublibrary's "Core" group and
# "<sublibrary>_<grp>" selects a named group. We then activate the sublibrary's own
# test environment and hand off to its runtests.jl via BOUNDARYVALUEDIFFEQ_TEST_GROUP.
# SciMLTesting.run_tests owns the sublibrary dispatch and group-env activation.
# GROUP="All" runs the Misc group plus QA; "Wrappers" and "QA" each activate their
# own test/<group> environment.
run_tests(;
    # Misc: the package's own integration tests in test/misc/, run in the light
    # main test environment. Modeled as both the Core group (so a bare "Core"
    # GROUP from the uniform downgrade matrix runs them) and the named "Misc"
    # group from test_groups.toml.
    core = function ()
        @time @safetestset "Adaptivity Tests" include("misc/adaptivity_tests.jl")
        @time @safetestset "Initial Guess Tests" include("misc/initial_guess_tests.jl")
        @time @safetestset "Scalar BVP Tests" include("misc/scalar_tests.jl")
        @time @safetestset "Non-Vector Input Tests" include("misc/non_vector_input_tests.jl")
        @time @safetestset "BigFloat Tests" include("misc/bigfloat_test.jl")
        @time @safetestset "Default Solvers Tests" include("misc/default_solvers.jl")
        @time @safetestset "Type Stability Tests" include("misc/type_stability_tests.jl")
        @time @safetestset "Verbose Tests" include("misc/verbose_tests.jl")
        @time @safetestset "Public API Package Splits" include("misc/public_api_package_split.jl")
        return @time @safetestset "Manifolds Tests" include("misc/manifolds_tests.jl")
    end,
    groups = Dict(
        # Wrappers: ODEInterface wrapper tests run in their own sub-environment
        # (adds ODEInterface) and are excluded from "All".
        "Wrappers" => (;
            env = joinpath(@__DIR__, "wrappers"),
            body = function ()
                return @time @safetestset "ODEInterface Wrapper Tests" include("wrappers/odeinterface_tests.jl")
            end,
        ),
        # "Misc" is the named test_groups.toml group for the test/misc/ tests; it
        # runs the same body as Core so `GROUP=Misc` works in addition to "All".
        "Misc" => function ()
            @time @safetestset "Adaptivity Tests" include("misc/adaptivity_tests.jl")
            @time @safetestset "Initial Guess Tests" include("misc/initial_guess_tests.jl")
            @time @safetestset "Scalar BVP Tests" include("misc/scalar_tests.jl")
            @time @safetestset "Non-Vector Input Tests" include("misc/non_vector_input_tests.jl")
            @time @safetestset "BigFloat Tests" include("misc/bigfloat_test.jl")
            @time @safetestset "Default Solvers Tests" include("misc/default_solvers.jl")
            @time @safetestset "Type Stability Tests" include("misc/type_stability_tests.jl")
            @time @safetestset "Verbose Tests" include("misc/verbose_tests.jl")
            @time @safetestset "Public API Package Splits" include("misc/public_api_package_split.jl")
            return @time @safetestset "Manifolds Tests" include("misc/manifolds_tests.jl")
        end,
    ),
    qa = (;
        env = joinpath(@__DIR__, "qa"),
        body = function ()
            # QA (Aqua + JET + ExplicitImports) runs on release + LTS Julia only;
            # skip on prerelease.
            isempty(VERSION.prerelease) || return nothing
            return @time @safetestset "Quality Assurance" include("qa/qa.jl")
        end,
    ),
    # The original ran the Misc and QA groups for the default GROUP="All", but
    # excluded the dep-adding Wrappers group (it ran only when selected by name).
    all = ["Core", "QA"],
    sublib_env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    lib_dir = LIB_DIR,
)
