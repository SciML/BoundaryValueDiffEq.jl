using Pkg
using SafeTestsets, Test
using SciMLTesting

const GROUP = current_group()
const LIB_DIR = joinpath(dirname(@__DIR__), "lib")

# Centralized sublibrary CI (SciML/.github sublibrary-project-tests.yml@v1) tests
# each lib/<name> via the project model and never routes through this file. This
# dispatcher only matters when the root suite is invoked with a GROUP that names a
# sublibrary (e.g. local `GROUP=BoundaryValueDiffEqMIRK julia test/runtests.jl`):
# the bare sublibrary name selects that sublibrary's "Core" group and
# "<sublibrary>_<grp>" selects a named group. We then activate the sublibrary's own
# test environment and hand off to its runtests.jl via BOUNDARYVALUEDIFFEQ_TEST_GROUP.
# The sublibrary Pkg.test is done explicitly here (rather than via run_tests's
# built-in lib_dir path) so the Julia < 1.11 transitive [sources] develop walk is
# preserved verbatim. Otherwise the main-package suite below runs via run_tests:
# GROUP="All" runs the Misc group plus QA; "Wrappers" and "QA" each activate their
# own test/<group> environment.
base_group, test_group = detect_sublibrary_group(GROUP, LIB_DIR)

if !isempty(base_group) && isdir(joinpath(LIB_DIR, base_group))
    sublib_path = joinpath(LIB_DIR, base_group)
    Pkg.activate(sublib_path)
    # On Julia < 1.11 the [sources] table in Project.toml is ignored, so develop the
    # local path dependencies (root + sublibraries) to test the PR branch code. Walk
    # [sources] transitively in case a developed dependency carries its own.
    if VERSION < v"1.11.0-DEV.0"
        developed = Set{String}()
        push!(developed, normpath(sublib_path))
        specs = Pkg.PackageSpec[]
        queue = [sublib_path]
        while !isempty(queue)
            pkg_dir = popfirst!(queue)
            toml_path = joinpath(pkg_dir, "Project.toml")
            isfile(toml_path) || continue
            toml = Pkg.TOML.parsefile(toml_path)
            if haskey(toml, "sources")
                for (dep_name, source_spec) in toml["sources"]
                    if source_spec isa Dict && haskey(source_spec, "path")
                        dep_path = normpath(joinpath(pkg_dir, source_spec["path"]))
                        if isdir(dep_path) && !(dep_path in developed)
                            push!(developed, dep_path)
                            @info "Queuing local source dependency" dep_name dep_path
                            push!(specs, Pkg.PackageSpec(path = dep_path))
                            push!(queue, dep_path)
                        end
                    end
                end
            end
        end
        isempty(specs) || Pkg.develop(specs)
    end
    # Hand the resolved test group to the sublibrary runtests.jl, which reads
    # BOUNDARYVALUEDIFFEQ_TEST_GROUP (matching the SublibraryCI group-env-name).
    withenv("BOUNDARYVALUEDIFFEQ_TEST_GROUP" => test_group) do
        Pkg.test(base_group; julia_args = ["--check-bounds=auto", "--compiled-modules=yes", "--depwarn=yes"], force_latest_compatible_version = false, allow_reresolve = true)
    end
else
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
end
