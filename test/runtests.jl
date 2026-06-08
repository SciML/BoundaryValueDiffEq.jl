using Pkg
using SafeTestsets, Test
using InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

const GROUP = get(ENV, "GROUP", "All")

function develop_local_path_deps()
    # On Julia < 1.11, the [sources] section in Project.toml is not honored.
    # Manually Pkg.develop the local path dependencies (root + sublibraries) so
    # the sub-environment tests the PR branch code.
    repo_root = dirname(@__DIR__)
    lib = joinpath(repo_root, "lib")
    return Pkg.develop([
        Pkg.PackageSpec(path = repo_root),
        Pkg.PackageSpec(path = joinpath(lib, "BoundaryValueDiffEqAscher")),
        Pkg.PackageSpec(path = joinpath(lib, "BoundaryValueDiffEqCore")),
        Pkg.PackageSpec(path = joinpath(lib, "BoundaryValueDiffEqFIRK")),
        Pkg.PackageSpec(path = joinpath(lib, "BoundaryValueDiffEqMIRK")),
        Pkg.PackageSpec(path = joinpath(lib, "BoundaryValueDiffEqMIRKN")),
        Pkg.PackageSpec(path = joinpath(lib, "BoundaryValueDiffEqShooting"))
    ])
end

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    VERSION < v"1.11.0-DEV.0" && develop_local_path_deps()
    return Pkg.instantiate()
end

function activate_wrappers_env()
    Pkg.activate(joinpath(@__DIR__, "wrappers"))
    VERSION < v"1.11.0-DEV.0" && develop_local_path_deps()
    return Pkg.instantiate()
end

@time begin
    # Detect sublibrary test groups.
    # GROUP can be a bare sublibrary name (its Core test group) or
    # "{sublibrary}_{TEST_GROUP}" for any custom group (e.g. QA, EXPANDED, ...).
    # Sublibraries declare their groups in test/test_groups.toml.
    lib_dir = joinpath(dirname(@__DIR__), "lib")

    # Scan underscores right-to-left to find the longest matching sublibrary prefix.
    function _detect_sublibrary_group(group, lib_dir)
        isdir(joinpath(lib_dir, group)) && return (group, "Core")
        for i in length(group):-1:1
            if group[i] == '_' && isdir(joinpath(lib_dir, group[1:(i - 1)]))
                return (group[1:(i - 1)], group[(i + 1):end])
            end
        end
        return (group, "Core")
    end
    base_group, test_group = _detect_sublibrary_group(GROUP, lib_dir)

    if isdir(joinpath(lib_dir, base_group))
        Pkg.activate(joinpath(lib_dir, base_group))
        # On Julia < 1.11, the [sources] section in Project.toml is not honored.
        # Manually Pkg.develop local path dependencies (transitively) so CI tests
        # the PR branch code.
        if VERSION < v"1.11.0-DEV.0"
            developed = Set{String}()
            push!(developed, normpath(joinpath(lib_dir, base_group)))
            specs = Pkg.PackageSpec[]
            queue = [joinpath(lib_dir, base_group)]
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
        withenv("BOUNDARYVALUEDIFFEQ_TEST_GROUP" => test_group) do
            Pkg.test(base_group, julia_args = ["--check-bounds=auto", "--compiled-modules=yes", "--depwarn=yes"], force_latest_compatible_version = false, allow_reresolve = true)
        end
    else
        # Root BoundaryValueDiffEq's own test groups.
        if GROUP == "All" || GROUP == "Misc"
            @time @safetestset "Adaptivity Tests" include("misc/adaptivity_tests.jl")
            @time @safetestset "Initial Guess Tests" include("misc/initial_guess_tests.jl")
            @time @safetestset "Scalar BVP Tests" include("misc/scalar_tests.jl")
            @time @safetestset "Non-Vector Input Tests" include("misc/non_vector_input_tests.jl")
            @time @safetestset "BigFloat Tests" include("misc/bigfloat_test.jl")
            @time @safetestset "Default Solvers Tests" include("misc/default_solvers.jl")
            @time @safetestset "Type Stability Tests" include("misc/type_stability_tests.jl")
            @time @safetestset "Verbose Tests" include("misc/verbose_tests.jl")
            @time @safetestset "Manifolds Tests" include("misc/manifolds_tests.jl")
        end

        if GROUP == "Wrappers"
            # Wrapper tests run in their own sub-environment (adds ODEInterface)
            # and are excluded from "All".
            activate_wrappers_env()
            @time @safetestset "ODEInterface Wrapper Tests" include("wrappers/odeinterface_tests.jl")
        end

        if (GROUP == "All" || GROUP == "QA") && isempty(VERSION.prerelease)
            activate_qa_env()
            @time @safetestset "Quality Assurance" include("qa/qa.jl")
            @time @safetestset "JET" include("qa/jet.jl")
        end
    end
end
