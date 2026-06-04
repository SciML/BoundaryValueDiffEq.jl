using Pkg

const GROUP_RAW = get(ENV, "GROUP", "All")

# Centralized sublibrary CI (SciML/.github sublibrary-tests.yml@v1) runs
# `Pkg.test()` on this root package for every affected sublibrary, passing the
# emitted GROUP as the matrix `group`: a bare sublibrary name (its Core group)
# or "<sublibrary>_<TEST_GROUP>" for any other group declared in
# lib/<sublibrary>/test/test_groups.toml. Route those legs into the matching
# sublibrary's own test suite, selecting the group via BVDE_TEST_GROUP.
const _LIB_DIR = joinpath(dirname(@__DIR__), "lib")

function _detect_sublibrary_group(group, lib_dir)
    isdir(joinpath(lib_dir, group)) && return (group, "Core")
    for i in length(group):-1:1
        if group[i] == '_' && isdir(joinpath(lib_dir, group[1:(i - 1)]))
            return (group[1:(i - 1)], group[(i + 1):end])
        end
    end
    return (group, "Core")
end

const _BASE_GROUP, _TEST_GROUP = _detect_sublibrary_group(GROUP_RAW, _LIB_DIR)

if isdir(joinpath(_LIB_DIR, _BASE_GROUP))
    using InteractiveUtils
    @info sprint(InteractiveUtils.versioninfo)
    @info "Routing GROUP=$(GROUP_RAW) to sublibrary $(_BASE_GROUP) test group $(_TEST_GROUP)"

    sublib_dir = joinpath(_LIB_DIR, _BASE_GROUP)
    Pkg.activate(sublib_dir)
    # On Julia < 1.11 the [sources] section is not honored, so develop local
    # path dependencies manually to test the PR branch code rather than a
    # registered release. Resolve transitively across each dep's own [sources].
    if VERSION < v"1.11.0-DEV.0"
        developed = Set{String}([normpath(sublib_dir)])
        specs = Pkg.PackageSpec[]
        queue = [sublib_dir]
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
                            push!(specs, Pkg.PackageSpec(path = dep_path))
                            push!(queue, dep_path)
                        end
                    end
                end
            end
        end
        isempty(specs) || Pkg.develop(specs)
    end
    Pkg.instantiate()
    withenv("BVDE_TEST_GROUP" => _TEST_GROUP) do
        Pkg.test(
            _BASE_GROUP;
            julia_args = ["--check-bounds=auto", "--depwarn=yes"],
            force_latest_compatible_version = false, allow_reresolve = true
        )
    end
else
    # Local / umbrella-package runs: GROUP=All runs the umbrella's own test
    # suite; GROUP=wrappers runs the wrapper tests; any other tag filters.
    using ReTestItems, BoundaryValueDiffEq, Hwloc, InteractiveUtils

    @info sprint(InteractiveUtils.versioninfo)

    const GROUP = lowercase(GROUP_RAW)

    const RETESTITEMS_NWORKERS = parse(
        Int,
        get(
            ENV, "RETESTITEMS_NWORKERS",
            string(min(ifelse(Sys.iswindows(), 0, Hwloc.num_physical_cores()), 4))
        )
    )
    const RETESTITEMS_NWORKER_THREADS = parse(
        Int,
        get(
            ENV, "RETESTITEMS_NWORKER_THREADS",
            string(max(Hwloc.num_virtual_cores() ÷ max(RETESTITEMS_NWORKERS, 1), 1))
        )
    )

    @info "Running tests for group: $(GROUP) with $(RETESTITEMS_NWORKERS) workers"

    if GROUP == "wrappers"
        # Wrapper tests must be explicitly requested via GROUP=wrappers
        ReTestItems.runtests(
            joinpath(@__DIR__, "wrappers");
            nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS
        )
    else
        # For "all" and other groups, exclude wrappers (they have external deps like ODEInterface)
        for dir in readdir(@__DIR__)
            dirpath = joinpath(@__DIR__, dir)
            isdir(dirpath) || continue
            dir in ("wrappers", "qa") && continue
            ReTestItems.runtests(
                dirpath;
                tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
                nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS
            )
        end
    end
end
