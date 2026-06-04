using ReTestItems, BoundaryValueDiffEqAscher, Hwloc, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Under the centralized sublibrary CI the root test/runtests.jl activates this
# sublibrary and sets BVDE_TEST_GROUP to the group name (Core) parsed from the
# matrix `group`. "core"/"all" run every test item; any other value filters by
# that tag. When run directly, honor GROUP or default to All.
const GROUP = lowercase(get(ENV, "BVDE_TEST_GROUP", get(ENV, "GROUP", "All")))
const TEST_TAGS = (GROUP == "all" || GROUP == "core") ? nothing : [Symbol(GROUP)]

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

ReTestItems.runtests(
    BoundaryValueDiffEqAscher; tags = TEST_TAGS,
    nworkers = RETESTITEMS_NWORKERS,
    nworker_threads = RETESTITEMS_NWORKER_THREADS, testitem_timeout = 3 * 60 * 60
)
