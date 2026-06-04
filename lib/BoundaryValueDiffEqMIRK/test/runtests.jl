using ReTestItems, BoundaryValueDiffEqMIRK, Hwloc, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Standard sublibrary test groups (Core / QA). The root test/runtests.jl
# activates this sublibrary and sets BVDE_TEST_GROUP to the standard group name.
# Core (and All) run every test item; QA runs the :qa-tagged Aqua tests.
const GROUP = get(ENV, "BVDE_TEST_GROUP", "All")
const TEST_TAGS = GROUP in ("All", "Core") ? nothing : [Symbol(lowercase(GROUP))]

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
    BoundaryValueDiffEqMIRK; tags = TEST_TAGS,
    nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
    testitem_timeout = 5 * 60 * 60
)
