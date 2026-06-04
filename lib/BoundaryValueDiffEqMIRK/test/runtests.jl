using ReTestItems, BoundaryValueDiffEqMIRK, Hwloc, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Standard sublibrary test groups (Core / QA). Under the centralized sublibrary
# CI the root test/runtests.jl activates this sublibrary and sets
# BVDE_TEST_GROUP to the standard group name (Core / QA) parsed from the emitted
# matrix `group`. When this file is run directly with GROUP set, honor the
# standard `<pkg>` / `<pkg>_<GROUP>` naming via the prefix-strip shim.
# Core (and All) run every test item; QA runs the :qa-tagged Aqua tests.
const _SUB = "BoundaryValueDiffEqMIRK"
const _G = get(ENV, "GROUP", "All")
const GROUP = get(
    ENV, "BVDE_TEST_GROUP",
    _G == _SUB ? "Core" :
        (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)
)
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
