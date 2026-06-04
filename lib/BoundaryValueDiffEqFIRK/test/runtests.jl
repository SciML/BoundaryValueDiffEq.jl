using ReTestItems, BoundaryValueDiffEqFIRK, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Standard sublibrary test groups (Core / QA). Under the centralized sublibrary
# CI the root test/runtests.jl activates this sublibrary and sets
# BVDE_TEST_GROUP to the standard group name (Core / QA) parsed from the emitted
# matrix `group`. When this file is run directly with GROUP set, honor the
# standard `<pkg>` / `<pkg>_<GROUP>` naming via the prefix-strip shim.
const _SUB = "BoundaryValueDiffEqFIRK"
const _G = get(ENV, "GROUP", "All")
const GROUP = get(
    ENV, "BVDE_TEST_GROUP",
    _G == _SUB ? "Core" :
        (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)
)

# Core: all functional FIRK tests (folds the former EXPANDED + NESTED groups).
if GROUP in ("Core", "All")
    @time "FIRK Expanded solvers" begin
        ReTestItems.runtests("expanded/", testitem_timeout = 5 * 60 * 60)
    end
    @time "FIRK Nested solvers" begin
        ReTestItems.runtests("nested/", testitem_timeout = 5 * 60 * 60)
    end
end

# QA: quality/static checks (Aqua, tagged :qa).
if GROUP in ("QA", "All")
    @time "FIRK QA" begin
        ReTestItems.runtests("qa_tests.jl"; tags = [:qa])
    end
end
