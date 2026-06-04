using ReTestItems, BoundaryValueDiffEqFIRK, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Standard sublibrary test groups (Core / QA). The root test/runtests.jl
# activates this sublibrary and sets BVDE_TEST_GROUP to the standard group name.
const GROUP = get(ENV, "BVDE_TEST_GROUP", "All")

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
