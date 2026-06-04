using ReTestItems, BoundaryValueDiffEqFIRK, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# GROUP selects which FIRK test group to run. Under the centralized sublibrary
# CI the root test/runtests.jl activates this sublibrary and sets BVDE_TEST_GROUP
# to the group name (EXPANDED / NESTED) parsed from the matrix `group`. When run
# directly, honor GROUP (legacy: EXPANDED/NESTED) or default to All (run both).
const GROUP = get(ENV, "BVDE_TEST_GROUP", get(ENV, "GROUP", "All"))

if GROUP == "EXPANDED" || GROUP == "All"
    @time "FIRK Expanded solvers" begin
        ReTestItems.runtests("expanded/", testitem_timeout = 5 * 60 * 60)
    end
end

if GROUP == "NESTED" || GROUP == "All"
    @time "FIRK Nested solvers" begin
        ReTestItems.runtests("nested/", testitem_timeout = 5 * 60 * 60)
    end
end
