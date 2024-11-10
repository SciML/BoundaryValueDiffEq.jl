using ReTestItems, BoundaryValueDiffEqFIRK, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

const GROUP = (get(ENV, "GROUP", "EXPANDED"))

@info GROUP

if GROUP == "EXPANDED"
    @time "FIRK Expanded solvers" begin
        ReTestItems.runtests("expanded/", testitem_timeout = 3 * 60 * 60)
    end
end

if GROUP == "NESTED"
    @time "FIRK Nested solvers" begin
        ReTestItems.runtests("nested/", testitem_timeout = 3 * 60 * 60)
    end
end
