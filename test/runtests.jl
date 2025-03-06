using ReTestItems, BoundaryValueDiffEq, Hwloc, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

@time "Test package" begin
    ReTestItems.runtests("misc/")
    ReTestItems.runtests("wrappers/")
end
