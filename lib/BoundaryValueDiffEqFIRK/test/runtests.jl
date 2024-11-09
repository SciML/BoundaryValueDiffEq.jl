using ReTestItems, BoundaryValueDiffEqFIRK

@time "FIRK Expanded solvers" begin
    ReTestItems.runtests("expanded/")
end

@time "FIRK Nested solvers" begin
    ReTestItems.runtests("nested/")
end
