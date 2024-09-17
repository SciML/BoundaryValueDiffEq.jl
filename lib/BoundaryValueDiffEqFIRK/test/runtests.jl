using ReTestItems

@time begin
    if GROUP == "All" || GROUP == "FIRK(EXPANDED)"
        @time "FIRK Expanded solvers" begin
            ReTestItems.runtests(joinpath(@__DIR__, "expanded/"))
        end
    end

    if GROUP == "All" || GROUP == "FIRK(NESTED)"
        @time "FIRK Nested solvers" begin
            ReTestItems.runtests(joinpath(@__DIR__, "nested/"))
        end
    end
end
