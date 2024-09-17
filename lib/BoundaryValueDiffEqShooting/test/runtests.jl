using ReTestItems

@time begin
    if GROUP == "All" || GROUP == "SHOOTING"
        @time "Shooting solvers" begin
            ReTestItems.runtests("basic_problems_tests.jl")
            ReTestItems.runtests("nlls_tests.jl")
            ReTestItems.runtests("orbital_tests.jl")
        end
    end
end
