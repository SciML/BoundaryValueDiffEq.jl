using ReTestItems

@time begin
    if GROUP == "All" || GROUP == "MIRK"
        @time "MIRK solvers" begin
            ReTestItems.runtests("ensemble_tests.jl")
            ReTestItems.runtests("mirk_basic_tests.jl")
            ReTestItems.runtests("nlls_tests.jl")
            ReTestItems.runtests("vectorofvector_initials_tests.jl")
        end
    end
end
