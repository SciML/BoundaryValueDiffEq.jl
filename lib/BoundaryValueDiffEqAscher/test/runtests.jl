using ReTestItems
const GROUP = get(ENV, "GROUP", "All")
@time begin
    if GROUP == "All" || GROUP == "ASCHER"
        @time "ASCHER solvers" begin
            ReTestItems.runtests("ascher_basic_tests.jl")
        end
    end
end
