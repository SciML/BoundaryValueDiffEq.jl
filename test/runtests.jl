using ReTestItems

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

@time begin
    #= Skip MISC tests for now
    if GROUP == "All" || GROUP == "MISC"
        @time "Miscellaneous" begin
            ReTestItems.runtests(joinpath(@__DIR__, "misc/"))
        end
    end
=#
    if GROUP == "All" || GROUP == "WRAPPERS"
        @time "WRAPPER solvers" begin
            if !Sys.iswindows() && !Sys.isapple()
                # Wrappers like ODEInterface don't support parallel testing
                ReTestItems.runtests(joinpath(@__DIR__, "wrappers/"); nworkers = 0)
            end
        end
    end
end
