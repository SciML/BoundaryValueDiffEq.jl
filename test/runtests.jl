using ReTestItems

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

@time begin
    if GROUP == "All" || GROUP == "MIRK"
        @time "MIRK solvers" begin
            ReTestItems.runtests(joinpath(@__DIR__, "mirk/"))
        end
    end

    if GROUP == "All" || GROUP == "MISC"
        @time "Miscellaneous" begin
            ReTestItems.runtests(joinpath(@__DIR__, "misc/"))
        end
    end

    if GROUP == "All" || GROUP == "SHOOTING"
        @time "Shooting solvers" begin
            ReTestItems.runtests(joinpath(@__DIR__, "shooting/"))
        end
    end

    if GROUP == "All" || GROUP == "FIRK"
        @time "FIRK solvers" begin
            ReTestItems.runtests(joinpath(@__DIR__, "firk/expanded/"))
            ReTestItems.runtests(joinpath(@__DIR__, "firk/nested/"))
        end
    end

    if GROUP == "All" || GROUP == "WRAPPERS"
        @time "WRAPPER solvers" begin
            if !Sys.iswindows() && !Sys.isapple()
                # Wrappers like ODEInterface don't support parallel testing
                ReTestItems.runtests(joinpath(@__DIR__, "wrappers/"); nworkers = 0)
            end
        end
    end
end

if !Sys.iswindows() && !Sys.isapple()
    # Wrappers like ODEInterface don't support parallel testing
    ReTestItems.runtests(joinpath(@__DIR__, "wrappers/"); nworkers = 0)
end
