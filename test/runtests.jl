using ReTestItems

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

@time begin
    if GROUP == "All" || GROUP == "MRIK"
        @time "MIRK solver" begin
            ReTestItems.runtests(joinpath(@__DIR__, "mirk/"))
        end
    end

    if GROUP == "All" || GROUP == "MISC"
        @time "Miscellaneous" begin
            ReTestItems.runtests(joinpath(@__DIR__, "misc/"))
        end
    end

    if GROUP == "All" || GROUP == "SHOOTING"
        @time "Shooting solver" begin
            ReTestItems.runtests(joinpath(@__DIR__, "shooting/"))
        end
    end

    if GROUP == "All" || GROUP == "FIRK"
        @time "FIRK solver" begin
            ReTestItems.runtests(joinpath(@__DIR__, "firk/expanded/"))
            ReTestItems.runtests(joinpath(@__DIR__, "firk/nested/"))
        end
    end
end

if !Sys.iswindows() && !Sys.isapple()
    # Wrappers like ODEInterface don't support parallel testing
    ReTestItems.runtests(joinpath(@__DIR__, "wrappers/"); nworkers = 0)
end
