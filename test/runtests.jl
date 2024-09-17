using ReTestItems

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

function activate_mirk()
    Pkg.activate("../lib/BoundaryValueDiffEqMIRK")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "MIRK"
        @time "MIRK solvers" begin
            activate_mirk()
            ReTestItems.runtests(joinpath(@__DIR__, "../lib/BoundaryValueDiffEqMIRK/test/mirk/"))
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

    if GROUP == "All" || GROUP == "FIRK(EXPANDED)"
        @time "FIRK Expanded solvers" begin
            ReTestItems.runtests(joinpath(@__DIR__, "firk/expanded/"))
        end
    end

    if GROUP == "All" || GROUP == "FIRK(NESTED)"
        @time "FIRK Nested solvers" begin
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
