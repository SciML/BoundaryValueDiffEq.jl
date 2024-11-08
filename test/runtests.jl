using ReTestItems, Pkg

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

function activate_mirk()
    Pkg.activate("../lib/BoundaryValueDiffEqMIRK")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

function activate_mirkn()
    Pkg.activate("../lib/BoundaryValueDiffEqMIRKN")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

function activate_firk()
    Pkg.activate("../lib/BoundaryValueDiffEqFIRK")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

function activate_shooting()
    Pkg.activate("../lib/BoundaryValueDiffEqShooting")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

function activate_ascher()
    Pkg.activate("../lib/BoundaryValueDiffEqAscher")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "MIRK"
        @time "MIRK solvers" begin
            activate_mirk()
            ReTestItems.runtests("../lib/BoundaryValueDiffEqMIRK/test/ensemble_tests.jl")
            ReTestItems.runtests("../lib/BoundaryValueDiffEqMIRK/test/mirk_basic_tests.jl")
            ReTestItems.runtests("../lib/BoundaryValueDiffEqMIRK/test/nlls_tests.jl")
            ReTestItems.runtests("../lib/BoundaryValueDiffEqMIRK/test/vectorofvector_initials_tests.jl")
        end
    end

    if GROUP == "All" || GROUP == "MIRKN"
        @time "MIRKN solvers" begin
            activate_mirkn()
            ReTestItems.runtests("../lib/BoundaryValueDiffEqMIRKN/test/mirkn_basic_tests.jl")
        end
    end

    if GROUP == "All" || GROUP == "FIRK(EXPANDED)"
        @time "FIRK Expanded solvers" begin
            activate_firk()
            ReTestItems.runtests("../lib/BoundaryValueDiffEqFIRK/test/expanded/")
        end
    end

    if GROUP == "All" || GROUP == "FIRK(NESTED)"
        @time "FIRK Nested solvers" begin
            activate_firk()
            ReTestItems.runtests("../lib/BoundaryValueDiffEqFIRK/test/nested/")
        end
    end

    if GROUP == "All" || GROUP == "ASCHER"
        @time "Ascher BVDAE solvers" begin
            activate_ascher()
            ReTestItems.runtests("../lib/BoundaryValueDiffEqAscher/test/nested/")
        end
    end

    if GROUP == "All" || GROUP == "MISC"
        @time "Miscellaneous" begin
            ReTestItems.runtests(joinpath(@__DIR__, "misc/"))
        end
    end

    if GROUP == "All" || GROUP == "SHOOTING"
        @time "Shooting solvers" begin
            activate_shooting()
            ReTestItems.runtests("../lib/BoundaryValueDiffEqShooting/test/basic_problems_tests.jl")
            ReTestItems.runtests("../lib/BoundaryValueDiffEqShooting/test/nlls_tests.jl")
            ReTestItems.runtests("../lib/BoundaryValueDiffEqShooting/test/orbital_tests.jl")
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
