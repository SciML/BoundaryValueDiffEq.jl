using BoundaryValueDiffEq
using DiffEqBase, OrdinaryDiffEq, DiffEqDevTools
using Test, SafeTestsets

@testset "Boundary Value Problem Tests" begin
    @time @testset "Shooting Method Tests" begin
        @time @safetestset "Shooting Tests" begin include("shooting_tests.jl") end
        @time @safetestset "Orbital" begin include("orbital.jl") end
    end
#=
    @time @testset "Collocation Method (MIRK) Tests" begin
        @time @safetestset "Ensemble" begin include("ensemble.jl") end
        @time @safetestset "MIRK Convergence Tests" begin include("mirk_convergence_tests.jl") end
        @time @safetestset "Vector of Vector" begin include("vectorofvector_initials.jl") end
    end
=#
end
