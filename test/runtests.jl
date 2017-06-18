using BoundaryValueDiffEq
using DiffEqBase, OrdinaryDiffEq, DiffEqDevTools
using Base.Test

@time @testset "Shooting Method Tests" begin include("shooting_tests.jl") end
@time @testset "Collocation Method (MIRK) Tests" begin include("mirk_convergence_tests.jl") end
