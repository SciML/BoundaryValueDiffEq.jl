using BoundaryValueDiffEq
using DiffEqBase, OrdinaryDiffEq, DiffEqDevTools
using Test
using NLsolve

@time @testset "Shooting Method Tests" begin
include("shooting_tests.jl")
include("orbital.jl")
end

@time @testset "Collocation Method (MIRK) Tests" begin include("mirk_convergence_tests.jl") end
