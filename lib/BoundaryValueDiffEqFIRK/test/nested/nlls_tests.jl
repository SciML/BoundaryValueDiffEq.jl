using BoundaryValueDiffEqFIRK
using SciMLBase
using Test

include("nlls_test_setup.jl")

@testset "Overconstrained BVP" begin
    using LinearAlgebra, BoundaryValueDiffEqFIRK

    @testset "Problem: $i" for i in 1:4
        prob = OverconstrainedProbArr[i]
        @testset "Solver: $name" for (name, solver) in zip(SOLVERS_NAMES, SOLVERS)
            sol = solve(prob, solver; verbose = false, dt = 1.0)
            @test norm(bc1(sol, nothing, sol.t), Inf) < 1.0e-2
        end
    end
end
