using BoundaryValueDiffEqFIRK
using Test

include("nlls_test_setup.jl")

# Not a very meaningful problem; it tests that the solvers do not throw. It is also by far
# the slowest testset in the package -- individual solves here take 4-20 min each of real
# solve time (not compilation), so the 22 cases run for hours. Kept in its own group for
# that reason.
@testset "Underconstrained BVP" begin
    using LinearAlgebra, BoundaryValueDiffEqFIRK, SciMLBase

    @testset "Problem: $i" for i in 1:2
        prob = UnderconstrainedProbArr[i]
        @testset "Solver: $name" for (name, solver) in zip(SOLVERS_NAMES, SOLVERS)
            if (i == 2) && (
                    (name == "RadauIIa5 with GaussNewton") ||
                        (name == "RadauIIa5 with NewtonRaphson")
                )
                # Actually have successful retcode
                continue
            else
                sol = solve(
                    prob, solver; verbose = false, dt = 0.1, abstol = 1.0e-1, reltol = 1.0e-1
                )
                @test SciMLBase.successful_retcode(sol.retcode)
            end
        end
    end
end
