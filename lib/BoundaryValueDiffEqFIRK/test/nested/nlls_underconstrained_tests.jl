using BoundaryValueDiffEqFIRK
using Test

include("nlls_test_setup.jl")

using SciMLBase

# Individual solves take minutes. CI runs each nonlinear solver in a separate job;
# the default still covers every case when running the aggregate group locally.
function test_underconstrained_bvp(solver_indices = eachindex(SOLVERS))
    return @testset "Underconstrained BVP" begin
        @testset "Problem: $i" for i in 1:2
            prob = UnderconstrainedProbArr[i]
            @testset "Solver: $(SOLVERS_NAMES[j])" for j in solver_indices
                name, solver = SOLVERS_NAMES[j], SOLVERS[j]
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
end
