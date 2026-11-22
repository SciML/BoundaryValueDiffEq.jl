using BoundaryValueDiffEqFIRK, SciMLBase, Test

include("nlls_test_setup.jl")

# Individual solves take minutes. CI partitions the nonlinear/FIRK solver pairs;
# the default still covers every case when running the aggregate group locally.
function test_underconstrained_bvp(solver_indices = eachindex(SOLVERS))
    return @testset "Underconstrained BVP" begin
        @testset "Problem: $i" for i in 1:2
            prob = UnderconstrainedProbArr[i]
            @testset "Solver: $(SOLVERS_NAMES[j])" for j in solver_indices
                name, solver = SOLVERS_NAMES[j], SOLVERS[j]
                i == 2 && name in ("RadauIIa5 with GaussNewton", "RadauIIa5 with NewtonRaphson") && continue
                @info "Solving underconstrained BVP" problem = i solver = name
                flush(stderr)
                @time sol = solve(
                    prob, solver; verbose = false, dt = 0.1, abstol = 1.0e-1, reltol = 1.0e-1
                )
                flush(stdout)
                @test SciMLBase.successful_retcode(sol.retcode)
            end
        end
    end
end
