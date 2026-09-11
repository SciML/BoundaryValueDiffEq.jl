using BoundaryValueDiffEqFIRK
using Test

include("firk_test_setup.jl")

@testset "Affineness" begin
    using LinearAlgebra

    @testset "Problem: $i" for i in (1, 2, 7, 8)
        prob = probArr[i]

        @testset "LobattoIIIa$stage" for stage in (2, 3, 4, 5)
            @time sol = solve(prob, lobattoIIIa_solver(Val(stage); nested_nlsolve = nested); dt = 0.2)
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < affineTol
        end
        @testset "LobattoIIIb$stage" for stage in (2, 3, 4, 5)
            @time if stage == 2 # LobattoIIIb2 doesn't support adaptivity
                sol = solve(
                    prob, lobattoIIIb_solver(Val(stage); nested_nlsolve = nested);
                    dt = 0.2, adaptive = false
                )
            else
                sol = solve(prob, lobattoIIIb_solver(Val(stage); nested_nlsolve = nested); dt = 0.2)
            end
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < 0.3
        end
        @testset "LobattoIIIc$stage" for stage in (2, 3, 4, 5)
            @time if stage == 2 # LobattoIIIc2 doesn't support adaptivity
                sol = solve(
                    prob, lobattoIIIc_solver(Val(stage); nested_nlsolve = nested);
                    dt = 0.2, adaptive = false
                )
            else
                sol = solve(prob, lobattoIIIc_solver(Val(stage); nested_nlsolve = nested); dt = 0.2)
            end
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < affineTol
        end

        @testset "RadauIIa$stage" for stage in (1, 2, 3, 5, 7)
            @time if stage == 1
                sol = solve(prob, radau_solver(Val(stage); nested_nlsolve = nested); dt = 0.2, adaptive = false)
            else
                sol = solve(prob, radau_solver(Val(stage); nested_nlsolve = nested); dt = 0.2)
            end
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < affineTol
        end
    end
end

# JET tests have been moved to the separate QA test group (test/qa/)
