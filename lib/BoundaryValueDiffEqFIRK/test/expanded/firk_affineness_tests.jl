using BoundaryValueDiffEqFIRK
using Test

include("firk_test_setup.jl")

@testset "Affineness" begin
    using LinearAlgebra

    @testset "Problem: $i" for i in (1, 2, 7, 8)
        prob = probArr[i]

        @testset "LobattoIIIa$stage" for stage in (2, 3, 4, 5)
            @time sol = solve(prob, lobattoIIIa_solver(Val(stage)); dt = 0.2, adaptive = false)
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < affineTol
        end
        @testset "LobattoIIIb$stage" for stage in (2, 3, 4, 5)
            @time sol = solve(prob, lobattoIIIb_solver(Val(stage)); dt = 0.2, adaptive = false)
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < affineTol
        end
        @testset "LobattoIIIc$stage" for stage in (2, 3, 4, 5)
            @time sol = solve(prob, lobattoIIIc_solver(Val(stage)); dt = 0.2, adaptive = false)
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < affineTol
        end

        @testset "RadauIIa$stage" for stage in (2, 3, 5, 7)
            @time sol = solve(prob, radau_solver(Val(stage)); dt = 0.2, adaptive = false)
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < affineTol
        end
    end
end
