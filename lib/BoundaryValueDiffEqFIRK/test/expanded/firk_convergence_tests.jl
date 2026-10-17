using BoundaryValueDiffEqFIRK
using Test

include("firk_test_setup.jl")

@testset "Convergence on Linear" begin
    using LinearAlgebra, DiffEqDevTools

    @testset "Problem: $i" for i in (3, 4, 9, 10)
        prob = probArr[i]

        @testset "LobattoIIIa$stage" for stage in (2, 3, 4, 5)
            stepsizes = stage == 4 ? dts_stage4 : dts
            @time sim = test_convergence(stepsizes, prob, lobattoIIIa_solver(Val(stage)); abstol = 1.0e-8)
            if stage == 5
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            end
        end

        @testset "LobattoIIIb$stage" for stage in (2, 3, 4, 5)
            stepsizes = stage == 4 ? dts_stage4 : dts
            @time sim = test_convergence(
                stepsizes, prob, lobattoIIIb_solver(Val(stage)); abstol = 1.0e-8, reltol = 1.0e-8
            )
            if stage == 5
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            end
        end

        @testset "LobattoIIIc$stage" for stage in (2, 3, 4, 5)
            stepsizes = stage == 4 ? dts_stage4 : dts
            @time sim = test_convergence(
                stepsizes, prob, lobattoIIIc_solver(Val(stage)); abstol = 1.0e-8, reltol = 1.0e-8
            )
            if stage != 4 && first(sim.errors[:final]) < 1.0e-12
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            end
        end

        @testset "RadauIIa$stage" for stage in (2, 3, 5, 7)
            @time sim = test_convergence(
                dts, prob, radau_solver(Val(stage)); abstol = 1.0e-8, reltol = 1.0e-8
            )
            if first(sim.errors[:final]) < 1.0e-12
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 1 atol = testTol
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 1 atol = testTol
            end
        end
    end
end
