using BoundaryValueDiffEqFIRK
using Test

include("firk_test_setup.jl")

@testset "Convergence on Linear" begin
    using LinearAlgebra, DiffEqDevTools

    @testset "Problem: $i" for i in (3, 4, 9, 10)
        prob = probArr[i]

        @testset "LobattoIIIa$stage" for stage in (2, 3, 4, 5)
            @time sim = test_convergence(
                dts, prob, lobattoIIIa_solver(Val(stage); nested_nlsolve = nested);
                abstol = 1.0e-8, reltol = 1.0e-8
            )
            if (stage == 4 && ((i == 9) || (i == 10)))
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            elseif first(sim.errors[:final]) < 1.0e-12
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            end
        end

        @testset "LobattoIIIb$stage" for stage in (2, 3, 4, 5)
            @time sim = test_convergence(
                dts, prob, lobattoIIIb_solver(Val(stage); nested_nlsolve = nested);
                abstol = 1.0e-8, reltol = 1.0e-8
            )
            if (stage == 4 && ((i == 9) || (i == 10)))
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            elseif first(sim.errors[:final]) < 1.0e-12
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            end
        end

        @testset "LobattoIIIc$stage" for stage in (2, 3, 4, 5)
            @time sim = test_convergence(
                dts, prob, lobattoIIIc_solver(Val(stage); nested_nlsolve = nested);
                abstol = 1.0e-8, reltol = 1.0e-8
            )
            if stage == 5 || ((stage == 4) && (i == 3 || i == 4))
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            end
        end

        @testset "RadauIIa$stage" for stage in (2, 3, 5, 7)
            @time sim = test_convergence(
                dts, prob, radau_solver(Val(stage); nested_nlsolve = nested);
                abstol = 1.0e-8, reltol = 1.0e-8
            )
            if (stage == 5) || (stage == 7)
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 1 atol = testTol
            elseif first(sim.errors[:final]) < 1.0e-12
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 1 atol = testTol
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 1 atol = testTol
            end
        end
    end
end
