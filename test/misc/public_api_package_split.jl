using BoundaryValueDiffEqFIRK: RadauIIa5
using BoundaryValueDiffEqMIRK: MIRK4
using BoundaryValueDiffEqMIRKN: MIRKN4
using BoundaryValueDiffEqShooting: Shooting
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: BVProblem, SecondOrderBVProblem, solve, successful_retcode
using Test

@testset "Public API package splits" begin
    function straight_line!(du, u, p, t)
        du[1] = u[2]
        du[2] = 0
        return du
    end

    function endpoints!(residual, u, p, t)
        residual[1] = u(0.0)[1]
        residual[2] = u(1.0)[1] - 1
        return residual
    end

    u0 = (p, t) -> [t, 1.0]
    prob = BVProblem(straight_line!, endpoints!, u0, (0.0, 1.0))

    for (name, alg, kwargs) in (
            ("MIRK", MIRK4(), (; dt = 0.05)),
            ("FIRK", RadauIIa5(; nested_nlsolve = true), (; dt = 0.05)),
            ("Shooting", Shooting(Tsit5()), (; abstol = 1.0e-8, reltol = 1.0e-8)),
        )
        @testset "$name" begin
            sol = solve(prob, alg; kwargs...)

            @test successful_retcode(sol)
            @test sol(0.0)[1] ≈ 0 atol = 1.0e-8
            @test sol(1.0)[1] ≈ 1 atol = 1.0e-8
        end
    end

    function free_particle!(ddu, du, u, p, t)
        ddu[1] = 0
        return ddu
    end

    function second_order_endpoints!(residual, du, u, p, t)
        residual[1] = u(0.0)[1]
        residual[2] = u(1.0)[1] - 1
        return residual
    end

    second_order_prob =
        SecondOrderBVProblem(free_particle!, second_order_endpoints!, [0.0], (0.0, 1.0))
    second_order_sol = solve(second_order_prob, MIRKN4(); dt = 0.05)

    @test successful_retcode(second_order_sol)
    @test second_order_sol(0.0)[1] ≈ 0 atol = 1.0e-8
    @test second_order_sol(1.0)[1] ≈ 1 atol = 1.0e-8
end
