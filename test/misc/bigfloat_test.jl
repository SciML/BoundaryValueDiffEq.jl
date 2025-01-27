@testitem "BigFloat compatibility" begin
    using BoundaryValueDiffEq
    tspan = (0.0, pi / 2)
    function simplependulum!(du, u, p, t)
        θ = u[1]
        dθ = u[2]
        du[1] = dθ
        du[2] = -9.81 * sin(θ)
    end
    function bc!(residual, u, p, t)
        residual[1] = u(pi / 4)[1] + big(pi / 2)
        residual[2] = u(pi / 2)[1] - big(pi / 2)
    end
    u0 = BigFloat.([pi / 2, pi / 2])
    prob = BVProblem(simplependulum!, bc!, u0, tspan)
    sol1 = solve(prob, MIRK4(), dt = 0.05)
    @test SciMLBase.successful_retcode(sol1.retcode)

    sol2 = solve(prob, RadauIIa5(), dt = 0.05)
    @test SciMLBase.successful_retcode(sol2.retcode)

    sol3 = solve(prob, LobattoIIIa4(nested_nlsolve = true), dt = 0.05)
    @test SciMLBase.successful_retcode(sol3.retcode)

    function second_f!(ddu, du, u, p, t)
        ϵ = 0.1
        ddu[1] = u[2]
        ddu[2] = (-u[1] * du[2] - u[3] * du[3]) / ϵ
        ddu[3] = (du[1] * u[3] - u[1] * du[3]) / ϵ
    end
    function second_bc!(res, du, u, p, t)
        res[1] = u(0.0)[1]
        res[2] = u(1.0)[1]
        res[3] = u(0.0)[3] + 1
        res[4] = u(1.0)[3] - 1
        res[5] = du(0.0)[1]
        res[6] = du(1.0)[1]
    end
    u0 = BigFloat.([1.0, 1.0, 1.0])
    tspan = (0.0, 1.0)
    prob = SecondOrderBVProblem(second_f!, second_bc!, u0, tspan)
    sol4 = solve(prob, MIRKN4(), dt = 0.01)
    @test SciMLBase.successful_retcode(sol4.retcode)
end
