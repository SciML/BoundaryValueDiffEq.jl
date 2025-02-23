@testitem "Initial guess providing" begin
    using BoundaryValueDiffEq
    tspan = (0.0, pi / 2)
    function simplependulum!(du, u, p, t)
        θ = u[1]
        dθ = u[2]
        du[1] = dθ
        du[2] = -9.81 * sin(θ)
    end
    function bc!(residual, u, p, t)
        residual[1] = u(pi / 4)[1] + pi / 2
        residual[2] = u(pi / 2)[1] - pi / 2
    end
    u0 = [pi / 2, pi / 2]
    prob = BVProblem(simplependulum!, bc!, u0, tspan)
    sol1 = solve(prob, MIRK4(), dt = 0.05)

    # Solution
    prob1 = BVProblem(simplependulum!, bc!, sol1, tspan)
    sol2 = solve(prob1, MIRK4())
    @test SciMLBase.successful_retcode(sol2)

    sol3 = solve(prob1, RadauIIa5())
    @test SciMLBase.successful_retcode(sol3)

    sol4 = solve(prob1, LobattoIIIa4(nested_nlsolve = true))
    @test SciMLBase.successful_retcode(sol4)

    # VectorOfArray
    prob2 = BVProblem(simplependulum!, bc!, VectorOfArray(sol1.u), tspan)
    sol2 = solve(prob2, MIRK4())
    @test SciMLBase.successful_retcode(sol2)

    sol3 = solve(prob2, RadauIIa5())
    @test SciMLBase.successful_retcode(sol3)

    sol4 = solve(prob2, LobattoIIIa4(nested_nlsolve = true))
    @test SciMLBase.successful_retcode(sol4)
end
