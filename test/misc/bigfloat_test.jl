@testitem "BigFloat compatibility" begin
    using BoundaryValueDiffEq
    # Need Sparspak for BigFloat
    using Sparspak

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
    multi_point_bvp = BVProblem(simplependulum!, bc!, u0, tspan)

    @testset "BigFloat compatibility with Multi-point BVP" begin
        for solver in [MIRK4(), RadauIIa5(), LobattoIIIa4(nested_nlsolve = true)]
            sol = solve(multi_point_bvp, solver, dt = 0.05)
            @test SciMLBase.successful_retcode(sol.retcode)
        end
    end

    function f!(du, u, p, t)
        du[1] = u[2]
        du[2] = u[1]
    end
    function bca!(resid_a, u_a, p)
        resid_a[1] = u_a[1] - 1
    end
    function bcb!(resid_b, u_b, p)
        resid_b[1] = u_b[1]
    end
    bvp_function = BVPFunction(f!, (bca!, bcb!), bcresid_prototype = (zeros(1), zeros(1)), twopoint = Val(true))
    tspan = (0.0, 1.0)
    two_point_bvp = BVProblem(bvp_function, BigFloat.([1.0, 0.0]), tspan)

    @testset "BigFloat compatibility with Two-point BVP" begin
        for solver in [MIRK4(), RadauIIa5(), LobattoIIIa4(nested_nlsolve = true)]
            sol = solve(two_point_bvp, solver, dt = 0.05)
            @test SciMLBase.successful_retcode(sol.retcode)
        end
    end

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
    @test_broken sol4 = solve(prob, MIRKN4(), dt = 0.01)
    @test_broken SciMLBase.successful_retcode(sol4.retcode)
end
