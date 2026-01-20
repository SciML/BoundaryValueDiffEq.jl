@testitem "Scalar BVP tests" begin
    using BoundaryValueDiffEq, LinearAlgebra, OrdinaryDiffEqTsit5, SciMLBase
    tspan = (0.0, 2.0)

    f(u, p, t) = t * u
    bca(ua, p) = ua - 1
    bcb(ub, p) = ub - exp(2)

    u₀ = 1.0

    bvproblem = TwoPointBVProblem(f, (bca, bcb), u₀, tspan)
    sol = solve(bvproblem, MIRK4(); dt = 0.01)
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid, Inf) < 1.0e-6

    shoot_sol = solve(
        bvproblem, Shooting(Tsit5()); abstol = 1.0e-8, reltol = 1.0e-8,
        odesolve_kwargs = (; abstol = 1.0e-9, reltol = 1.0e-9)
    )
    @test SciMLBase.successful_retcode(shoot_sol)
    @test norm(shoot_sol.resid, Inf) < 1.0e-6
end
