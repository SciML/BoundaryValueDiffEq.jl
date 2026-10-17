using BoundaryValueDiffEqFIRK
using ADTypes: AutoFiniteDiff, AutoSparse
using BoundaryValueDiffEqCore: BVPJacobianAlgorithm
using NonlinearSolveFirstOrder: NewtonRaphson
using SciMLBase: BVProblem, solve
using Test

include("firk_test_setup.jl")

@testset "Simple Pendulum" begin
    using StaticArrays

    tspan = (0.0, π / 2)
    function simplependulum!(du, u, p, t)
        g, L, θ, dθ = 9.81, 1.0, u[1], u[2]
        du[1] = dθ
        du[2] = -(g / L) * sin(θ)
    end

    function bc_pendulum!(residual, u, p, t)
        residual[1] = u(pi / 4)[1] + π / 2 # the solution at the middle of the time span should be -pi/2
        residual[2] = u(pi / 2)[1] - π / 2 # the solution at the end of the time span should be pi/2
    end

    u0 = MVector{2}([pi / 2, pi / 2])
    bvp1 = BVProblem(simplependulum!, bc_pendulum!, u0, tspan)

    jac_alg = BVPJacobianAlgorithm(;
        bc_diffmode = AutoFiniteDiff(), nonbc_diffmode = AutoSparse(AutoFiniteDiff())
    )
    nlsolve = NewtonRaphson()
    nested = true

    # Using ForwardDiff might lead to Cache expansion warnings
    @test_nowarn solve(bvp1, LobattoIIIa2(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)
    @test_nowarn solve(bvp1, LobattoIIIa3(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)
    @test_nowarn solve(bvp1, LobattoIIIa4(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)
    @test_nowarn solve(bvp1, LobattoIIIa5(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)

    @test_nowarn solve(
        bvp1, LobattoIIIb2(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005, adaptive = false
    )
    @test_nowarn solve(bvp1, LobattoIIIb3(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)
    @test_nowarn solve(bvp1, LobattoIIIb4(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)
    @test_nowarn solve(bvp1, LobattoIIIb5(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)

    @test_nowarn solve(
        bvp1, LobattoIIIc2(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005, adaptive = false
    )
    @test_nowarn solve(bvp1, LobattoIIIc3(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)
    @test_nowarn solve(bvp1, LobattoIIIc4(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)
    @test_nowarn solve(bvp1, LobattoIIIc5(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)

    @test_nowarn solve(
        bvp1, RadauIIa1(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005, adaptive = false
    )
    @test_nowarn solve(bvp1, RadauIIa2(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)
    @test_nowarn solve(bvp1, RadauIIa3(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.005)
    @test_nowarn solve(bvp1, RadauIIa5(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.05)
    @test_nowarn solve(bvp1, RadauIIa7(; nlsolve, jac_alg, nested_nlsolve = nested); dt = 0.05)
end
