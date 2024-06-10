@testsetup module MIRKConvergenceTests

using BoundaryValueDiffEq

for order in (2, 3, 4, 5, 6)
    s = Symbol("MIRK$(order)")
    @eval mirk_solver(::Val{$order}, args...; kwargs...) = $(s)(args...; kwargs...)
end

# First order test
function f1!(du, u, p, t)
    du[1] = u[2]
    du[2] = 0
end
f1(u, p, t) = [u[2], 0]

# Second order linear test
function f2!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
end
f2(u, p, t) = [u[2], -u[1]]

function boundary!(residual, u, p, t)
    residual[1] = u[1][1] - 5
    residual[2] = u[end][1]
end
boundary(u, p, t) = [u[1][1] - 5, u[end][1]]

function boundary_two_point_a!(resida, ua, p)
    resida[1] = ua[1] - 5
end
function boundary_two_point_b!(residb, ub, p)
    residb[1] = ub[1]
end

boundary_two_point_a(ua, p) = [ua[1] - 5]
boundary_two_point_b(ub, p) = [ub[1]]

# Not able to change the initial condition.
# Hard coded solution.
odef1! = ODEFunction(f1!, analytic = (u0, p, t) -> [5 - t, -1])
odef1 = ODEFunction(f1, analytic = (u0, p, t) -> [5 - t, -1])

odef2! = ODEFunction(f2!,
    analytic = (u0, p, t) -> [
        5 * (cos(t) - cot(5) * sin(t)), 5 * (-cos(t) * cot(5) - sin(t))])
odef2 = ODEFunction(f2,
    analytic = (u0, p, t) -> [
        5 * (cos(t) - cot(5) * sin(t)), 5 * (-cos(t) * cot(5) - sin(t))])

bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

tspan = (0.0, 5.0)
u0 = [5.0, -3.5]

probArr = [BVProblem(odef1!, boundary!, u0, tspan, nlls = Val(false)),
    BVProblem(odef1, boundary, u0, tspan, nlls = Val(false)),
    BVProblem(odef2!, boundary!, u0, tspan, nlls = Val(false)),
    BVProblem(odef2, boundary, u0, tspan, nlls = Val(false)),
    TwoPointBVProblem(odef1!, (boundary_two_point_a!, boundary_two_point_b!),
        u0, tspan; bcresid_prototype, nlls = Val(false)),
    TwoPointBVProblem(odef1, (boundary_two_point_a, boundary_two_point_b),
        u0, tspan; bcresid_prototype, nlls = Val(false)),
    TwoPointBVProblem(odef2!, (boundary_two_point_a!, boundary_two_point_b!),
        u0, tspan; bcresid_prototype, nlls = Val(false)),
    TwoPointBVProblem(odef2, (boundary_two_point_a, boundary_two_point_b),
        u0, tspan; bcresid_prototype, nlls = Val(false))]

testTol = 0.2
affineTol = 1e-2
dts = 1 .// 2 .^ (3:-1:1)

export probArr, testTol, affineTol, dts, mirk_solver

end

@testitem "Affineness" setup=[MIRKConvergenceTests] begin
    using LinearAlgebra

    @testset "Problem: $i" for i in (1, 2, 5, 6)
        prob = probArr[i]
        @testset "MIRK$order" for order in (2, 3, 4, 5, 6)
            sol = solve(prob, mirk_solver(Val(order)); dt = 0.2)
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol[1][1] - 5) < affineTol
        end
    end
end

@testitem "JET: Runtime Dispatches" setup=[MIRKConvergenceTests] begin
    using JET

    @testset "Problem: $i" for i in 1:8
        prob = probArr[i]
        @testset "MIRK$order" for order in (2, 3, 4, 5, 6)
            solver = mirk_solver(Val(order); nlsolve = NewtonRaphson(),
                jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2)))
            @test_opt target_modules=(NonlinearSolve, BoundaryValueDiffEq) solve(
                prob, solver; dt = 0.2)
            @test_call target_modules=(NonlinearSolve, BoundaryValueDiffEq) solve(
                prob, solver; dt = 0.2)
        end
    end
end

@testitem "Convergence on Linear" setup=[MIRKConvergenceTests] begin
    using LinearAlgebra, DiffEqDevTools

    @testset "Problem: $i" for i in (3, 4, 7, 8)
        prob = probArr[i]
        @testset "MIRK$order" for (i, order) in enumerate((2, 3, 4, 5, 6))
            sim = test_convergence(
                dts, prob, mirk_solver(Val(order)); abstol = 1e-8, reltol = 1e-8)
            @test sim.𝒪est[:final]≈order atol=testTol
        end
    end
end

# FIXME: This is a really bad test. Needs interpolation
@testitem "Simple Pendulum" begin
    using StaticArrays

    tspan = (0.0, π / 2)
    function simplependulum!(du, u, p, t)
        g, L, θ, dθ = 9.81, 1.0, u[1], u[2]
        du[1] = dθ
        du[2] = -(g / L) * sin(θ)
    end

    function bc_pendulum!(residual, u, p, t)
        residual[1] = u[end ÷ 2][1] + π / 2 # the solution at the middle of the time span should be -pi/2
        residual[2] = u[end][1] - π / 2 # the solution at the end of the time span should be pi/2
    end

    u0 = MVector{2}([pi / 2, pi / 2])
    bvp1 = BVProblem(simplependulum!, bc_pendulum!, u0, tspan)

    jac_alg = BVPJacobianAlgorithm(;
        bc_diffmode = AutoFiniteDiff(), nonbc_diffmode = AutoSparse(AutoFiniteDiff()))

    # Using ForwardDiff might lead to Cache expansion warnings
    @test_nowarn solve(bvp1, MIRK2(; jac_alg); dt = 0.005)
    @test_nowarn solve(bvp1, MIRK3(; jac_alg); dt = 0.005)
    @test_nowarn solve(bvp1, MIRK4(; jac_alg); dt = 0.05)
    @test_nowarn solve(bvp1, MIRK5(; jac_alg); dt = 0.05)
    @test_nowarn solve(bvp1, MIRK6(; jac_alg); dt = 0.05)
end

@testitem "Interpolation" begin
    using LinearAlgebra

    λ = 1
    function prob_bvp_linear_analytic(u, λ, t)
        a = 1 / sqrt(λ)
        return [(exp(-a * t) - exp((t - 2) * a)) / (1 - exp(-2 * a)),
            (-a * exp(-t * a) - a * exp((t - 2) * a)) / (1 - exp(-2 * a))]
    end

    function prob_bvp_linear_f!(du, u, p, t)
        du[1] = u[2]
        du[2] = 1 / p * u[1]
    end
    function prob_bvp_linear_bc!(res, u, p, t)
        res[1] = u[1][1] - 1
        res[2] = u[end][1]
    end

    prob_bvp_linear_function = ODEFunction(
        prob_bvp_linear_f!, analytic = prob_bvp_linear_analytic)
    prob_bvp_linear_tspan = (0.0, 1.0)
    prob_bvp_linear = BVProblem(
        prob_bvp_linear_function, prob_bvp_linear_bc!, [1.0, 0.0], prob_bvp_linear_tspan, λ)
    testTol = 1e-6

    for order in (2, 3, 4, 5, 6)
        s = Symbol("MIRK$(order)")
        @eval mirk_solver(::Val{$order}) = $(s)()
    end

    @testset "MIRK$order" for order in (2, 3, 4, 5, 6)
        sol = solve(prob_bvp_linear, mirk_solver(Val(order)); dt = 0.001)
        @test sol(0.001)≈[0.998687464, -1.312035941] atol=testTol
        @test sol(0.001; idxs = [1, 2])≈[0.998687464, -1.312035941] atol=testTol
        @test sol(0.001; idxs = 1)≈0.998687464 atol=testTol
        @test sol(0.001; idxs = 2)≈-1.312035941 atol=testTol
    end
end

@testitem "Swirling Flow III" begin
    # Reported in https://github.com/SciML/BoundaryValueDiffEq.jl/issues/153
    eps = 0.01
    function swirling_flow!(du, u, p, t)
        eps = p
        du[1] = u[2]
        du[2] = (u[1] * u[4] - u[3] * u[2]) / eps
        du[3] = u[4]
        du[4] = u[5]
        du[5] = u[6]
        du[6] = (-u[3] * u[6] - u[1] * u[2]) / eps
        return
    end

    function swirling_flow_bc!(res, u, p, t)
        res[1] = u[1][1] + 1.0
        res[2] = u[1][3]
        res[3] = u[1][4]
        res[4] = u[end][1] - 1.0
        res[5] = u[end][3]
        res[6] = u[end][4]
        return
    end

    tspan = (0.0, 1.0)
    u0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    jac_alg = BVPJacobianAlgorithm(;
        bc_diffmode = AutoForwardDiff(), nonbc_diffmode = AutoSparse(AutoForwardDiff()))
    prob = BVProblem(swirling_flow!, swirling_flow_bc!, u0, tspan, eps)

    sol = solve(prob, MIRK4(; jac_alg); dt = 0.01)
    @test SciMLBase.successful_retcode(sol.retcode)
end

@testitem "Solve using Continuation" begin
    using RecursiveArrayTools

    g = 9.81
    L = 1.0
    tspan = (0.0, pi / 2)
    function simplependulum!(du, u, p, t)
        θ = u[1]
        dθ = u[2]
        du[1] = dθ
        du[2] = -(g / L) * sin(θ)
    end

    function bc2a!(resid_a, u_a, p) # u_a is at the beginning of the time span
        x0 = p
        resid_a[1] = u_a[1] - x0 # the solution at the beginning of the time span should be -pi/2
    end
    function bc2b!(resid_b, u_b, p) # u_b is at the ending of the time span
        x0 = p
        resid_b[1] = u_b[1] - pi / 2 # the solution at the end of the time span should be pi/2
    end

    bvp3 = TwoPointBVProblem(
        simplependulum!, (bc2a!, bc2b!), [pi / 2, pi / 2], (pi / 4, pi / 2),
        -pi / 2; bcresid_prototype = (zeros(1), zeros(1)))
    sol3 = solve(bvp3, MIRK4(), dt = 0.05)

    # Needs a SciMLBase fix
    bvp4 = TwoPointBVProblem(simplependulum!, (bc2a!, bc2b!), sol3, (0, pi / 2),
        pi / 2; bcresid_prototype = (zeros(1), zeros(1)))
    @test_broken solve(bvp4, MIRK4(), dt = 0.05) isa SciMLBase.ODESolution

    bvp5 = TwoPointBVProblem(simplependulum!, (bc2a!, bc2b!), DiffEqArray(sol3.u, sol3.t),
        (0, pi / 2), pi / 2; bcresid_prototype = (zeros(1), zeros(1)))
    @test SciMLBase.successful_retcode(solve(bvp5, MIRK4(), dt = 0.05).retcode)
end
