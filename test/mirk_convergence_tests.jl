using BoundaryValueDiffEq, DiffEqBase, DiffEqDevTools, LinearAlgebra, Test

for order in (2, 3, 4, 5, 6)
    s = Symbol("MIRK$(order)")
    @eval mirk_solver(::Val{$order}) = $(s)()
end

# First order test
function func_1!(du, u, p, t)
    du[1] = u[2]
    du[2] = 0
end

# Not able to change the initial condition.
# Hard coded solution.
func_1 = ODEFunction(func_1!, analytic = (u0, p, t) -> [5 - t, -1])

function boundary!(residual, u, p, t)
    residual[1] = u[1][1] - 5
    residual[2] = u[end][1]
end

function boundary_two_point!(residual, u, p, t)
    ua = u[1]
    ub = u[end]
    residual[1] = ua[1] - 5
    residual[2] = ub[1]
end

# Second order linear test
function func_2!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
end

# Not able to change the initial condition.
# Hard coded solution.
func_2 = ODEFunction(func_2!,
    analytic = (u0, p, t) -> [5 * (cos(t) - cot(5) * sin(t)),
        5 * (-cos(t) * cot(5) - sin(t))])
tspan = (0.0, 5.0)
u0 = [5.0, -3.5]
probArr = [BVProblem(func_1, boundary!, u0, tspan),
    BVProblem(func_2, boundary!, u0, tspan),
    TwoPointBVProblem(func_1, boundary_two_point!, u0, tspan),
    TwoPointBVProblem(func_2, boundary_two_point!, u0, tspan)]

testTol = 0.2
affineTol = 1e-2
dts = 1 .// 2 .^ (3:-1:1)

@info "Collocation method (MIRK)"

@testset "Affineness" begin
    @testset "Problem: $i" for i in (1, 3)
        prob = probArr[i]
        @testset "MIRK$order" for order in (2, 3, 4, 5, 6)
            @time sol = solve(prob, mirk_solver(Val(order)), dt = 0.2)
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol[1][1] - 5) < affineTol
        end
    end
end

@testset "Convergence on Linear" begin
    @testset "Problem: $i" for i in (2, 4)
        prob = probArr[i]
        @testset "MIRK$order" for (i, order) in enumerate((2, 3, 4, 5, 6))
            @time sim = test_convergence(dts, prob, mirk_solver(Val(order));
                abstol = 1e-8, reltol = 1e-8)
            @test sim.𝒪est[:final]≈order atol=testTol
        end
    end
end

# Simple Pendulum
using StaticArrays

tspan = (0.0, π / 2)
function simplependulum!(du, u, p, t)
    g, L, θ, dθ = 9.81, 1.0, u[1], u[2]
    du[1] = dθ
    du[2] = -(g / L) * sin(θ)
end

function bc1!(residual, u, p, t)
    residual[1] = u[end ÷ 2][1] + π / 2 # the solution at the middle of the time span should be -pi/2
    residual[2] = u[end][1] - π / 2 # the solution at the end of the time span should be pi/2
end

u0 = MVector{2}([pi / 2, pi / 2])
bvp1 = BVProblem(simplependulum!, bc1!, u0, tspan)

jac_alg = MIRKJacobianComputationAlgorithm(; bc_diffmode = AutoFiniteDiff(),
    collocation_diffmode = AutoSparseFiniteDiff())

# Using ForwardDiff might lead to Cache expansion warnings
@test_nowarn solve(bvp1, MIRK2(; jac_alg); dt = 0.005)
@test_nowarn solve(bvp1, MIRK3(; jac_alg); dt = 0.005)
@test_nowarn solve(bvp1, MIRK4(; jac_alg); dt = 0.05)
@test_nowarn solve(bvp1, MIRK5(; jac_alg); dt = 0.05)
@test_nowarn solve(bvp1, MIRK6(; jac_alg); dt = 0.05)
