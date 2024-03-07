using BoundaryValueDiffEq, DiffEqBase, DiffEqDevTools, LinearAlgebra, Test

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
		5 * (cos(t) - cot(5) * sin(t)),
		5 * (-cos(t) * cot(5) - sin(t)),
	])
odef2 = ODEFunction(f2,
	analytic = (u0, p, t) -> [
		5 * (cos(t) - cot(5) * sin(t)),
		5 * (-cos(t) * cot(5) - sin(t)),
	])

bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

tspan = (0.0, 5.0)
u0 = [5.0, -3.5]

probArr = [
	BVProblem(odef1!, boundary!, u0, tspan),
	BVProblem(odef1, boundary, u0, tspan),
	BVProblem(odef2!, boundary!, u0, tspan),
	BVProblem(odef2, boundary, u0, tspan),
	TwoPointBVProblem(odef1!, (boundary_two_point_a!, boundary_two_point_b!), u0, tspan;
		bcresid_prototype),
	TwoPointBVProblem(odef1, (boundary_two_point_a, boundary_two_point_b), u0, tspan;
		bcresid_prototype),
	TwoPointBVProblem(odef2!, (boundary_two_point_a!, boundary_two_point_b!), u0, tspan;
		bcresid_prototype),
	TwoPointBVProblem(odef2, (boundary_two_point_a, boundary_two_point_b), u0, tspan;
		bcresid_prototype),
];

testTol = 0.2
affineTol = 1e-2
dts = 1 .// 2 .^ (5:-1:3)

for stage in (2, 3, 4, 5)
	s = Symbol("LobattoIIIc$(stage)")
	@eval lobatto_solver(::Val{$stage}) = $(s)(NewtonRaphson(), BVPJacobianAlgorithm(AutoSparseForwardDiff()), false)
end

@testset "Affineness" begin
	@testset "Problem: $i" for i in (1, 2, 5, 6)
		prob = probArr[i]
		@testset "LobattoIIIc$stage" for stage in (3, 4, 5)
			@time sol = solve(prob, lobatto_solver(Val(stage)); dt = 0.2, adaptive = false)
			@test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol[1][1] - 5) < affineTol
		end
	end
end

@testset "Convergence on Linear" begin
	@testset "Problem: $i" for i in (3, 4, 7, 8)
		prob = probArr[i]
		@testset "LobattoIIIc$stage" for stage in (2, 3, 4, 5)
			@time sim = test_convergence(dts, prob, lobatto_solver(Val(stage));
				abstol = 1e-8, reltol = 1e-8)
			if first(sim.errors[:final]) > 1e-12
				@test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
			end
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

# FIXME: This is a really bad test. Needs interpolation
function bc_pendulum!(residual, u, p, t)
	residual[1] = u[end÷2][1] + π / 2 # the solution at the middle of the time span should be -pi/2
	residual[2] = u[end][1] - π / 2 # the solution at the end of the time span should be pi/2
end

u0 = MVector{2}([pi / 2, pi / 2])
bvp1 = BVProblem(simplependulum!, bc_pendulum!, u0, tspan)

jac_alg = BVPJacobianAlgorithm(AutoFiniteDiff(); bc_diffmode = AutoFiniteDiff(),
	nonbc_diffmode = AutoSparseFiniteDiff())

nl_solve = NewtonRaphson()

# Using ForwardDiff might lead to Cache expansion warnings
@test_nowarn solve(bvp1, LobattoIIIc2(nl_solve, jac_alg, true); dt = 0.005)
@test_nowarn solve(bvp1, LobattoIIIc3(nl_solve, jac_alg, true); dt = 0.005)
@test_nowarn solve(bvp1, LobattoIIIc4(nl_solve, jac_alg, true); dt = 0.005)
@test_nowarn solve(bvp1, LobattoIIIc5(nl_solve, jac_alg, true); dt = 0.005)

@test_nowarn solve(bvp1, LobattoIIIc2(nl_solve, jac_alg, false); dt = 0.005)
@test_nowarn solve(bvp1, LobattoIIIc3(nl_solve, jac_alg, false); dt = 0.005)
@test_nowarn solve(bvp1, LobattoIIIc4(nl_solve, jac_alg, false); dt = 0.005)
@test_nowarn solve(bvp1, LobattoIIIc5(nl_solve, jac_alg, false); dt = 0.005)
