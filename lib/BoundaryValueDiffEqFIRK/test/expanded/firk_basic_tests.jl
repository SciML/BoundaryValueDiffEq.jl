@testsetup module FIRKExpandedConvergenceTests

using BoundaryValueDiffEqFIRK

nested = false

for stage in (2, 3, 4, 5)
    s = Symbol("LobattoIIIa$(stage)")
    @eval lobattoIIIa_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
end

for stage in (2, 3, 4, 5)
    s = Symbol("LobattoIIIb$(stage)")
    @eval lobattoIIIb_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
end

for stage in (2, 3, 4, 5)
    s = Symbol("LobattoIIIc$(stage)")
    @eval lobattoIIIc_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
end

for stage in (2, 3, 5, 7)
    s = Symbol("RadauIIa$(stage)")
    @eval radau_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
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
    residual[1] = u(0.0)[1] - 5
    residual[2] = u(5.0)[1]
end
boundary(u, p, t) = [u(0.0)[1] - 5, u(5.0)[1]]

# Array indexing for boundary conditions
function boundary_indexing!(residual, u, p, t)
    residual[1] = u[:, 1][1] - 5
    residual[2] = u[:, end][1]
end
boundary_indexing(u, p, t) = [u[:, 1][1] - 5, u[:, end][1]]

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

odef2! = ODEFunction(
    f2!, analytic = (
        u0, p, t,
    ) -> [5 * (cos(t) - cot(5) * sin(t)), 5 * (-cos(t) * cot(5) - sin(t))]
)
odef2 = ODEFunction(
    f2, analytic = (
        u0, p, t,
    ) -> [5 * (cos(t) - cot(5) * sin(t)), 5 * (-cos(t) * cot(5) - sin(t))]
)

bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

tspan = (0.0, 5.0)
u0 = [5.0, -3.5]

probArr = [
    BVProblem(odef1!, boundary!, u0, tspan, nlls = Val(false)),
    BVProblem(odef1, boundary, u0, tspan, nlls = Val(false)),
    BVProblem(odef2!, boundary!, u0, tspan, nlls = Val(false)),
    BVProblem(odef2, boundary, u0, tspan, nlls = Val(false)),
    BVProblem(odef2!, boundary_indexing!, u0, tspan, nlls = Val(false)),
    BVProblem(odef2, boundary_indexing, u0, tspan, nlls = Val(false)),
    TwoPointBVProblem(
        odef1!, (boundary_two_point_a!, boundary_two_point_b!),
        u0, tspan; bcresid_prototype, nlls = Val(false)
    ),
    TwoPointBVProblem(
        odef1, (boundary_two_point_a, boundary_two_point_b),
        u0, tspan; bcresid_prototype, nlls = Val(false)
    ),
    TwoPointBVProblem(
        odef2!, (boundary_two_point_a!, boundary_two_point_b!),
        u0, tspan; bcresid_prototype, nlls = Val(false)
    ),
    TwoPointBVProblem(
        odef2, (boundary_two_point_a, boundary_two_point_b),
        u0, tspan; bcresid_prototype, nlls = Val(false)
    ),
]

testTol = 0.3
affineTol = 1.0e-2
dts = 1 .// 2 .^ (5:-1:3)

export probArr, testTol, affineTol, dts, lobattoIIIa_solver, lobattoIIIb_solver,
    lobattoIIIc_solver, radau_solver

end

@testitem "Affineness" setup = [FIRKExpandedConvergenceTests] begin
    using LinearAlgebra

    @testset "Problem: $i" for i in (1, 2, 7, 8)
        prob = probArr[i]

        @testset "LobattoIIIa$stage" for stage in (2, 3, 4, 5)
            @time sol = solve(prob, lobattoIIIa_solver(Val(stage)); dt = 0.2, adaptive = false)
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < affineTol
        end
        @testset "LobattoIIIb$stage" for stage in (2, 3, 4, 5)
            @time sol = solve(prob, lobattoIIIb_solver(Val(stage)); dt = 0.2, adaptive = false)
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < affineTol
        end
        @testset "LobattoIIIc$stage" for stage in (2, 3, 4, 5)
            @time sol = solve(prob, lobattoIIIc_solver(Val(stage)); dt = 0.2, adaptive = false)
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < affineTol
        end

        @testset "RadauIIa$stage" for stage in (2, 3, 5, 7)
            @time sol = solve(prob, radau_solver(Val(stage)); dt = 0.2, adaptive = false)
            @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol.u[1][1] - 5) < affineTol
        end
    end
end

# JET tests have been moved to the separate QA test group (test/qa/)

@testitem "Convergence on Linear" setup = [FIRKExpandedConvergenceTests] begin
    using LinearAlgebra, DiffEqDevTools

    @testset "Problem: $i" for i in (3, 4, 9, 10)
        prob = probArr[i]

        @testset "LobattoIIIa$stage" for stage in (2, 3, 4, 5)
            @time sim = test_convergence(dts, prob, lobattoIIIa_solver(Val(stage)); abstol = 1.0e-8)
            if (stage == 5) || (((i == 9) || (i == 10)) && stage == 4)
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            end
        end

        @testset "LobattoIIIb$stage" for stage in (2, 3, 4, 5)
            @time sim = test_convergence(
                dts, prob, lobattoIIIb_solver(Val(stage)); abstol = 1.0e-8, reltol = 1.0e-8
            )
            if (stage == 5) || (stage == 4 && i == 10)
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            elseif stage == 4
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = 0.6
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            end
        end

        @testset "LobattoIIIc$stage" for stage in (2, 3, 4, 5)
            @time sim = test_convergence(
                dts, prob, lobattoIIIc_solver(Val(stage)); abstol = 1.0e-8, reltol = 1.0e-8
            )
            if stage == 4
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            elseif first(sim.errors[:final]) < 1.0e-12
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 2 atol = testTol
            end
        end

        @testset "RadauIIa$stage" for stage in (2, 3, 5, 7)
            @time sim = test_convergence(
                dts, prob, radau_solver(Val(stage)); abstol = 1.0e-8, reltol = 1.0e-8
            )
            if first(sim.errors[:final]) < 1.0e-12
                @test_broken sim.𝒪est[:final] ≈ 2 * stage - 1 atol = testTol
            else
                @test sim.𝒪est[:final] ≈ 2 * stage - 1 atol = testTol
            end
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
        residual[1] = u(pi / 4)[1] + π / 2 # the solution at the middle of the time span should be -pi/2
        residual[2] = u(pi / 2)[1] - π / 2 # the solution at the end of the time span should be pi/2
    end

    u0 = MVector{2}([pi / 2, pi / 2])
    bvp1 = BVProblem(simplependulum!, bc_pendulum!, u0, tspan)

    jac_alg = BVPJacobianAlgorithm(;
        bc_diffmode = AutoFiniteDiff(), nonbc_diffmode = AutoSparse(AutoFiniteDiff())
    )
    nlsolve = NewtonRaphson()
    nested = false

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

@testitem "Interpolation" setup = [FIRKExpandedConvergenceTests] begin
    using LinearAlgebra

    λ = 1
    function prob_bvp_linear_analytic(u, λ, t)
        a = 1 / sqrt(λ)
        return [
            (exp(-a * t) - exp((t - 2) * a)) / (1 - exp(-2 * a)),
            (-a * exp(-t * a) - a * exp((t - 2) * a)) / (1 - exp(-2 * a)),
        ]
    end

    function prob_bvp_linear_analytic_derivative(u, λ, t)
        a = 1 / sqrt(λ)
        return [
            (-a * exp(-t * a) - a * exp((t - 2) * a)) / (1 - exp(-2 * a)),
            (exp(-a * t) - exp((t - 2) * a)) / (1 - exp(-2 * a)),
        ]
    end

    function prob_bvp_linear_f!(du, u, p, t)
        du[1] = u[2]
        du[2] = 1 / p * u[1]
    end
    function prob_bvp_linear_bc!(res, u, p, t)
        res[1] = u(0.0)[1] - 1
        res[2] = u(1.0)[1]
    end

    prob_bvp_linear_function = ODEFunction(prob_bvp_linear_f!, analytic = prob_bvp_linear_analytic)
    prob_bvp_linear_tspan = (0.0, 1.0)
    prob_bvp_linear = BVProblem(
        prob_bvp_linear_function, prob_bvp_linear_bc!, [1.0, 0.0], prob_bvp_linear_tspan, λ
    )

    testTol = 1.0e-6
    nested = false

    @testset "Radau interpolations" begin
        @testset "Interpolation tests for RadauIIa$stage" for stage in (2, 3, 5, 7)
            @time sol = solve(prob_bvp_linear, radau_solver(Val(stage)); dt = 0.001)
            @test sol(0.001) ≈ [0.998687464, -1.312035941] atol = testTol
            @test sol(0.001; idxs = [1, 2]) ≈ [0.998687464, -1.312035941] atol = testTol
            @test sol(0.001; idxs = 1) ≈ 0.998687464 atol = testTol
            @test sol(0.001; idxs = 2) ≈ -1.312035941 atol = testTol
        end

        @testset "Derivative Interpolation tests for RadauIIa$stage" for stage in
            (2, 3, 5, 7)
            @time sol = solve(prob_bvp_linear, radau_solver(Val(stage)); dt = 0.001)
            sol_analytic = prob_bvp_linear_analytic(nothing, λ, 0.04)
            dsol_analytic = prob_bvp_linear_analytic_derivative(nothing, λ, 0.04)

            @test sol(0.04, Val{0}) ≈ sol_analytic atol = testTol
            @test sol(0.04, Val{1}) ≈ dsol_analytic atol = testTol
        end
    end

    @testset "LobattoIII interpolations" begin
        @testset "Interpolation tests for Lobatto" begin
            for (id, lobatto_solver) in zip(
                    ("a", "b", "c"), (
                        lobattoIIIa_solver, lobattoIIIb_solver, lobattoIIIc_solver,
                    )
                )
                begin
                    @testset "Interpolation tests for LobattoIII$(id)$stage" for stage in (
                            2, 3, 4, 5,
                        )
                        adaptive = ifelse(stage == 2, false, true) # LobattoIIIa2 is not adaptive
                        @time sol = solve(
                            prob_bvp_linear, lobatto_solver(Val(stage)); dt = 0.001, adaptive = adaptive
                        )
                        @test sol(0.001) ≈ [0.998687464, -1.312035941] atol = testTol
                        @test sol(0.001; idxs = [1, 2]) ≈ [0.998687464, -1.312035941] atol = testTol
                        @test sol(0.001; idxs = 1) ≈ 0.998687464 atol = testTol
                        @test sol(0.001; idxs = 2) ≈ -1.312035941 atol = testTol
                    end

                    @testset "Derivative Interpolation tests for lobatto$(id)$stage" for stage in
                        (
                            2, 3, 4, 5,
                        )
                        adaptive = ifelse(stage == 2, false, true) # LobattoIIIa2 is not adaptive
                        @time sol = solve(
                            prob_bvp_linear, lobatto_solver(Val(stage)); dt = 0.001, adaptive = adaptive
                        )
                        sol_analytic = prob_bvp_linear_analytic(nothing, λ, 0.04)
                        dsol_analytic = prob_bvp_linear_analytic_derivative(nothing, λ, 0.04)

                        @test sol(0.04, Val{0}) ≈ sol_analytic atol = testTol
                        @test sol(0.04, Val{1}) ≈ dsol_analytic atol = testTol
                    end
                end
            end
        end
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
        res[1] = u(0.0)[1] + 1.0
        res[2] = u(0.0)[3]
        res[3] = u(0.0)[4]
        res[4] = u(1.0)[1] - 1.0
        res[5] = u(1.0)[3]
        res[6] = u(1.0)[4]
        return
    end

    tspan = (0.0, 1.0)
    u0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    prob = BVProblem(swirling_flow!, swirling_flow_bc!, u0, tspan, eps)

    @test_nowarn solve(prob, RadauIIa5(); dt = 0.01)
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
        -pi / 2; bcresid_prototype = (zeros(1), zeros(1))
    )
    sol3 = solve(bvp3, RadauIIa5(), dt = 0.05)

    bvp4 = TwoPointBVProblem(
        simplependulum!, (bc2a!, bc2b!), sol3, (0, pi / 2),
        pi / 2; bcresid_prototype = (zeros(1), zeros(1))
    )
    @test SciMLBase.successful_retcode(solve(bvp4, RadauIIa5(), dt = 0.05))

    bvp5 = TwoPointBVProblem(
        simplependulum!, (bc2a!, bc2b!), DiffEqArray(sol3.u, sol3.t),
        (0, pi / 2), pi / 2; bcresid_prototype = (zeros(1), zeros(1))
    )
    @test SciMLBase.successful_retcode(solve(bvp5, RadauIIa5(), dt = 0.05))
end

@testitem "Test unknown parameters estimation" setup = [FIRKExpandedConvergenceTests] begin
    tspan = (0.0, pi)
    function f!(du, u, p, t)
        du[1] = u[2]
        du[2] = -(p[1] - 10 * cos(2 * t)) * u[1]
    end
    function bca!(res, u, p)
        res[1] = u[2]
        res[2] = u[1] - 1.0
    end
    function bcb!(res, u, p)
        res[1] = u[2]
    end
    function guess(p, t)
        return [cos(4t); -4sin(4t)]
    end
    bvp = TwoPointBVProblem(
        f!, (bca!, bcb!), guess, tspan, [15.0],
        bcresid_prototype = (zeros(2), zeros(1)), tune_parameters = true
    )
    sol = solve(bvp, RadauIIa5(), dt = 0.05)

    @test sol.prob.p ≈ [17.09658] atol = 1.0e-5

    tspan = (0.0, pi)
    function f!(du, u, p, t)
        du[1] = u[2]
        du[2] = -(p[1] - 10 * cos(2 * t)) * u[1]
    end
    function bc!(res, u, p, t)
        res[1] = u(0.0)[2]
        res[2] = u(0.0)[1] - 1.0
        res[3] = u(pi)[2]
    end
    function guess(p, t)
        return [cos(4t); -4sin(4t)]
    end
    bvp = TwoPointBVProblem(
        f!, (bca!, bcb!), guess, tspan, [15.0],
        bcresid_prototype = (zeros(2), zeros(1)), tune_parameters = true
    )
    sol = solve(bvp, RadauIIa5(), dt = 0.05)

    @test sol.prob.p ≈ [17.09658] atol = 1.0e-5
end

@testitem "Test unknown parameters estimation with SciMLStructures" begin
    using BoundaryValueDiffEqFIRK, SciMLStructures

    # Define a custom struct that wraps parameters
    struct MyParams{T}
        params::T
    end

    # Implement SciMLStructures interface
    SciMLStructures.isscimlstructure(::MyParams) = true
    SciMLStructures.ismutablescimlstructure(::MyParams) = false
    function SciMLStructures.canonicalize(::SciMLStructures.Tunable, p::MyParams)
        repack = let p = p
            (newp) -> MyParams(newp)
        end
        return p.params, repack, false
    end

    # Problem setup (same as vector test)
    tspan = (0.0, pi)
    function f!(du, u, p, t)
        params = p isa MyParams ? p.params : p
        du[1] = u[2]
        du[2] = -(params[1] - 10 * cos(2 * t)) * u[1]
    end
    function bca!(res, u, p)
        res[1] = u[2]
        res[2] = u[1] - 1.0
    end
    function bcb!(res, u, p)
        res[1] = u[2]
    end
    function guess(p, t)
        return [cos(4t); -4sin(4t)]
    end

    # Solve with plain vector
    bvp_vec = TwoPointBVProblem(
        f!, (bca!, bcb!), guess, tspan, [15.0],
        bcresid_prototype = (zeros(2), zeros(1)), tune_parameters = true
    )
    sol_vec = solve(bvp_vec, RadauIIa5(), dt = 0.05)

    # Solve with SciMLStructures-compatible struct
    bvp_struct = TwoPointBVProblem(
        f!, (bca!, bcb!), guess, tspan, MyParams([15.0]),
        bcresid_prototype = (zeros(2), zeros(1)), tune_parameters = true
    )
    sol_struct = solve(bvp_struct, RadauIIa5(), dt = 0.05)

    # Both should give the same result
    @test sol_vec.prob.p ≈ [17.09658] atol = 1.0e-5
    @test sol_struct.prob.p isa MyParams
    @test sol_struct.prob.p.params ≈ [17.09658] atol = 1.0e-5
    @test sol_struct.prob.p.params ≈ sol_vec.prob.p atol = 1.0e-10
end

"""
@testitem "Test initial guess" begin
    tspan = (0.0, 1.0)
    function f!(du, u, p, t)
        cond = 0.002
        vol_heat = 0.2
        du[1] = -u[2] / cond
        du[2] = vol_heat
        du[3] = 0.0
    end
    function bca!(res_a, u_a, p)
        res_a[1] = u_a[2]
        res_a[2] = u_a[1] - 100.0
    end
    function bcb!(res_b, u_b, p)
        tref = 20.0
        res_b[1] = u_b[3] * (u_b[1] - tref) - u_b[2]
    end
    u_guess = [
        [100.0, 0.0, 0.006666666666666668],
        [99.5, 0.020000000000000004, 0.006666666666666668],
        [98.0, 0.04000000000000001, 0.006666666666666668],
        [95.5, 0.060000000000000005, 0.006666666666666668],
        [92.0, 0.08000000000000002, 0.006666666666666668],
        [87.5, 0.1, 0.006666666666666668],
        [82.0, 0.12000000000000001, 0.006666666666666668],
        [75.5, 0.14, 0.006666666666666668],
        [68.0, 0.16000000000000003, 0.006666666666666668],
        [59.49999999999999, 0.18000000000000002, 0.006666666666666668],
        [50.0, 0.2, 0.006666666666666668],
    ]

    bvp1 = TwoPointBVProblem(f!, (bca!, bcb!), u_guess, tspan; bcresid_prototype = (zeros(2), zeros(1)))
    sol1 = solve(bvp1, RadauIIa5(), dt = 0.1, adaptive = false, nlsolve_kwargs = (; maxiters = 0))
    @test sol1.u == u_guess

    bvp2 = TwoPointBVProblem(f!, (bca!, bcb!), sol1, tspan; bcresid_prototype = (zeros(2), zeros(1)))
    sol2 = solve(bvp2, RadauIIa5(), dt = 0.1, adaptive = false, nlsolve_kwargs = (; maxiters = 0))
    @test sol2.u == u_guess
end
"""
