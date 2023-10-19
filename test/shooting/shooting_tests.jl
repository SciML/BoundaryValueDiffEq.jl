using BoundaryValueDiffEq, LinearAlgebra, OrdinaryDiffEq, Test

@testset "Basic Shooting Tests" begin
    SOLVERS = [Shooting(Tsit5()), MultipleShooting(10, Tsit5())]

    tspan = (0.0, 100.0)
    u0 = [0.0, 1.0]

    # Inplace
    function f1!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
        return nothing
    end

    function bc1!(resid, sol, p, t)
        t₀, t₁ = first(t), last(t)
        resid[1] = sol(t₀)[1]
        resid[2] = sol(t₁)[1] - 1
        return nothing
    end

    bvp1 = BVProblem(f1!, bc1!, u0, tspan)
    @test SciMLBase.isinplace(bvp1)
    for solver in SOLVERS
        resid_f = Array{Float64}(undef, 2)
        sol = solve(bvp1, solver; abstol = 1e-13, reltol = 1e-13)
        @test SciMLBase.successful_retcode(sol)
        bc1!(resid_f, sol, nothing, sol.t)
        @test norm(resid_f) < 1e-12
    end

    # Out of Place
    f1(u, p, t) = [u[2], -u[1]]

    function bc1(sol, p, t)
        t₀, t₁ = first(t), last(t)
        return [sol(t₀)[1], sol(t₁)[1] - 1]
    end

    @test_throws SciMLBase.NonconformingFunctionsError BVProblem(f1!, bc1, u0, tspan)
    @test_throws SciMLBase.NonconformingFunctionsError BVProblem(f1, bc1!, u0, tspan)

    bvp2 = BVProblem(f1, bc1, u0, tspan)
    @test !SciMLBase.isinplace(bvp2)
    for solver in SOLVERS
        sol = solve(bvp2, solver; abstol = 1e-13, reltol = 1e-13)
        @test SciMLBase.successful_retcode(sol)
        resid_f = bc1(sol, nothing, sol.t)
        @test norm(resid_f) < 1e-12
    end

    # Inplace
    bc2a!(resid, ua, p) = (resid[1] = ua[1])
    bc2b!(resid, ub, p) = (resid[1] = ub[1] - 1)

    bvp3 = TwoPointBVProblem(f1!, (bc2a!, bc2b!), u0, tspan;
        bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1)))
    @test SciMLBase.isinplace(bvp3)
    for solver in SOLVERS
        sol = solve(bvp3, solver; abstol = 1e-13, reltol = 1e-13)
        @test SciMLBase.successful_retcode(sol)
        resid_f = (Array{Float64, 1}(undef, 1), Array{Float64, 1}(undef, 1))
        bc2a!(resid_f[1], sol(tspan[1]), nothing)
        bc2b!(resid_f[2], sol(tspan[2]), nothing)
        @test norm(reduce(vcat, resid_f)) < 1e-12
    end

    # Out of Place
    bc2a(ua, p) = [ua[1]]
    bc2b(ub, p) = [ub[1] - 1]

    bvp4 = TwoPointBVProblem(f1, (bc2a, bc2b), u0, tspan)
    @test !SciMLBase.isinplace(bvp4)
    for solver in SOLVERS
        sol = solve(bvp4, solver; abstol = 1e-13, reltol = 1e-13)
        @test SciMLBase.successful_retcode(sol)
        resid_f = reduce(vcat, (bc2a(sol(tspan[1]), nothing), bc2b(sol(tspan[2]), nothing)))
        @test norm(resid_f) < 1e-12
    end
end

@testset "Shooting with Complex Values" begin
    # Test for complex values
    function f1!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
        return nothing
    end

    function bc1!(resid, sol, p, t)
        t₀, t₁ = first(t), last(t)
        resid[1] = sol(t₀)[1]
        resid[2] = sol(t₁)[1] - 1
        return nothing
    end

    tspan = (0.0, 100.0)
    u0 = [0.0, 1.0] .+ 1im
    bvp = BVProblem(f1!, bc1!, u0, tspan)
    resid_f = Array{ComplexF64}(undef, 2)

    # We will automatically use FiniteDiff if we can't use dual numbers
    for solver in [Shooting(Tsit5()), MultipleShooting(10, Tsit5())]
        sol = solve(bvp, solver; abstol = 1e-13, reltol = 1e-13)
        @test SciMLBase.successful_retcode(sol)
        bc1!(resid_f, sol, nothing, sol.t)
        @test norm(resid_f) < 1e-12
    end
end

@testset "Flow In a Channel" begin
    function flow_in_a_channel!(du, u, p, t)
        R, P = p
        A, f′′, f′, f, h′, h, θ′, θ = u
        du[1] = 0
        du[2] = R * (f′^2 - f * f′′) - R * A
        du[3] = f′′
        du[4] = f′
        du[5] = -R * f * h′ - 1
        du[6] = h′
        du[7] = -P * f * θ′
        du[8] = θ′
    end

    function bc_flow!(resid, sol, p, tspan)
        t₁, t₂ = extrema(tspan)
        solₜ₁ = sol(t₁)
        solₜ₂ = sol(t₂)
        resid[1] = solₜ₁[4]
        resid[2] = solₜ₁[3]
        resid[3] = solₜ₂[4] - 1
        resid[4] = solₜ₂[3]
        resid[5] = solₜ₁[6]
        resid[6] = solₜ₂[6]
        resid[7] = solₜ₁[8]
        resid[8] = solₜ₂[8] - 1
    end

    tspan = (0.0, 1.0)
    p = [10.0, 7.0]
    u0 = zeros(8)

    flow_bvp = BVProblem{true}(flow_in_a_channel!, bc_flow!, u0, tspan, p)

    sol_shooting = solve(flow_bvp,
        Shooting(AutoTsit5(Rosenbrock23()), NewtonRaphson());
        maxiters = 100)
    @test SciMLBase.successful_retcode(sol_shooting)

    resid = zeros(8)
    bc_flow!(resid, sol_shooting, p, sol_shooting.t)
    @test norm(resid, Inf) < 1e-6

    sol_msshooting = solve(flow_bvp,
        MultipleShooting(10, AutoTsit5(Rosenbrock23()); nlsolve = NewtonRaphson());
        maxiters = 100)
    @test SciMLBase.successful_retcode(sol_msshooting)

    resid = zeros(8)
    bc_flow!(resid, sol_msshooting, p, sol_msshooting.t)
    @test norm(resid, Inf) < 1e-6
end
