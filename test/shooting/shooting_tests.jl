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
        @test norm(reduce(vcat, resid_f)) < 1e-11
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
        @test norm(resid_f) < 1e-11
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

    nlsolve = NewtonRaphson(; autodiff = AutoFiniteDiff())
    jac_alg = BVPJacobianAlgorithm(; bc_diffmode = AutoFiniteDiff(),
        nonbc_diffmode = AutoSparseFiniteDiff())
    for solver in [Shooting(Tsit5(); nlsolve),
        MultipleShooting(10, Tsit5(); nlsolve, jac_alg)]
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

@testset "Ray Tracing BVP" begin
    # Example 1.7 from
    # "Numerical Solution to Boundary Value Problems for Ordinary Differential equations",
    # 'Ascher, Mattheij, Russell'

    # Earthquake happens at known position (x0, y0, z0)
    # Earthquake is detected by seismograph at (xi, yi, zi)

    # Find the path taken by the first ray that reached seismograph.
    # i.e. given some velocity field finds the quickest path from
    # (x0,y0,z0) to (xi, yi, zi)

    # du = [dx, dy, dz, dξ, dη, dζ, dT, dS]
    # du = [x, y, z, ξ, η, ζ, T, S]
    # p = [ν(x,y,z), μ_x(x,y,z), μ_y(x,y,z), μ_z(x,y,z)]
    @inline v(x, y, z, p) = 1 / (4 + cos(p[1] * x) + sin(p[2] * y) - cos(p[3] * z))
    @inline ux(x, y, z, p) = -p[1] * sin(p[1] * x)
    @inline uy(x, y, z, p) = p[2] * cos(p[2] * y)
    @inline uz(x, y, z, p) = p[3] * sin(p[3] * z)

    function ray_tracing(u, p, t)
        du = similar(u)
        ray_tracing!(du, u, p, t)
        return du
    end

    function ray_tracing!(du, u, p, t)
        x, y, z, ξ, η, ζ, T, S = u

        nu = v(x, y, z, p) # Velocity of a sound wave, function of space;
        μx = ux(x, y, z, p) # ∂(slowness)/∂x, function of space
        μy = uy(x, y, z, p) # ∂(slowness)/∂y, function of space
        μz = uz(x, y, z, p) # ∂(slowness)/∂z, function of space

        du[1] = S * nu * ξ
        du[2] = S * nu * η
        du[3] = S * nu * ζ

        du[4] = S * μx
        du[5] = S * μy
        du[6] = S * μz

        du[7] = S / nu
        du[8] = 0

        return nothing
    end

    function ray_tracing_bc(sol, p, t)
        res = similar(first(sol))
        ray_tracing_bc!(res, sol, p, t)
        return res
    end

    function ray_tracing_bc!(res, sol, p, t)
        ua = sol(0.0)
        ub = sol(1.0)
        nu = v(ua[1], ua[2], ua[3], p) # Velocity of a sound wave, function of space;

        res[1] = ua[1] - x0
        res[2] = ua[2] - y0
        res[3] = ua[3] - z0
        res[4] = ua[7]      # T(0) = 0
        res[5] = ua[4]^2 + ua[5]^2 + ua[6]^2 - 1 / nu^2
        res[6] = ub[1] - xi
        res[7] = ub[2] - yi
        res[8] = ub[3] - zi
        return nothing
    end

    a = 0
    b = 1
    c = 2
    x0 = 0
    y0 = 0
    z0 = 0
    xi = 4
    yi = 3
    zi = 2.0
    p = [a, b, c, x0, y0, z0, xi, yi, zi]

    dx = xi - x0
    dy = yi - y0
    dz = zi - z0

    u0 = zeros(8)
    u0[1:3] .= 0 # position
    u0[4] = dx / v(x0, y0, z0, p)
    u0[5] = dy / v(x0, y0, z0, p)
    u0[6] = dz / v(x0, y0, z0, p)
    u0[8] = 1

    tspan = (0.0, 1.0)

    prob_oop = BVProblem{false}(ray_tracing, ray_tracing_bc, u0, tspan, p)
    alg = MultipleShooting(16, AutoVern7(Rodas4P()); nlsolve = NewtonRaphson(),
        grid_coarsening = Base.Fix2(div, 3))

    sol = solve(prob_oop, alg; reltol = 1e-6, abstol = 1e-6)
    @test SciMLBase.successful_retcode(sol.retcode)
    resid = zeros(8)
    ray_tracing_bc!(resid, sol, p, sol.t)
    @test norm(resid, Inf) < 1e-6

    prob_iip = BVProblem{true}(ray_tracing!, ray_tracing_bc!, u0, tspan, p)
    alg = MultipleShooting(16, AutoVern7(Rodas4P()); nlsolve = NewtonRaphson(),
        grid_coarsening = Base.Fix2(div, 3))

    sol = solve(prob_iip, alg; reltol = 1e-6, abstol = 1e-6)
    @test SciMLBase.successful_retcode(sol.retcode)
    resid = zeros(8)
    ray_tracing_bc!(resid, sol, p, sol.t)
    @test norm(resid, Inf) < 1e-6
end
