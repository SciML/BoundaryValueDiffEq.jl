using BoundaryValueDiffEq, LinearAlgebra, LinearSolve, OrdinaryDiffEq, Test

@testset "Overconstrained BVP" begin
    SOLVERS = [
        Shooting(Tsit5()),
        Shooting(Tsit5(); nlsolve = LevenbergMarquardt()),
        Shooting(Tsit5(); nlsolve = GaussNewton()),
        MultipleShooting(10, Tsit5()),
        MultipleShooting(10, Tsit5(); nlsolve = LevenbergMarquardt()),
        MultipleShooting(10, Tsit5(); nlsolve = GaussNewton())]

    # OOP MP-BVP
    f1(u, p, t) = [u[2], -u[1]]

    function bc1(sol, p, t)
        t₁, t₂ = extrema(t)
        solₜ₁ = sol(t₁)
        solₜ₂ = sol(t₂)
        solₜ₃ = sol((t₁ + t₂) / 2)
        # We know that this overconstrained system has a solution
        return [solₜ₁[1], solₜ₂[1] - 1, solₜ₃[1] - 0.51735, solₜ₃[2] + 1.92533]
    end

    tspan = (0.0, 100.0)
    u0 = [0.0, 1.0]

    bvp1 = BVProblem(BVPFunction{false}(f1, bc1; bcresid_prototype = zeros(4)), u0, tspan)

    for solver in SOLVERS
        @time sol = solve(bvp1, solver;
            nlsolve_kwargs = (; abstol = 1e-8, reltol = 1e-8, maxiters = 1000),
            verbose = false)
        @test norm(bc1(sol, nothing, sol.t)) < 1e-4
    end

    # IIP MP-BVP
    function f1!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
        return nothing
    end

    function bc1!(resid, sol, p, t)
        (t₁, t₂) = extrema(t)
        solₜ₁ = sol(t₁)
        solₜ₂ = sol(t₂)
        solₜ₃ = sol((t₁ + t₂) / 2)
        # We know that this overconstrained system has a solution
        resid[1] = solₜ₁[1]
        resid[2] = solₜ₂[1] - 1
        resid[3] = solₜ₃[1] - 0.51735
        resid[4] = solₜ₃[2] + 1.92533
        return nothing
    end

    bvp2 = BVProblem(BVPFunction{true}(f1!, bc1!; bcresid_prototype = zeros(4)), u0, tspan)

    for solver in SOLVERS
        @time sol = solve(bvp2, solver;
            nlsolve_kwargs = (; abstol = 1e-8, reltol = 1e-8, maxiters = 1000),
            verbose = false)
        resid_f = Array{Float64}(undef, 4)
        bc1!(resid_f, sol, nothing, sol.t)
        @test norm(resid_f) < 1e-4
    end

    # OOP TP-BVP
    bc1a(ua, p) = [ua[1]]
    bc1b(ub, p) = [ub[1] - 1, ub[2] + 1.729109]

    bvp3 = TwoPointBVProblem(BVPFunction{false}(f1, (bc1a, bc1b); twopoint = Val(true),
            bcresid_prototype = (zeros(1), zeros(2))), u0, tspan)

    for solver in SOLVERS
        @time sol = solve(bvp3, solver;
            nlsolve_kwargs = (; abstol = 1e-8, reltol = 1e-8, maxiters = 1000),
            verbose = false)
        @test norm(vcat(bc1a(sol(0.0), nothing), bc1b(sol(100.0), nothing))) < 1e-4
    end

    # IIP TP-BVP
    bc1a!(resid, ua, p) = (resid[1] = ua[1])
    bc1b!(resid, ub, p) = (resid[1] = ub[1] - 1; resid[2] = ub[2] + 1.729109)

    bvp4 = TwoPointBVProblem(BVPFunction{true}(f1!, (bc1a!, bc1b!); twopoint = Val(true),
            bcresid_prototype = (zeros(1), zeros(2))), u0, tspan)

    for solver in SOLVERS
        @time sol = solve(bvp4, solver;
            nlsolve_kwargs = (; abstol = 1e-8, reltol = 1e-8, maxiters = 1000),
            verbose = false)
        resida = Array{Float64}(undef, 1)
        residb = Array{Float64}(undef, 2)
        bc1a!(resida, sol(0.0), nothing)
        bc1b!(residb, sol(100.0), nothing)
        @test norm(vcat(resida, residb)) < 1e-4
    end
end
