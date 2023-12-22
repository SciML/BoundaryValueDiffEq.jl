using BoundaryValueDiffEq, LinearAlgebra, OrdinaryDiffEq, Test

@testset "Overconstrained BVP" begin
    SOLVERS = [
        Shooting(Tsit5()),
        Shooting(Tsit5(); nlsolve = GaussNewton()),
        Shooting(Tsit5(); nlsolve = TrustRegion()),
        MultipleShooting(10, Tsit5()),
        MultipleShooting(10, Tsit5(); nlsolve = GaussNewton()),
        MultipleShooting(10, Tsit5(); nlsolve = TrustRegion())]

    @info "Solving Overconstrained BVPs"

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

    bvp1 = BVProblem(BVPFunction{false}(f1, bc1; bcresid_prototype = zeros(4)), u0, tspan;
        nlls = Val(true))

    for solver in SOLVERS
        @info "Testing $solver"
        sol = @time solve(bvp1, solver; verbose = false, abstol = 1e-6, reltol = 1e-6,
            odesolve_kwargs = (; abstol = 1e-6, reltol = 1e-6))
        @test norm(sol.resid, Inf) < 0.005
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

    bvp2 = BVProblem(BVPFunction{true}(f1!, bc1!; bcresid_prototype = zeros(4)), u0, tspan;
        nlls = Val(true))

    for solver in SOLVERS
        @info "Testing $solver"
        sol = @time solve(bvp2, solver; verbose = false, abstol = 1e-6, reltol = 1e-6,
            odesolve_kwargs = (; abstol = 1e-6, reltol = 1e-6))
        @test norm(sol.resid, Inf) < 0.005
    end

    # OOP TP-BVP
    bc1a(ua, p) = [ua[1]]
    bc1b(ub, p) = [ub[1] - 1, ub[2] + 1.729109]

    bvp3 = BVProblem(BVPFunction{false}(f1, (bc1a, bc1b); twopoint = Val(true),
            bcresid_prototype = (zeros(1), zeros(2))), u0, tspan; nlls = Val(true))

    for solver in SOLVERS
        @info "Testing $solver"
        sol = @time solve(bvp3, solver; verbose = false)
        @test norm(sol.resid, Inf) < 1e-4
    end

    # IIP TP-BVP
    bc1a!(resid, ua, p) = (resid[1] = ua[1])
    bc1b!(resid, ub, p) = (resid[1] = ub[1] - 1; resid[2] = ub[2] + 1.729109)

    bvp4 = BVProblem(BVPFunction{true}(f1!, (bc1a!, bc1b!); twopoint = Val(true),
            bcresid_prototype = (zeros(1), zeros(2))), u0, tspan; nlls = Val(true))

    for solver in SOLVERS
        @info "Testing $solver"
        sol = @time solve(bvp4, solver; verbose = false)
        @test norm(sol.resid, Inf) < 1e-4
    end
end
