@testitem "Overconstrained BVP" begin
    using BoundaryValueDiffEqMIRK, LinearAlgebra

    SOLVERS = [mirk(; nlsolve)
               for mirk in (MIRK4, MIRK5, MIRK6),
    nlsolve in (LevenbergMarquardt(), GaussNewton(), TrustRegion(), nothing)]

    # OOP MP-BVP
    f1(u, p, t) = [u[2], -u[1]]

    function bc1(sol, p, t)
        solₜ₁ = sol[:, 1]
        solₜ₂ = sol[:, end]
        return [solₜ₁[1], solₜ₂[1] - 1, solₜ₂[2] + 1.729109]
    end

    tspan = (0.0, 100.0)
    u0 = [0.0, 1.0]

    bvp1 = BVProblem(BVPFunction{false}(f1, bc1; bcresid_prototype = zeros(3)), u0, tspan)

    for solver in SOLVERS
        sol = solve(bvp1, solver; verbose = false, dt = 1.0)
        @test norm(bc1(sol, nothing, sol.t), Inf) < 1e-2
    end

    # IIP MP-BVP
    function f1!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
        return nothing
    end

    function bc1!(resid, sol, p, t)
        solₜ₁ = sol[:, 1]
        solₜ₂ = sol[:, end]
        # We know that this overconstrained system has a solution
        resid[1] = solₜ₁[1]
        resid[2] = solₜ₂[1] - 1
        resid[3] = solₜ₂[2] + 1.729109
        return nothing
    end

    bvp2 = BVProblem(BVPFunction{true}(f1!, bc1!; bcresid_prototype = zeros(3)), u0, tspan)

    for solver in SOLVERS
        sol = solve(bvp2, solver; verbose = false, dt = 1.0)
        resid_f = Array{Float64}(undef, 3)
        bc1!(resid_f, sol, nothing, sol.t)
        @test norm(resid_f, Inf) < 1e-2
    end

    # OOP TP-BVP
    bc1a(ua, p) = [ua[1]]
    bc1b(ub, p) = [ub[1] - 1, ub[2] + 1.729109]

    bvp3 = TwoPointBVProblem(
        BVPFunction{false}(f1, (bc1a, bc1b); twopoint = Val(true),
            bcresid_prototype = (zeros(1), zeros(2))),
        u0,
        tspan)

    for solver in SOLVERS
        sol = solve(bvp3, solver; verbose = false, dt = 1.0)
        @test norm(vcat(bc1a(sol.u[1], nothing), bc1b(sol.u[end], nothing)), Inf) < 1e-2
    end

    # IIP TP-BVP
    bc1a!(resid, ua, p) = (resid[1] = ua[1])
    bc1b!(resid, ub, p) = (resid[1] = ub[1] - 1; resid[2] = ub[2] + 1.729109)

    bvp4 = TwoPointBVProblem(
        BVPFunction{true}(f1!, (bc1a!, bc1b!); twopoint = Val(true),
            bcresid_prototype = (zeros(1), zeros(2))),
        u0,
        tspan)

    for solver in SOLVERS
        sol = solve(bvp3, solver; verbose = false, dt = 1.0, abstol = 1e-3, reltol = 1e-3)
        resida = Array{Float64}(undef, 1)
        residb = Array{Float64}(undef, 2)
        bc1a!(resida, sol(0.0), nothing)
        bc1b!(residb, sol(100.0), nothing)
        @test norm(vcat(resida, residb), Inf) < 1e-2
    end
end

# This is not a very meaningful problem, but it tests that our solvers are not throwing an
# error
@testitem "Underconstrained BVP: Rod BVP" begin
    using LinearAlgebra

    # Force normal form for GN
    SOLVERS = [mirk(; nlsolve)
               for mirk in (MIRK4, MIRK5, MIRK6),
    nlsolve in (TrustRegion(), GaussNewton(), LevenbergMarquardt(), nothing)]

    function hat(y)
        return [0 -y[3] y[2]
                y[3] 0 -y[1]
                -y[2] y[1] 0]
    end

    function inv_hat(skew)
        [skew[3, 2]; skew[1, 3]; skew[2, 1]]
    end

    function rod_ode!(dy, y, p, t, Kse_inv, Kbt_inv, rho, A, g)
        R = reshape(@view(y[4:12]), 3, 3)
        n = @view y[13:15]
        m = @view y[16:18]

        v = Kse_inv * R' * n
        v[3] += 1.0
        u = Kbt_inv * R' * m
        ps = R * v
        @views dy[1:3] .= ps
        @views dy[4:12] .= vec(R * hat(u))
        @views dy[13:15] .= -rho * A * g
        @views dy[16:18] .= -hat(ps) * n
    end

    function bc_a!(residual, y, p)
        # Extract rotations from y
        R0_u = reshape(@view(y[4:12]), 3, 3)

        # Extract rotations from p
        R0 = reshape(@view(p[4:12]), 3, 3)

        @views residual[1:3] = y[1:3] .- p[1:3]
        @views residual[4:6] = inv_hat(R0_u' * R0 - R0_u * R0')
        return nothing
    end

    function bc_b!(residual, y, p)
        # Extract rotations from y
        RL_u = reshape(@view(y[4:12]), 3, 3)

        # Extract rotations from p
        RL = reshape(@view(p[16:24]), 3, 3)

        @views residual[1:3] = y[1:3] .- p[13:15]
        @views residual[4:6] = inv_hat(RL_u' * RL - RL_u * RL')
        return nothing
    end

    function bc!(residual, sol, p, t)
        y1 = first(sol)
        y2 = last(sol)
        R0_u = reshape(@view(y1[4:12]), 3, 3)
        RL_u = reshape(@view(y2[4:12]), 3, 3)

        # Extract rotations from p
        R0 = reshape(@view(p[4:12]), 3, 3)
        RL = reshape(@view(p[16:24]), 3, 3)

        @views residual[1:3] = y1[1:3] .- p[1:3]
        @views residual[4:6] = inv_hat(R0_u' * R0 - R0_u * R0')
        @views residual[7:9] = y2[1:3] .- p[13:15]
        @views residual[10:12] = inv_hat(RL_u' * RL - RL_u * RL')

        return nothing
    end

    # Parameters
    E = 200e9
    G = 80e9
    r = 0.001
    rho = 8000
    g = [9.81; 0; 0]
    L = 0.5
    A = pi * r^2
    I = pi * r^4 / 4
    J = 2 * I
    Kse = diagm([G * A, G * A, E * A])
    Kbt = diagm([E * I, E * I, G * J])

    # Boundary Conditions
    p0 = [0; 0; 0]
    R0 = vec(LinearAlgebra.I(3))
    pL = [0; -0.1 * L; 0.8 * L]
    RL = vec(LinearAlgebra.I(3))

    # Main Simulation
    tspan = (0.0, L)
    rod_ode!(dy, y, p, t) = rod_ode!(dy, y, p, t, inv(Kse), inv(Kbt), rho, A, g)
    y0 = vcat(p0, R0, zeros(6))
    p = vcat(p0, R0, pL, RL)
    prob_tp = TwoPointBVProblem(
        rod_ode!, (bc_a!, bc_b!), y0, tspan, p, bcresid_prototype = (zeros(6), zeros(6)))
    prob = BVProblem(
        BVPFunction(rod_ode!, bc!; bcresid_prototype = zeros(12)), y0, tspan, p)

    for solver in SOLVERS
        sol = solve(
            prob_tp, solver; verbose = false, dt = 0.1, abstol = 1e-1, reltol = 1e-1)
        @test SciMLBase.successful_retcode(sol.retcode)
        sol = solve(prob, solver; verbose = false, dt = 0.1, abstol = 1e-1, reltol = 1e-1)
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end
