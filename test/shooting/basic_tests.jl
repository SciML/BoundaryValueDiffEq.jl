using BoundaryValueDiffEq, LinearAlgebra, LinearSolve, OrdinaryDiffEq, Test

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

    @info "Basic MP Shooting IIP"

    bvp1 = BVProblem(f1!, bc1!, u0, tspan)
    @test SciMLBase.isinplace(bvp1)
    for solver in SOLVERS
        @info "Solver: $solver"
        resid_f = Array{Float64}(undef, 2)
        sol = @time solve(bvp1, solver; abstol = 1e-13, reltol = 1e-13,
            odesolve_kwargs = (; abstol = 1e-6, reltol = 1e-3))
        @test SciMLBase.successful_retcode(sol)
        bc1!(resid_f, sol, nothing, sol.t)
        @test norm(resid_f, Inf) < 1e-12
    end

    # Out of Place
    f1(u, p, t) = [u[2], -u[1]]

    function bc1(sol, p, t)
        t₀, t₁ = first(t), last(t)
        return [sol(t₀)[1], sol(t₁)[1] - 1]
    end

    @test_throws SciMLBase.NonconformingFunctionsError BVProblem(f1!, bc1, u0, tspan)
    @test_throws SciMLBase.NonconformingFunctionsError BVProblem(f1, bc1!, u0, tspan)

    @info "Basic MP Shooting OOP"

    bvp2 = BVProblem(f1, bc1, u0, tspan)
    @test !SciMLBase.isinplace(bvp2)
    for solver in SOLVERS
        @info "Solver: $solver"
        sol = @time solve(bvp2, solver; abstol = 1e-13, reltol = 1e-13,
            odesolve_kwargs = (; abstol = 1e-6, reltol = 1e-3))
        @test SciMLBase.successful_retcode(sol)
        resid_f = bc1(sol, nothing, sol.t)
        @test norm(resid_f, Inf) < 1e-12
    end

    # Inplace
    bc2a!(resid, ua, p) = (resid[1] = ua[1])
    bc2b!(resid, ub, p) = (resid[1] = ub[1] - 1)

    @info "Basic TP Shooting IIP"

    bvp3 = TwoPointBVProblem(f1!, (bc2a!, bc2b!), u0, tspan;
        bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1)))
    @test SciMLBase.isinplace(bvp3)
    for solver in SOLVERS
        @info "Solver: $solver"
        sol = @time solve(bvp3, solver; abstol = 1e-13, reltol = 1e-13,
            odesolve_kwargs = (; abstol = 1e-6, reltol = 1e-3))
        @test SciMLBase.successful_retcode(sol)
        resid_f = (Array{Float64, 1}(undef, 1), Array{Float64, 1}(undef, 1))
        bc2a!(resid_f[1], sol(tspan[1]), nothing)
        bc2b!(resid_f[2], sol(tspan[2]), nothing)
        @test norm(reduce(vcat, resid_f), Inf) < 1e-12
    end

    # Out of Place
    bc2a(ua, p) = [ua[1]]
    bc2b(ub, p) = [ub[1] - 1]

    @info "Basic TP Shooting OOP"

    bvp4 = TwoPointBVProblem(f1, (bc2a, bc2b), u0, tspan)
    @test !SciMLBase.isinplace(bvp4)
    for solver in SOLVERS
        @info "Solver: $solver"
        sol = @time solve(bvp4, solver; abstol = 1e-13, reltol = 1e-13,
            odesolve_kwargs = (; abstol = 1e-6, reltol = 1e-3))
        @test SciMLBase.successful_retcode(sol)
        resid_f = reduce(vcat, (bc2a(sol(tspan[1]), nothing), bc2b(sol(tspan[2]), nothing)))
        @test norm(resid_f, Inf) < 1e-12
    end
end

@testset "Shooting with Complex Values" begin
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

    @info "Shooting with Complex Values"

    for solver in [Shooting(Vern7()), MultipleShooting(10, Vern7())]
        @info "Solver: $solver"
        sol = @time solve(bvp, solver; abstol = 1e-13, reltol = 1e-13,
            odesolve_kwargs = (; abstol = 1e-8, reltol = 1e-8))
        @test SciMLBase.successful_retcode(sol)
        bc1!(resid_f, sol, nothing, sol.t)
        @test norm(resid_f, Inf) < 1e-12
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

    @info "Flow in a Channel"

    for solver in [
        Shooting(AutoTsit5(Rosenbrock23())),
        MultipleShooting(10, AutoTsit5(Rosenbrock23())),
    ]
        @info "Solver: $solver"
        sol = @time solve(flow_bvp, solver; abstol = 1e-13, reltol = 1e-13,
            odesolve_kwargs = (; abstol = 1e-8, reltol = 1e-8))
        @test SciMLBase.successful_retcode(sol)
        resid = zeros(8)
        bc_flow!(resid, sol, p, sol.t)
        @test norm(resid, Inf) < 1e-6
    end
end

@testset "Ray Tracing" begin
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

        res[1] = ua[1] - p[4]
        res[2] = ua[2] - p[5]
        res[3] = ua[3] - p[6]
        res[4] = ua[7]      # T(0) = 0
        res[5] = ua[4]^2 + ua[5]^2 + ua[6]^2 - 1 / nu^2
        res[6] = ub[1] - p[7]
        res[7] = ub[2] - p[8]
        res[8] = ub[3] - p[9]
        return nothing
    end

    function ray_tracing_bc_a(ua, p)
        resa = similar(ua, 5)
        ray_tracing_bc_a!(resa, ua, p)
        return resa
    end

    function ray_tracing_bc_a!(resa, ua, p)
        nu = v(ua[1], ua[2], ua[3], p) # Velocity of a sound wave, function of space;

        resa[1] = ua[1] - p[4]
        resa[2] = ua[2] - p[5]
        resa[3] = ua[3] - p[5]
        resa[4] = ua[7]
        resa[5] = ua[4]^2 + ua[5]^2 + ua[6]^2 - 1 / nu^2

        return nothing
    end

    function ray_tracing_bc_b(ub, p)
        resb = similar(ub, 3)
        ray_tracing_bc_b!(resb, ub, p)
        return resb
    end

    function ray_tracing_bc_b!(resb, ub, p)
        resb[1] = ub[1] - p[7]
        resb[2] = ub[2] - p[8]
        resb[3] = ub[3] - p[9]
        return nothing
    end

    p = [0, 1, 2, 0, 0, 0, 4, 3, 2.0]

    dx = p[7] - p[4]
    dy = p[8] - p[5]
    dz = p[9] - p[6]

    u0 = zeros(8)
    u0[1:3] .= 0 # position
    u0[4] = dx / v(p[4], p[5], p[6], p)
    u0[5] = dy / v(p[4], p[5], p[6], p)
    u0[6] = dz / v(p[4], p[5], p[6], p)
    u0[8] = 1

    tspan = (0.0, 1.0)

    prob_oop = BVProblem(BVPFunction{false}(ray_tracing, ray_tracing_bc), u0, tspan, p;
        nlls = Val(false))
    prob_iip = BVProblem(BVPFunction{true}(ray_tracing!, ray_tracing_bc!), u0, tspan, p;
        nlls = Val(true))
    prob_tp_oop = BVProblem(BVPFunction{false}(ray_tracing,
            (ray_tracing_bc_a, ray_tracing_bc_b); twopoint = Val(true)), u0, tspan, p;
        nlls = Val(true))
    prob_tp_iip = BVProblem(BVPFunction{true}(ray_tracing!,
            (ray_tracing_bc_a!, ray_tracing_bc_b!);
            bcresid_prototype = (zeros(5), zeros(3)), twopoint = Val(true)), u0, tspan, p;
        nlls = Val(true))

    @info "Ray Tracing: Multiple Shooting"

    alg_sp = MultipleShooting(10, AutoVern7(Rodas4P()); grid_coarsening = true,
        nlsolve = TrustRegion(),
        jac_alg = BVPJacobianAlgorithm(; bc_diffmode = AutoForwardDiff(; chunksize = 8),
            nonbc_diffmode = AutoSparseForwardDiff(; chunksize = 8)))
    alg_dense = MultipleShooting(10, AutoVern7(Rodas4P()); grid_coarsening = true,
        nlsolve = TrustRegion(),
        jac_alg = BVPJacobianAlgorithm(; bc_diffmode = AutoForwardDiff(; chunksize = 8),
            nonbc_diffmode = AutoForwardDiff(; chunksize = 8)))
    alg_default = MultipleShooting(10, AutoVern7(Rodas4P()); grid_coarsening = true)

    for (prob, alg) in Iterators.product((prob_oop, prob_iip, prob_tp_oop, prob_tp_iip),
        (alg_sp, alg_dense, alg_default))
        @info "Solver: $alg"
        @time sol = solve(prob, alg; abstol = 1e-6, reltol = 1e-6, maxiters = 1000,
            odesolve_kwargs = (; abstol = 1e-8, reltol = 1e-5))
        @test SciMLBase.successful_retcode(sol.retcode)

        if prob.problem_type isa TwoPointBVProblem
            resida, residb = zeros(5), zeros(3)
            ray_tracing_bc_a!(resida, sol.u[1], p)
            ray_tracing_bc_b!(residb, sol.u[end], p)
            @test norm(vcat(resida, residb), Inf) < 1e-6
        else
            resid = zeros(8)
            ray_tracing_bc!(resid, sol, p, sol.t)
            @test norm(resid, Inf) < 1e-6
        end
    end
end