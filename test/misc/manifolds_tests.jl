@testitem "Manifolds.jl Integration" begin
    using LinearAlgebra, OrdinaryDiffEq

    struct EmbeddedTorus
        R::Float64
        r::Float64
    end

    function affine_connection!(M::EmbeddedTorus, Zc, i, a, Xc, Yc)
        θ = a[1] .+ i[1]
        sinθ, cosθ = sincos(θ)
        Γ¹₂₂ = (M.R + M.r * cosθ) * sinθ / M.r
        Γ²₁₂ = -M.r * sinθ / (M.R + M.r * cosθ)

        Zc[1] = Xc[2] * Γ¹₂₂ * Yc[2]
        Zc[2] = Γ²₁₂ * (Xc[1] * Yc[2] + Xc[2] * Yc[1])
        return Zc
    end

    M = EmbeddedTorus(3, 2)
    a1 = [0.5, -1.2]
    a2 = [-0.5, 0.3]
    i = (0, 0)
    solver = MIRK4()
    dt = 0.05
    tspan = (0.0, 1.0)

    function bc1!(residual, u, params, t)
        M, i, a1, a2 = params
        mid = div(length(u(0.0)), 2)
        residual[1:mid] = u(0.0)[1:mid] - a1
        residual[(mid + 1):end] = u(1.0)[1:mid] - a2
        return
    end

    function chart_log_problem!(du, u, params, t)
        M, i, a1, a2 = params
        mid = div(length(u), 2)
        a = u[1:mid]
        dx = u[(mid + 1):end]
        ddx = similar(dx)
        affine_connection!(M, ddx, i, a, dx, dx)
        ddx .*= -1
        du[1:mid] .= dx
        du[(mid + 1):end] .= ddx
        return du
    end

    @testset "Successful Convergence" begin
        u0 = [vcat(a1, zero(a1)), vcat(a2, zero(a1))]
        bvp1 = BVProblem(chart_log_problem!, bc1!, u0, tspan, (M, i, a1, a2))
        sol1 = solve(bvp1, solver, dt = dt)
        @test SciMLBase.successful_retcode(sol1.retcode)
    end

    function initial_guess_1(p, t)
        _, _, a1, a2 = p
        return vcat(t * a1 + (1 - t) * a2, zero(a1))
    end

    algs = [Shooting(Tsit5()), MultipleShooting(10, Tsit5()), MIRK4(), MIRK5(), MIRK6()]

    @testset "Initial Guess Functions" begin
        bvp = BVProblem(chart_log_problem!, bc1!, initial_guess_1, tspan, (M, i, a1, a2))

        for alg in algs
            if alg isa Shooting || alg isa MultipleShooting
                sol = solve(bvp, alg)
            else
                sol = solve(bvp, alg; dt, abstol = 1e-8)
            end
            @test SciMLBase.successful_retcode(sol)
            resid = zeros(4)
            bc1!(resid, sol, (M, i, a1, a2), sol.t)
            @test norm(resid, Inf) < 1e-10
        end
    end
end
