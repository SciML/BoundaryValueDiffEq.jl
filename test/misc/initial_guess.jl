using BoundaryValueDiffEq, OrdinaryDiffEq, Test, LinearAlgebra

@testset "Initial Guess" begin
    # Problem taken from https://github.com/SciML/BoundaryValueDiffEq.jl/issues/117#issuecomment-1780981510
    function affine_connection(a, Xc, Yc)
        MR = 3.0
        Mr = 2.0
        Zc = similar(Xc)
        θ = a[1]
        sinθ, cosθ = sincos(θ)
        Γ¹₂₂ = (MR + Mr * cosθ) * sinθ / Mr
        Γ²₁₂ = -Mr * sinθ / (MR + Mr * cosθ)

        Zc[1] = Xc[2] * Γ¹₂₂ * Yc[2]
        Zc[2] = Γ²₁₂ * (Xc[1] * Yc[2] + Xc[2] * Yc[1])
        return Zc
    end

    function chart_log_problem!(du, u, p, t)
        mid = div(length(u), 2)
        a = u[1:mid]
        dx = u[(mid + 1):end]
        ddx = -affine_connection(a, dx, dx)
        du[1:mid] .= dx
        du[(mid + 1):end] .= ddx
        return du
    end

    function bc1!(residual, u, p, t)
        a1, a2 = p[1:2], p[3:4]
        mid = div(length(u[1]), 2)
        residual[1:mid] = u[1][1:mid] - a1
        residual[(mid + 1):end] = u[end][1:mid] - a2
        return residual
    end

    function initial_guess_1(p, t)
        a1, a2 = p[1:2], p[3:4]
        return vcat(t * a1 + (1 - t) * a2, zero(a1))
    end

    function initial_guess_2(t)
        a1, a2 = [0.5, -1.2], [-0.5, 0.3]
        return vcat(t * a1 + (1 - t) * a2, zero(a1))
    end

    dt = 0.05
    p = [0.5, -1.2, -0.5, 0.3]
    tspan = (0.0, 1.0)

    bvp1 = BVProblem(chart_log_problem!, bc1!, initial_guess_1, tspan, p)

    algs = [Shooting(Tsit5()), MultipleShooting(10, Tsit5()), MIRK4(), MIRK5(), MIRK6()]

    for alg in algs
        if alg isa Shooting || alg isa MultipleShooting
            sol = solve(bvp1, alg)
        else
            sol = solve(bvp1, alg; dt)
        end
        @test SciMLBase.successful_retcode(sol)
        resid = zeros(4)
        bc1!(resid, sol, p, sol.t)
        @test norm(resid) < 1e-10
    end

    bvp2 = BVProblem(chart_log_problem!, bc1!, initial_guess_2, tspan, p)

    for alg in algs
        if alg isa Shooting || alg isa MultipleShooting
            sol = solve(bvp2, alg)
            @test_deprecated solve(bvp2, alg)
        else
            sol = solve(bvp2, alg; dt)
            @test_deprecated solve(bvp2, alg; dt)
        end
        @test SciMLBase.successful_retcode(sol)
        resid = zeros(4)
        bc1!(resid, sol, p, sol.t)
        @test norm(resid) < 1e-10
    end
end
