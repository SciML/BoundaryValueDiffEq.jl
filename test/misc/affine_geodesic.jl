using BoundaryValueDiffEq

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

function bc1!(residual, u, p, t)
    mid = div(length(u[1]), 2)
    residual[1:mid] = u[1][1:mid] - a1
    residual[(mid + 1):end] = u[end][1:mid] - a2
    return
end

function chart_log_problem!(du, u, params, t)
    M, i = params
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

@testset "successful convergence" begin
    u0 = [vcat(a1, zero(a1)), vcat(a2, zero(a1))]
    bvp1 = BVProblem(chart_log_problem!, bc1!, u0, tspan, (M, i))
    sol1 = solve(bvp1, solver, dt = dt)
    @test SciMLBase.successful_retcode(sol1.retcode)
end
