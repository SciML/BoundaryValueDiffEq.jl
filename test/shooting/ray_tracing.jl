using BoundaryValueDiffEq, LinearAlgebra, OrdinaryDiffEq, Test

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

prob_oop = BVProblem{false}(ray_tracing, ray_tracing_bc, u0, tspan, p)
prob_iip = BVProblem{true}(ray_tracing!, ray_tracing_bc!, u0, tspan, p)
prob_tp_oop = TwoPointBVProblem{false}(ray_tracing, (ray_tracing_bc_a, ray_tracing_bc_b),
    u0, tspan, p)
prob_tp_iip = TwoPointBVProblem{true}(ray_tracing!, (ray_tracing_bc_a!, ray_tracing_bc_b!),
    u0, tspan, p; bcresid_prototype = (zeros(5), zeros(3)))

alg_sp = MultipleShooting(10, AutoVern7(Rodas4P()); grid_coarsening = true,
    jac_alg = BVPJacobianAlgorithm(; bc_diffmode = AutoForwardDiff(),
        nonbc_diffmode = AutoSparseForwardDiff()))
alg_dense = MultipleShooting(10, AutoVern7(Rodas4P()); grid_coarsening = true,
    jac_alg = BVPJacobianAlgorithm(; bc_diffmode = AutoForwardDiff(),
        nonbc_diffmode = AutoForwardDiff()))
alg_default = MultipleShooting(10, AutoVern7(Rodas4P()); grid_coarsening = true)

for (prob, alg) in Iterators.product((prob_oop, prob_iip, prob_tp_oop, prob_tp_iip),
    (alg_sp, alg_dense, alg_default))
    @time sol = solve(prob, alg; abstol = 1e-9, reltol = 1e-9, maxiters = 1000)
    @test SciMLBase.successful_retcode(sol.retcode)

    if prob.problem_type isa TwoPointBVProblem
        resida, residb = zeros(5), zeros(3)
        ray_tracing_bc_a!(resida, sol.u[1], p)
        ray_tracing_bc_b!(residb, sol.u[end], p)
        @test norm(vcat(resida, residb), 2) < 5e-5
    else
        resid = zeros(8)
        ray_tracing_bc!(resid, sol, p, sol.t)
        @test norm(resid, 2) < 5e-5
    end
end
