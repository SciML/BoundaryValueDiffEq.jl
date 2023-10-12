# Lambert's Problem
using BoundaryValueDiffEq, OrdinaryDiffEq, LinearAlgebra, Test

y0 = [
    -4.7763169762853989E+06,
    -3.8386398704441520E+05,
    -5.3500183933132319E+06,
    -5528.612564911408,
    1216.8442360202787,
    4845.114446429901,
]
init_val = [
    -4.7763169762853989E+06,
    -3.8386398704441520E+05,
    -5.3500183933132319E+06,
    7.0526926403748598E+06,
    -7.9650476230388973E+05,
    -1.1911128863666430E+06,
]
J2 = 1.08262668E-3
req = 6378137
myu = 398600.4418E+9
t0 = 86400 * 2.3577475462484435E+04
t1 = 86400 * 2.3577522023524125E+04
tspan = (t0, t1)

# ODE solver
function orbital!(dy, y, p, t)
    r2 = (y[1]^2 + y[2]^2 + y[3]^2)
    r3 = r2^(3 / 2)
    w = 1 + 1.5J2 * (req * req / r2) * (1 - 5y[3] * y[3] / r2)
    w2 = 1 + 1.5J2 * (req * req / r2) * (3 - 5y[3] * y[3] / r2)
    dy[1] = y[4]
    dy[2] = y[5]
    dy[3] = y[6]
    dy[4] = -myu * y[1] * w / r3
    dy[5] = -myu * y[2] * w / r3
    dy[6] = -myu * y[3] * w2 / r3
end

function bc!_generator(resid, sol, init_val)
    resid[1] = sol(t0)[1] - init_val[1]
    resid[2] = sol(t0)[2] - init_val[2]
    resid[3] = sol(t0)[3] - init_val[3]
    resid[4] = sol(t1)[1] - init_val[4]
    resid[5] = sol(t1)[2] - init_val[5]
    resid[6] = sol(t1)[3] - init_val[6]
end

function bc!_generator_2p_a(resid0, ua, init_val)
    resid0[1] = ua[1] - init_val[1]
    resid0[2] = ua[2] - init_val[2]
    resid0[3] = ua[3] - init_val[3]
end
function bc!_generator_2p_b(resid1, ub, init_val)
    resid1[1] = ub[1] - init_val[4]
    resid1[2] = ub[2] - init_val[5]
    resid1[3] = ub[3] - init_val[6]
end

cur_bc! = (resid, sol, p, t) -> bc!_generator(resid, sol, init_val)
cur_bc_2point_a! = (resid, sol, p) -> bc!_generator_2p_a(resid, sol, init_val)
cur_bc_2point_b! = (resid, sol, p) -> bc!_generator_2p_b(resid, sol, init_val)
resid_f = Array{Float64}(undef, 6)
resid_f_2p = (Array{Float64, 1}(undef, 3), Array{Float64, 1}(undef, 3))

TestTol = 0.05

### Now use the BVP solver to get closer
bvp = BVProblem(orbital!, cur_bc!, y0, tspan)
for autodiff in (AutoForwardDiff(), AutoFiniteDiff(; fdtype = Val(:central)),
    AutoSparseForwardDiff(), AutoFiniteDiff(; fdtype = Val(:forward)),
    AutoSparseFiniteDiff())
    nlsolve = NewtonRaphson(; autodiff)
    @time sol = solve(bvp, Shooting(DP5(); nlsolve); force_dtmin = true,
        abstol = 1e-13, reltol = 1e-13)
    cur_bc!(resid_f, sol, nothing, sol.t)
    @test norm(resid_f, Inf) < TestTol

    jac_alg = BVPJacobianAlgorithm(; nonbc_diffmode = autodiff)
    @time sol = solve(bvp, MultipleShooting(10, DP5(); nlsolve, jac_alg); abstol = 1e-6,
        reltol = 1e-6)
    @test SciMLBase.successful_retcode(sol)
    cur_bc!(resid_f, sol, nothing, sol.t)
    @test norm(resid_f, Inf) < 1e-6
end

### Using the TwoPoint BVP Structure
bvp = TwoPointBVProblem(orbital!, (cur_bc_2point_a!, cur_bc_2point_b!), y0, tspan;
    bcresid_prototype = (Array{Float64}(undef, 3), Array{Float64}(undef, 3)))
for autodiff in (AutoForwardDiff(), AutoFiniteDiff(; fdtype = Val(:central)),
    AutoSparseForwardDiff(), AutoFiniteDiff(; fdtype = Val(:forward)),
    AutoSparseFiniteDiff())
    nlsolve = NewtonRaphson(; autodiff)
    @time sol = solve(bvp, Shooting(DP5(); nlsolve); force_dtmin = true, abstol = 1e-13,
        reltol = 1e-13)
    cur_bc_2point_a!(resid_f_2p[1], sol(t0), nothing)
    cur_bc_2point_b!(resid_f_2p[2], sol(t1), nothing)
    @test norm(reduce(vcat, resid_f_2p), Inf) < TestTol

    jac_alg = BVPJacobianAlgorithm(; nonbc_diffmode = autodiff)
    @time sol = solve(bvp, MultipleShooting(10, DP5(); nlsolve, jac_alg); abstol = 1e-6,
        reltol = 1e-6)
    @test SciMLBase.successful_retcode(sol)
    cur_bc_2point_a!(resid_f_2p[1], sol(t0), nothing)
    cur_bc_2point_b!(resid_f_2p[2], sol(t1), nothing)
    @test norm(reduce(vcat, resid_f_2p), Inf) < TestTol
end
