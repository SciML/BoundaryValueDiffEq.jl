using BoundaryValueDiffEq, OrdinaryDiffEq, LinearAlgebra, Test

@testset "Lambert's Problem" begin
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

    @info "Solving Lambert's Problem in Multi Point BVP Form"

    bvp = BVProblem(orbital!, cur_bc!, y0, tspan; nlls = Val(false))
    for autodiff in (AutoForwardDiff(; chunksize = 6),
        AutoFiniteDiff(; fdtype = Val(:central)), AutoSparseForwardDiff(; chunksize = 6),
        AutoFiniteDiff(; fdtype = Val(:forward)), AutoSparseFiniteDiff())
        nlsolve = TrustRegion(; autodiff)

        @info "Single Shooting Lambert's Problem: $(autodiff)"
        @time sol = solve(bvp, Shooting(DP5(); nlsolve); force_dtmin = true, abstol = 1e-6,
            reltol = 1e-6, verbose = false,
            odesolve_kwargs = (abstol = 1e-6, reltol = 1e-3))

        @info "Single Shooting Lambert's Problem: $(norm(sol.resid, Inf))"
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, Inf) < 1e-6

        @info "Multiple Shooting Lambert's Problem: $(autodiff)"
        jac_alg = BVPJacobianAlgorithm(; nonbc_diffmode = autodiff,
            bc_diffmode = BoundaryValueDiffEq.__get_non_sparse_ad(autodiff))
        @time sol = solve(bvp, MultipleShooting(10, DP5(); nlsolve, jac_alg);
            force_dtmin = true, abstol = 1e-6, reltol = 1e-6, verbose = false,
            odesolve_kwargs = (abstol = 1e-6, reltol = 1e-3))

        @info "Multiple Shooting Lambert's Problem: $(norm(sol.resid, Inf))"
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, Inf) < 1e-6
    end

    @info "Solving Lambert's Problem in Two Point BVP Form"

    bvp = TwoPointBVProblem(orbital!, (cur_bc_2point_a!, cur_bc_2point_b!), y0, tspan;
        bcresid_prototype = (Array{Float64}(undef, 3), Array{Float64}(undef, 3)),
        nlls = Val(false))
    for autodiff in (AutoForwardDiff(; chunksize = 6), AutoSparseFiniteDiff(),
        AutoFiniteDiff(; fdtype = Val(:central)), AutoFiniteDiff(; fdtype = Val(:forward)),
        AutoSparseForwardDiff(; chunksize = 6))
        nlsolve = TrustRegion(; autodiff)

        @info "Single Shooting Lambert's Problem: $(autodiff)"
        @time sol = solve(bvp, Shooting(DP5(); nlsolve); force_dtmin = true, abstol = 1e-6,
            reltol = 1e-6, verbose = false,
            odesolve_kwargs = (abstol = 1e-6, reltol = 1e-3))

        @info "Single Shooting Lambert's Problem: $(norm(sol.resid, Inf))"
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, Inf) < 1e-6

        @info "Multiple Shooting Lambert's Problem: $(autodiff)"
        jac_alg = BVPJacobianAlgorithm(; nonbc_diffmode = autodiff, bc_diffmode = autodiff)
        @time sol = solve(bvp, MultipleShooting(10, DP5(); nlsolve, jac_alg);
            force_dtmin = true, abstol = 1e-6, reltol = 1e-6, verbose = false,
            odesolve_kwargs = (abstol = 1e-6, reltol = 1e-3))

        @info "Multiple Shooting Lambert's Problem: $(norm(sol.resid, Inf))"
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, Inf) < 1e-6
    end
end
