module BoundaryValueDiffEqOrdinaryDiffEqExt

# This extension doesn't add any new feature atm but is used to precompile some common
# shooting workflows

import Preferences: @load_preference
import PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@recompile_invalidations begin
    using BoundaryValueDiffEq, OrdinaryDiffEq
end

@setup_workload begin
    function f1!(du, u, p, t)
        du[1] = u[2]
        du[2] = 0
    end
    f1(u, p, t) = [u[2], 0]

    function bc1!(residual, u, p, t)
        residual[1] = u(0.0)[1] - 5
        residual[2] = u(5.0)[1]
    end
    bc1(u, p, t) = [u(0.0)[1] - 5, u(5.0)[1]]

    bc1_a!(residual, ua, p) = (residual[1] = ua[1] - 5)
    bc1_b!(residual, ub, p) = (residual[1] = ub[1])

    bc1_a(ua, p) = [ua[1] - 5]
    bc1_b(ub, p) = [ub[1]]

    tspan = (0.0, 5.0)
    u0 = [5.0, -3.5]
    bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

    probs = [
        BVProblem(f1!, bc1!, u0, tspan),
        BVProblem(f1, bc1, u0, tspan),
        TwoPointBVProblem(f1!, (bc1_a!, bc1_b!), u0, tspan; bcresid_prototype),
        TwoPointBVProblem(f1, (bc1_a, bc1_b), u0, tspan; bcresid_prototype),
    ]

    algs = []

    if @load_preference("PrecompileShooting", true)
        push!(algs,
            Shooting(Tsit5();
                nlsolve = NewtonRaphson(; autodiff = AutoForwardDiff(chunksize = 2))))
    end

    if @load_preference("PrecompileMultipleShooting", true)
        push!(algs,
            MultipleShooting(10,
                Tsit5();
                nlsolve = NewtonRaphson(; autodiff = AutoForwardDiff(chunksize = 2)),
                jac_alg = BVPJacobianAlgorithm(;
                    bc_diffmode = AutoForwardDiff(; chunksize = 2),
                    nonbc_diffmode = AutoSparseForwardDiff(; chunksize = 2))))
    end

    @compile_workload begin
        for prob in probs, alg in algs
            solve(prob, alg)
        end
    end

    function f1_nlls!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    f1_nlls(u, p, t) = [u[2], -u[1]]

    function bc1_nlls!(resid, sol, p, t)
        solₜ₁ = sol(0.0)
        solₜ₂ = sol(100.0)
        resid[1] = solₜ₁[1]
        resid[2] = solₜ₂[1] - 1
        resid[3] = solₜ₂[2] + 1.729109
        return nothing
    end
    bc1_nlls(sol, p, t) = [sol(0.0)[1], sol(100.0)[1] - 1, sol(100.0)[2] + 1.729109]

    bc1_nlls_a!(resid, ua, p) = (resid[1] = ua[1])
    bc1_nlls_b!(resid, ub, p) = (resid[1] = ub[1] - 1; resid[2] = ub[2] + 1.729109)

    bc1_nlls_a(ua, p) = [ua[1]]
    bc1_nlls_b(ub, p) = [ub[1] - 1, ub[2] + 1.729109]

    tspan = (0.0, 100.0)
    u0 = [0.0, 1.0]
    bcresid_prototype1 = Array{Float64}(undef, 3)
    bcresid_prototype2 = (Array{Float64}(undef, 1), Array{Float64}(undef, 2))

    probs = [
        BVProblem(BVPFunction(f1_nlls!, bc1_nlls!; bcresid_prototype = bcresid_prototype1),
            u0, tspan),
        BVProblem(BVPFunction(f1_nlls, bc1_nlls; bcresid_prototype = bcresid_prototype1),
            u0, tspan),
        TwoPointBVProblem(f1_nlls!, (bc1_nlls_a!, bc1_nlls_b!), u0, tspan;
            bcresid_prototype = bcresid_prototype2),
        TwoPointBVProblem(f1_nlls, (bc1_nlls_a, bc1_nlls_b), u0, tspan;
            bcresid_prototype = bcresid_prototype2),
    ]

    algs = []

    if @load_preference("PrecompileShootingNLLS", VERSION≥v"1.10-")
        append!(algs,
            [
                Shooting(Tsit5();
                    nlsolve = LevenbergMarquardt(;
                        autodiff = AutoForwardDiff(chunksize = 2))),
                Shooting(Tsit5();
                    nlsolve = GaussNewton(; autodiff = AutoForwardDiff(chunksize = 2))),
            ])
    end

    if @load_preference("PrecompileMultipleShootingNLLS", VERSION≥v"1.10-")
        append!(algs,
            [
                MultipleShooting(10, Tsit5();
                    nlsolve = LevenbergMarquardt(;
                        autodiff = AutoForwardDiff(chunksize = 2)),
                    jac_alg = BVPJacobianAlgorithm(;
                        bc_diffmode = AutoForwardDiff(; chunksize = 2),
                        nonbc_diffmode = AutoSparseForwardDiff(; chunksize = 2))),
                MultipleShooting(10, Tsit5();
                    nlsolve = GaussNewton(; autodiff = AutoForwardDiff(chunksize = 2)),
                    jac_alg = BVPJacobianAlgorithm(;
                        bc_diffmode = AutoForwardDiff(; chunksize = 2),
                        nonbc_diffmode = AutoSparseForwardDiff(; chunksize = 2))),
            ])
    end

    @compile_workload begin
        for prob in probs, alg in algs
            solve(prob, alg)
        end
    end
end

end
