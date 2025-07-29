@testitem "EnsembleProblem" begin
    using BoundaryValueDiffEqFIRK, Random

    function ode!(du, u, p, t)
        du[1] = u[2]
        du[2] = -p[1] * u[1]
    end

    function bc!(residual, u, p, t)
        residual[1] = u(0.0)[1] - 1.0
        residual[2] = u(1.0)[1]
    end

    prob_func(prob, i, repeat) = remake(prob, p = [rand()])

    u0 = [0.0, 1.0]
    tspan = (0, pi / 2)
    p = [rand()]
    bvp = BVProblem(ode!, bc!, u0, tspan, p)
    ensemble_prob = EnsembleProblem(bvp; prob_func)
    nlsolve = NewtonRaphson()
    nested = false

    @testset "$(solver)" for solver in (RadauIIa2, RadauIIa3, RadauIIa5, RadauIIa7) # RadauIIa1 doesn't have adaptivity
        jac_algs = [BVPJacobianAlgorithm(),
            BVPJacobianAlgorithm(
                AutoSparse(AutoFiniteDiff()); bc_diffmode = AutoFiniteDiff(),
                nonbc_diffmode = AutoSparse(AutoFiniteDiff()))]
        for jac_alg in jac_algs
            sol = solve(ensemble_prob, solver(nlsolve, jac_alg; nested);
                trajectories = 10, dt = 0.1)
            @test sol.converged
        end
    end

    @testset "$(solver)" for solver in (
        LobattoIIIa2, LobattoIIIa3, LobattoIIIa4, LobattoIIIa5)
        jac_algs = [BVPJacobianAlgorithm(),
            BVPJacobianAlgorithm(
                AutoSparse(AutoFiniteDiff()); bc_diffmode = AutoFiniteDiff(),
                nonbc_diffmode = AutoSparse(AutoFiniteDiff()))]
        for jac_alg in jac_algs
            sol = solve(ensemble_prob, solver(nlsolve, jac_alg; nested);
                trajectories = 10, dt = 0.1)
            @test sol.converged
        end
    end

    @testset "$(solver)" for solver in (LobattoIIIb3, LobattoIIIb4, LobattoIIIb5) # LobattoIIIb2 doesn't have adaptivity
        jac_algs = [BVPJacobianAlgorithm(),
            BVPJacobianAlgorithm(
                AutoSparse(AutoFiniteDiff()); bc_diffmode = AutoFiniteDiff(),
                nonbc_diffmode = AutoSparse(AutoFiniteDiff()))]
        for jac_alg in jac_algs
            sol = solve(ensemble_prob, solver(nlsolve, jac_alg; nested);
                trajectories = 10, dt = 0.1)
            @test sol.converged
        end
    end

    @testset "$(solver)" for solver in (LobattoIIIc3, LobattoIIIc4, LobattoIIIc5) # LobattoIIIc2 doesn't have adaptivity
        jac_algs = [BVPJacobianAlgorithm(),
            BVPJacobianAlgorithm(
                AutoSparse(AutoFiniteDiff()); bc_diffmode = AutoFiniteDiff(),
                nonbc_diffmode = AutoSparse(AutoFiniteDiff()))]
        for jac_alg in jac_algs
            sol = solve(ensemble_prob, solver(nlsolve, jac_alg; nested);
                trajectories = 10, dt = 0.1)
            @test sol.converged
        end
    end
end
