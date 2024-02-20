using BoundaryValueDiffEq, Random, Test

function ode!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1] * u[1]
end

function bc!(residual, u, p, t)
    residual[1] = u[1][1] - 1.0
    residual[2] = u[end][1]
end

prob_func(prob, i, repeat) = remake(prob, p = [rand()])

u0 = [0.0, 1.0]
tspan = (0, pi / 2)
p = [rand()]
bvp = BVProblem(ode!, bc!, u0, tspan, p)
ensemble_prob = EnsembleProblem(bvp; prob_func)
nlsolve = NewtonRaphson()

@testset "$(solver)" for solver in (RadauIIa3, RadauIIa5, RadauIIa9, RadauIIa13) # RadauIIa1 doesn't have adaptivity
    jac_algs = [#BVPJacobianAlgorithm(),
        BVPJacobianAlgorithm(AutoSparseFiniteDiff(); bc_diffmode = AutoFiniteDiff(),
                             nonbc_diffmode = AutoSparseFiniteDiff())]
    for jac_alg in jac_algs
        sol = solve(ensemble_prob, solver(; nlsolve, jac_alg); trajectories = 10, dt = 0.1)
        @test sol.converged
    end
end

@testset "$(solver)" for solver in (LobattoIIIa2, LobattoIIIa3, LobattoIIIa4, LobattoIIIa5)
    jac_algs = [#BVPJacobianAlgorithm(),
        BVPJacobianAlgorithm(AutoSparseFiniteDiff(); bc_diffmode = AutoFiniteDiff(),
                             nonbc_diffmode = AutoSparseFiniteDiff())]
    for jac_alg in jac_algs
        sol = solve(ensemble_prob, solver(; nlsolve, jac_alg); trajectories = 10, dt = 0.1)
        @test sol.converged
    end
end

@testset "$(solver)" for solver in (LobattoIIIb3, LobattoIIIb4, LobattoIIIb5) # LobattoIIIb2 doesn't have adaptivity
    jac_algs = [#BVPJacobianAlgorithm(),
        BVPJacobianAlgorithm(AutoSparseFiniteDiff(); bc_diffmode = AutoFiniteDiff(),
                             nonbc_diffmode = AutoSparseFiniteDiff())]
    for jac_alg in jac_algs
        sol = solve(ensemble_prob, solver(; nlsolve, jac_alg); trajectories = 10, dt = 0.1)
        @test sol.converged
    end
end

@testset "$(solver)" for solver in (LobattoIIIc3, LobattoIIIc4, LobattoIIIc5) # LobattoIIIc2 doesn't have adaptivity
    jac_algs = [#BVPJacobianAlgorithm(),
        BVPJacobianAlgorithm(AutoSparseFiniteDiff(); bc_diffmode = AutoFiniteDiff(),
                             nonbc_diffmode = AutoSparseFiniteDiff())]
    for jac_alg in jac_algs
        sol = solve(ensemble_prob, solver(; nlsolve, jac_alg); trajectories = 10, dt = 0.1)
        @test sol.converged
    end
end
