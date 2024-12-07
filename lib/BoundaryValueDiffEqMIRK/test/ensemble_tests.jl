
@testitem "EnsembleProblem" begin
    using BoundaryValueDiffEqMIRK
    using Random

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

    @testset "$(solver)" for solver in (MIRK2, MIRK3, MIRK4, MIRK5, MIRK6)
        jac_algs = [BVPJacobianAlgorithm(),
            BVPJacobianAlgorithm(; bc_diffmode = AutoFiniteDiff(),
                nonbc_diffmode = AutoSparse(AutoFiniteDiff()))]
        for jac_alg in jac_algs
            sol = solve(ensemble_prob, solver(; jac_alg); trajectories = 10, dt = 0.1)
            @test sol.converged
        end
    end
end
