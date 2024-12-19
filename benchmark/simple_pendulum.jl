module SimplePendulumBenchmark

using BoundaryValueDiffEq, OrdinaryDiffEq, NonlinearSolveFirstOrder

const tspan = (0.0, π / 2)

function simple_pendulum!(du, u, p, t)
    g, L, θ, dθ = 9.81, 1.0, u[1], u[2]
    du[1] = dθ
    du[2] = -(g / L) * sin(θ)
    return nothing
end

function bc_pendulum!(residual, u, p, t)
    t0, t1 = tspan
    residual[1] = u((t0 + t1) / 2)[1] + π / 2
    residual[2] = u(t1)[1] - π / 2
    return nothing
end

function simple_pendulum(u, p, t)
    g, L, θ, dθ = 9.81, 1.0, u[1], u[2]
    return [dθ, -(g / L) * sin(θ)]
end

function bc_pendulum(u, p, t)
    t0, t1 = tspan
    return [u((t0 + t1) / 2)[1] + π / 2, u(t1)[1] - π / 2]
end

const prob_oop = BVProblem{false}(simple_pendulum, bc_pendulum, [π / 2, π / 2], tspan)
const prob_iip = BVProblem{true}(simple_pendulum!, bc_pendulum!, [π / 2, π / 2], tspan)

end

function create_simple_pendulum_benchmark()
    suite = BenchmarkGroup()

    iip_suite = BenchmarkGroup()
    oop_suite = BenchmarkGroup()

    suite["IIP"] = iip_suite
    suite["OOP"] = oop_suite

    if @isdefined(MultipleShooting)
        iip_suite["MultipleShooting(100, Tsit5; grid_coarsening = true)"] = @benchmarkable solve(
            $SimplePendulumBenchmark.prob_iip,
            $MultipleShooting(100, Tsit5(), NewtonRaphson()))
        iip_suite["MultipleShooting(100, Tsit5; grid_coarsening = false)"] = @benchmarkable solve(
            $SimplePendulumBenchmark.prob_iip,
            $MultipleShooting(100, Tsit5(), NewtonRaphson(); grid_coarsening = false))
        iip_suite["MultipleShooting(10, Tsit5; grid_coarsening = true)"] = @benchmarkable solve(
            $SimplePendulumBenchmark.prob_iip,
            $MultipleShooting(10, Tsit5(), NewtonRaphson()))
        iip_suite["MultipleShooting(10, Tsit5; grid_coarsening = false)"] = @benchmarkable solve(
            $SimplePendulumBenchmark.prob_iip,
            $MultipleShooting(10, Tsit5(), NewtonRaphson(); grid_coarsening = false))
    end
    if @isdefined(Shooting)
        iip_suite["Shooting(Tsit5())"] = @benchmarkable solve(
            $SimplePendulumBenchmark.prob_iip, $Shooting(Tsit5(), NewtonRaphson()))
    end
    for alg in (MIRK2, MIRK3, MIRK4, MIRK5, MIRK6)
        if @isdefined(alg)
            iip_suite["$alg()"] = @benchmarkable solve(
                $SimplePendulumBenchmark.prob_iip, $alg(), dt = 0.05)
        end
    end

    if @isdefined(MultipleShooting)
        oop_suite["MultipleShooting(100, Tsit5; grid_coarsening = true)"] = @benchmarkable solve(
            $SimplePendulumBenchmark.prob_oop,
            $MultipleShooting(100, Tsit5(), NewtonRaphson()))
        oop_suite["MultipleShooting(100, Tsit5; grid_coarsening = false)"] = @benchmarkable solve(
            $SimplePendulumBenchmark.prob_oop,
            $MultipleShooting(100, Tsit5(), NewtonRaphson(); grid_coarsening = false))
        oop_suite["MultipleShooting(10, Tsit5; grid_coarsening = true)"] = @benchmarkable solve(
            $SimplePendulumBenchmark.prob_oop,
            $MultipleShooting(10, Tsit5(), NewtonRaphson()))
        oop_suite["MultipleShooting(10, Tsit5; grid_coarsening = false)"] = @benchmarkable solve(
            $SimplePendulumBenchmark.prob_oop,
            $MultipleShooting(10, Tsit5(), NewtonRaphson(); grid_coarsening = false))
    end
    if @isdefined(Shooting)
        oop_suite["Shooting(Tsit5())"] = @benchmarkable solve(
            $SimplePendulumBenchmark.prob_oop, $Shooting(Tsit5(), NewtonRaphson()))
    end
    for alg in (MIRK2, MIRK3, MIRK4, MIRK5, MIRK6)
        if @isdefined(alg)
            oop_suite["$alg()"] = @benchmarkable solve(
                $SimplePendulumBenchmark.prob_oop, $alg(), dt = 0.05)
        end
    end

    return suite
end
