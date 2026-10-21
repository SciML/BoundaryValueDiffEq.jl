using DiffEqGPU, OrdinaryDiffEqTsit5, OrdinaryDiffEqVerner

function kernel_algorithm_tests(platform)
    return @testset "DiffEqGPU algorithm selection and physical time" begin
        @test Base.get_extension(DeviceShooting, :BoundaryValueDiffEqShootingDiffEqGPUExt) !== nothing
        # Nonzero t0, nonunit intervals, and nested array parameters exercise
        # the normalized-time RHS and static parameter conversion independently.
        rhs(u, p, t) = SVector(p.rate[1] * u[1] + p.forcing[1][1] * t)
        rhs!(du, u, p, t) = (du[1] = p.rate[1] * u[1] + p.forcing[1][1] * t; nothing)
        left(u, p) = SVector(u[1] - 1)
        right(u, p) = SVector{0, eltype(u)}()
        left!(r, u, p) = (r[1] = u[1] - 1; nothing)
        right!(r, u, p) = nothing
        parameters = (rate = [0.3], forcing = ([0.7],))
        exact(t) = exp(0.3 * (t - 2)) * (1 + 0.7 * (2 / 0.3 + 1 / 0.3^2)) -
            0.7 * (t / 0.3 + 1 / 0.3^2)
        for (ode_alg, cpu_alg) in ((GPUTsit5(), Tsit5()), (GPUVern7(), Vern7())), iip in (true, false)
            prob = TwoPointBVProblem(
                iip ? rhs! : rhs, iip ? (left!, right!) : (left, right), [1.0],
                (2.0, 3.5), parameters; bcresid_prototype = (zeros(1), zeros(0)), nlls = Val(false)
            )
            alg = MultipleShooting(3, ode_alg; platform, device_steps = 5)
            (; u, cache, work, plan) = DeviceShooting.__shooting_device_setup(prob, alg)
            DeviceShooting.__shooting_residual!(work.residual, u, cache, work)
            residual = Array(work.residual)
            # Compare the one-interval map with the corresponding CPU solver.
            for i in 1:3
                t0 = 2.0 + (i - 1) / 2
                odeprob = DeviceShooting.ODEProblem(rhs, SVector(1.0), (t0, t0 + 0.5), parameters)
                reference = solve(odeprob, cpu_alg; adaptive = false, dt = 0.1)
                @test residual[i + 1] ≈ reference.u[end][1] - 1 atol = 1.0e-11
            end
            DeviceShooting.__shooting_jacobian!(plan.matrix, u, cache, plan)
            jac = Matrix(plan.matrix isa SparseMatrixCSC ? plan.matrix : SparseMatrixCSC(plan.matrix))
            expected = [1.0 0 0 0; exp(0.15) -1 0 0; 0 exp(0.15) -1 0; 0 0 exp(0.15) -1]
            @test jac ≈ expected atol = 1.0e-10
            sol = solve(prob, alg; abstol = 1.0e-10)
            @test successful_retcode(sol)
            @test Array(sol.u[end])[1] ≈ exact(3.5) atol = 1.0e-8
            # Reuse the same buffers with new states to detect stale batch data.
            u .+= 0.2
            DeviceShooting.__shooting_residual!(work.residual, u, cache, work)
            @test Array(work.residual)[2:end] ≈ residual[2:end] .+ 0.2 * (exp(0.15) - 1) atol = 1.0e-10
        end
        @testset "Implicit kernels with in-place time differentiation" begin
            for ode_alg in (GPURosenbrock23(), GPURodas4(), GPURodas5P())
                problems = map((true, false)) do iip
                    TwoPointBVProblem(
                        iip ? rhs! : rhs, iip ? (left!, right!) : (left, right), [1.0],
                        (2.0, 3.5), parameters;
                        bcresid_prototype = (zeros(1), zeros(0)), nlls = Val(false)
                    )
                end
                alg = MultipleShooting(3, ode_alg; platform, device_steps = 5)
                setups = map(prob -> DeviceShooting.__shooting_device_setup(prob, alg), problems)
                for setup in setups
                    (; u, cache, work, plan) = setup
                    DeviceShooting.__shooting_residual!(work.residual, u, cache, work)
                    DeviceShooting.__shooting_jacobian!(plan.matrix, u, cache, plan)
                end
                # Out-of-place arithmetic naturally carries the time duals;
                # the in-place scratch buffer must retain them as well.
                @test Array(setups[1].work.residual) ≈ Array(setups[2].work.residual) atol = 1.0e-11
                host_jac(setup) = setup.plan.matrix isa SparseMatrixCSC ? setup.plan.matrix : SparseMatrixCSC(setup.plan.matrix)
                @test host_jac(setups[1]) ≈ host_jac(setups[2]) atol = 1.0e-11
                batch = Array(setups[1].work.odecache.batch)[1]
                time_gradient = ForwardDiff.derivative(t -> batch.f(batch.u0, batch.p, t), 0.0)
                @test time_gradient ≈ SVector(0.7 * 0.5^2) atol = 1.0e-14
                solutions = map(prob -> solve(prob, alg; abstol = 1.0e-10), problems)
                @test all(successful_retcode, solutions)
                @test Array(solutions[1].original.u) ≈ Array(solutions[2].original.u) atol = 1.0e-10
            end
        end
        if !(platform isa CPU)
            prob = TwoPointBVProblem(
                rhs!, (left!, right!), [1.0], (2.0, 3.5), parameters;
                bcresid_prototype = (zeros(1), zeros(0)), nlls = Val(false)
            )
            @test_throws ArgumentError solve(prob, MultipleShooting(3, Tsit5(); platform, device_steps = 5))
        end
    end
end
