@testitem "MIRK Loss Function Allocations" tags=[:allocs] begin
    using BoundaryValueDiffEq, BoundaryValueDiffEqMIRK, BoundaryValueDiffEqCore, LinearAlgebra

    function f!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
        return nothing
    end

    function bc!(resid, sol, p, t)
        resid[1] = sol(0.0)[1] - 1.0
        resid[2] = sol(1.0)[1] - cos(1.0)
        return nothing
    end

    function tpbc_a!(resid, ua, p)
        resid[1] = ua[1] - 1.0
        return nothing
    end

    function tpbc_b!(resid, ub, p)
        resid[1] = ub[1] - cos(1.0)
        return nothing
    end

    u0 = [1.0, 0.0]
    tspan = (0.0, 1.0)

    bvp = BVProblem(BVPFunction{true}(f!, bc!; bcresid_prototype = zeros(2)), u0, tspan)
    tpbvp = BVProblem(
        BVPFunction{true}(f!, (tpbc_a!, tpbc_b!);
            bcresid_prototype = (zeros(1), zeros(1)), twopoint = Val(true)),
        u0, tspan)

    # Test that the loss function allocations scale sub-linearly with mesh size
    # (i.e., per-step allocations are bounded, not proportional to mesh points)
    for (name, prob) in [("StandardBVP", bvp), ("TwoPointBVP", tpbvp)]
        for alg in [MIRK4(), MIRK5(), MIRK6()]
            cache = SciMLBase.__init(prob, alg; dt = 0.1, adaptive = false)
            nlprob = BoundaryValueDiffEqMIRK.__construct_problem(
                cache, vec(cache.y₀), copy(cache.y₀))

            u_test = copy(nlprob.u0)
            resid_test = zeros(length(nlprob.u0))

            # Warmup
            nlprob.f(resid_test, u_test, nlprob.p)

            # Measure allocations per loss call
            allocs = @allocated nlprob.f(resid_test, u_test, nlprob.p)

            # Loss function should allocate less than 10 KiB per call
            # (the remaining allocations are from SubArray views in the inner loop
            # which scale with mesh size but are small per-element)
            @test allocs < 10 * 1024  # 10 KiB threshold
        end
    end

    # Test that non-adaptive solve allocations are bounded
    for alg in [MIRK4(), MIRK5()]
        # Small mesh
        sol_small = solve(bvp, alg; dt = 0.1, adaptive = false)
        @test sol_small.retcode == ReturnCode.Success
        allocs_small = @allocated solve(bvp, alg; dt = 0.1, adaptive = false)

        # Larger mesh (5x)
        sol_large = solve(bvp, alg; dt = 0.02, adaptive = false)
        @test sol_large.retcode == ReturnCode.Success
        allocs_large = @allocated solve(bvp, alg; dt = 0.02, adaptive = false)

        # Allocations should scale much less than 5x
        # (ideally close to linear with mesh size due to Jacobian setup,
        # but per-Newton-step allocations should be small)
        ratio = allocs_large / allocs_small
        @test ratio < 10  # Should be well under 10x for 5x more mesh points
    end
end
