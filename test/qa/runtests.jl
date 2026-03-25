using BoundaryValueDiffEq, BoundaryValueDiffEqMIRK, BoundaryValueDiffEqCore,
    Aqua, JET, Test, SciMLBase

@testset "Quality Assurance" begin
    @testset "Aqua" begin
        Aqua.test_all(
            BoundaryValueDiffEq; ambiguities = false,
            piracies = (broken = false, treat_as_own = [SciMLBase.BVProblem])
        )
    end

    @testset "JET" begin
        rep = JET.report_package(
            BoundaryValueDiffEq;
            target_modules = (BoundaryValueDiffEq,)
        )
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "Zero per-step allocations in MIRK loss function" begin
        function _f!(du, u, p, t)
            du[1] = u[2]
            du[2] = -u[1]
            return nothing
        end
        function _bc!(resid, sol, p, t)
            resid[1] = sol(0.0)[1] - 1.0
            resid[2] = sol(1.0)[1] - cos(1.0)
            return nothing
        end
        u0 = [1.0, 0.0]
        tspan = (0.0, 1.0)
        bvp = BVProblem(
            BVPFunction{true}(_f!, _bc!; bcresid_prototype = zeros(2)), u0, tspan)

        cache = SciMLBase.__init(bvp, MIRK4(); dt = 0.1, adaptive = false)
        nlprob = BoundaryValueDiffEqMIRK.__construct_problem(
            cache, vec(cache.y₀), copy(cache.y₀))

        u_test = copy(nlprob.u0)
        resid_test = zeros(length(nlprob.u0))
        p_test = nlprob.p

        # Verify loss function is allocation-free at runtime
        function _bench_loss(f, resid, u, p, N)
            for _ in 1:N
                f(resid, u, p)
            end
        end
        _bench_loss(nlprob.f, resid_test, u_test, p_test, 10)  # warmup
        stats = @timed _bench_loss(nlprob.f, resid_test, u_test, p_test, 10_000)
        bytes_per_call = stats.bytes / 10_000
        @test bytes_per_call == 0.0
    end
end
