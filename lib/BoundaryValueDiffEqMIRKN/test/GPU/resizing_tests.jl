using BoundaryValueDiffEqMIRKN, Test

function test_mirkn_flat_buffers(upload, platform)
    MN = BoundaryValueDiffEqMIRKN
    f!(a, v, u, p, t) = (a[1] = u[1]; nothing)
    bc!(r, v, u, p, t) = (r[1] = u(first(t))[1] - 1; r[2] = v(last(t))[1] - exp(1); nothing)
    @testset "Flat MIRKN $Alg sparse=$sparse" for Alg in (MIRKN4, MIRKN6), sparse in (false, true)
        prob = SecondOrderBVProblem(f!, bc!, upload([0.5]), (0.0, 1.0))
        cache = MN.__init_mirkn_device(
            prob, Alg(; nlsolve = NewtonRaphson()), prob.u0;
            dt = 0.125, abstol = 1.0e-10, adaptive = false, controller = NoErrorControl(),
            nlsolve_kwargs = (; abstol = 1.0e-10), optimize_kwargs = (;), verbose = false
        )
        if !sparse
            fields = map(fieldnames(typeof(cache))) do name
                name === :jacobian_cache && return nothing
                name === :jac_prototype && return similar(cache.y, length(cache.residual) * length(cache.y))
                getfield(cache, name)
            end
            cache = MN.MIRKNCache{true, Float64}(fields...)
        end
        @test cache isa MN.MIRKNCache
        names = (:y, :mesh, :mesh_dt, :residual, :k_discrete, :collocation_cache)
        owners = map(name -> getproperty(cache, name), names)
        dense = cache.jac_prototype
        saved = solve!(cache)
        @test successful_retcode(saved)
        @test !ismutabletype(typeof(cache))
        @test all(x -> x isa AbstractVector, owners)
        @test all(x -> typeof(BoundaryValueDiffEqMIRKN.KernelAbstractions.get_backend(x)) === typeof(platform), owners)
        for nodes in (17, 17, 5, 33)
            times = [0.13, 0.43, 0.87]
            oldvalues = map(t -> Array(saved(t).x[1]), times)
            oldderivs = map(t -> Array(saved(t, Val{1}).x[1]), times)
            MN.__device_jacobian!(MN.__mirkn_jacobian(cache), cache.y, cache)
            work = copy(cache.work_buffers)
            @test !isempty(cache.device_cache)
            target = collect(range(0.0, 1.0; length = nodes)) .^ 1.1
            resize!(cache.host_mesh, nodes)
            copyto!(cache.host_mesh, target)
            MN.__mirkn_resize_buffers!(cache)
            @test isempty(cache.device_cache)
            # MIRKN is fixed-mesh: a resized problem needs a fresh initial guess,
            # rather than pretending to support an adaptive error controller.
            fill!(cache.y, 0.5)
            @test all(getproperty(cache, name) === owner for (name, owner) in zip(names, owners))
            @test cache.jac_prototype === dense
            @test Array(cache.mesh) == target
            @test Array(cache.mesh_dt) ≈ diff(target)
            @test size(MN.__mirkn_stages(cache)) == (cache.M, cache.TU.s, nodes - 1)
            @test size(MN.__mirkn_collocation(cache)) == (2cache.M, nodes - 1)
            next = solve!(cache)
            @test successful_retcode(next)
            @test all(cache.work_buffers[key] === owner for (key, owner) in work)
            @test size(MN.__mirkn_jacobian(cache)) == (length(cache.residual), length(cache.y))
            @test map(t -> Array(saved(t).x[1]), times) == oldvalues
            @test map(t -> Array(saved(t, Val{1}).x[1]), times) == oldderivs
            @test Array(next.u[end].x[1]) ≈ [exp(1)] atol = 3.0e-5
            saved = next
        end
    end
    return test_mirkn_buffer_features(upload, platform)
end

function test_mirkn_buffer_features(upload, platform)
    MN = BoundaryValueDiffEqMIRKN
    f!(a, v, u, p, t) = (a[1] = u[1]; nothing)
    bc!(r, v, u, p, t) = (r[1] = u(first(t))[1] - 1; r[2] = v(last(t))[1] - exp(1); nothing)
    return @testset "MIRKN boundary sparsity on repeated solve" begin
        timed_bc!(r, v, u, p, t) = (r[1] = u(t[1])[1] - 1; r[2] = v(p[1])[1] - exp(p[1]); nothing)
        prob = SecondOrderBVProblem(f!, timed_bc!, upload([0.8]), (0.0, 1.0), upload([0.5]))
        cache = MN.__init_mirkn_device(
            prob, MIRKN6(; nlsolve = NewtonRaphson()), prob.u0;
            dt = 0.1, abstol = 1.0e-10, adaptive = false, controller = NoErrorControl(),
            nlsolve_kwargs = (; abstol = 1.0e-10), optimize_kwargs = (;), verbose = false
        )
        first_sol = solve!(cache)
        value = Array(first_sol.u[end].x[1])
        pattern = copy(MN.__mirkn_jacobian_plan(cache).pattern)
        copyto!(prob.p, [1.0])
        second_sol = solve!(cache)
        @test successful_retcode(first_sol) && successful_retcode(second_sol)
        @test MN.__mirkn_jacobian_plan(cache).pattern != pattern
        @test Array(first_sol.u[end].x[1]) == value
        @test Array(second_sol.u[end].x[1]) ≈ [exp(1)] atol = 1.0e-5
    end

end
