using BoundaryValueDiffEqFIRK, Test

function test_firk_flat_buffers(upload, platform)
    FK = BoundaryValueDiffEqFIRK
    f!(du, u, p, t) = (du[1] = u[2]; du[2] = u[1]; nothing)
    bc!(r, u, p, t) = (r[1] = u(first(t))[1] - 1; r[2] = u(last(t))[1] - exp(1); nothing)
    @testset "Flat FIRK nested=$nested sparse=$sparse" for nested in (false, true), sparse in (false, true)
        prob = BVProblem(f!, bc!, upload([0.5, 0.5]), (0.0, 1.0))
        mode = sparse ? AutoSparse(AutoForwardDiff(; chunksize = 2)) : AutoForwardDiff(; chunksize = 2)
        cache = FK.__init_firk_device(
            prob, RadauIIa3(;
                nested_nlsolve = nested, platform,
                jac_alg = BVPJacobianAlgorithm(mode), nlsolve = NewtonRaphson()
            ), prob.u0;
            dt = 0.125, adaptive = false, abstol = 1.0e-10
        )
        # Core can choose a sparse local stencil even for a dense AD mode. Exercise
        # the dense storage fallback independently of the installed extension.
        if !sparse
            fields = map(fieldnames(typeof(cache))) do name
                name === :jacobian_cache && return nothing
                name === :jac_prototype && return similar(cache.y, length(cache.residual) * length(cache.unknowns))
                getfield(cache, name)
            end
            cache_type = nested ? FK.FIRKCacheNested : FK.FIRKCacheExpand
            cache = cache_type{true, Float64, FK.NoDiffCacheNeeded, false}(fields...)
        end
        @test cache isa (nested ? FK.FIRKCacheNested : FK.FIRKCacheExpand)
        names = (:y, :unknowns, :mesh, :mesh_dt, :residual)
        owners = map(name -> getproperty(cache, name), names)
        dense = cache.jac_prototype
        saved = solve!(cache)
        @test successful_retcode(saved)
        @test !ismutabletype(typeof(cache))
        @test all(x -> x isa AbstractVector, owners)
        @test all(x -> typeof(BoundaryValueDiffEqFIRK.KernelAbstractions.get_backend(x)) === typeof(platform), owners)
        @test nested ? cache.y !== cache.unknowns : cache.y === cache.unknowns
        for nodes in (17, 17, 5, 33)
            times = [0.13, 0.43, 0.87]
            oldvalues = map(t -> Array(saved(t)), times)
            oldderivs = map(t -> Array(saved(t, Val{1})), times)
            target = collect(range(0.0, 1.0; length = nodes)) .^ 1.1
            expected = reduce(hcat, map(t -> Array(saved(t)), target))
            FK.__device_jacobian!(FK.__firk_jacobian(cache), cache.unknowns, cache)
            scratch = copy(cache.work_buffers)
            @test !isempty(cache.device_cache)
            FK.__firk_refine!(cache, target)
            @test isempty(cache.device_cache)
            @test all(getproperty(cache, name) === owner for (name, owner) in zip(names, owners))
            @test cache.jac_prototype === dense
            @test Array(cache.mesh) == cache.host_mesh == target
            @test Array(cache.mesh_dt) ≈ diff(target)
            @test Array(view(FK.__firk_states(cache), :, 1:(cache.TU.s + 1):size(FK.__firk_states(cache), 2))) ≈ expected atol = 1.0e-10
            @test size(FK.__firk_jacobian(cache)) == (length(cache.residual), length(cache.unknowns))
            next = solve!(cache)
            @test successful_retcode(next)
            @test all(cache.work_buffers[key] === owner for (key, owner) in scratch)
            @test map(t -> Array(saved(t)), times) == oldvalues
            @test map(t -> Array(saved(t, Val{1})), times) == oldderivs
            @test Array(next(0.37)) ≈ fill(exp(0.37), 2) atol = 2.0e-5
            saved = next
        end
    end
    return test_firk_buffer_features(upload, platform)
end

function test_firk_buffer_features(upload, platform)
    FK = BoundaryValueDiffEqFIRK
    f!(du, u, p, t) = (du[1] = u[2]; du[2] = u[1]; nothing)
    bc!(r, u, p, t) = (r[1] = u(first(t))[1] - 1; r[2] = u(last(t))[1] - exp(1); nothing)
    @testset "Adaptive FIRK flat buffers" for nested in (false, true), controller in
            (DefectControl(), GlobalErrorControl(method = REErrorControl()))
        prob = BVProblem(f!, bc!, upload([0.5, 0.5]), (0.0, 1.0))
        cache = FK.__init_firk_device(
            prob, RadauIIa3(; nested_nlsolve = nested, platform), prob.u0;
            dt = 0.5, abstol = 1.0e-7, controller
        )
        owner = cache.y
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test length(sol.t) > 3
        @test cache.y === owner
        @test Array(sol(0.37)) ≈ fill(exp(0.37), 2) atol = 2.0e-5
    end
    return @testset "FIRK boundary sparsity on repeated solve" begin
        timed_bc!(r, u, p, t) = (r[1] = u(p[1])[1] - exp(p[1]); r[2] = u(p[2])[1] - exp(p[2]); nothing)
        prob = BVProblem(f!, timed_bc!, upload([0.5, 0.5]), (0.0, 1.0), upload([0.13, 0.87]))
        cache = FK.__init_firk_device(prob, RadauIIa3(; platform), prob.u0; dt = 0.1, adaptive = false)
        first_sol = solve!(cache)
        pattern = copy(FK.__firk_jacobian_plan(cache).pattern)
        copyto!(prob.p, [0.33, 0.67])
        second_sol = solve!(cache)
        @test successful_retcode(first_sol) && successful_retcode(second_sol)
        @test FK.__firk_jacobian_plan(cache).pattern != pattern
        @test Array(second_sol(0.37)) ≈ fill(exp(0.37), 2) atol = 1.0e-5
    end

end
