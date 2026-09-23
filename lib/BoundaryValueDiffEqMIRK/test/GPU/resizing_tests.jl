using BoundaryValueDiffEqMIRK, Test

function test_resident_resizing(cache, is_device)
    MIRK = BoundaryValueDiffEqMIRK
    fields = (
        :mesh, :mesh_dt, :y, :k_discrete, :k_interp, :collocation_cache,
        :fᵢ₂_cache, :residual, :errors,
    )
    owners = map(name -> getproperty(cache, name), fields)
    dense = cache.jac_prototype
    saved = solve!(cache)
    @test successful_retcode(saved)
    @test !ismutabletype(typeof(cache))
    @test all(buffer -> buffer isa AbstractVector && is_device(buffer), owners)

    # Grow, reuse the same size, shrink, and grow again. Warm the AD workspace
    # before remeshing so stale shaped views must be discarded on every path.
    for nodes in (9, 9, 3, 17)
        times = collect(range(first(saved.t), last(saved.t); length = 5))[2:4]
        oldmesh = copy(saved.t)
        oldvalues = map(t -> Array(saved(t)), times)
        oldderivs = map(t -> Array(saved(t, Val{1})), times)
        target = collect(range(first(saved.t), last(saved.t); length = nodes))
        expected = reduce(hcat, map(t -> Array(saved(t)), target))
        MIRK.__device_jacobian!(MIRK.__mirk_jacobian(cache), cache.y, cache)
        @test !isempty(cache.device_cache)

        MIRK.__mirk_device_remesh!(cache, target)
        @test isempty(cache.device_cache)
        @test all(getproperty(cache, name) === owner for (name, owner) in zip(fields, owners))
        @test cache.jac_prototype === dense
        @test Array(cache.mesh) == cache.host_mesh == target
        @test Array(cache.mesh_dt) ≈ diff(target)
        @test Array(MIRK.__mirk_states(cache)) ≈ expected atol = 1.0e-10
        @test size(MIRK.__mirk_stages(cache)) == (cache.M, cache.stage, nodes - 1)
        @test size(MIRK.__mirk_interp_stages(cache)) == (cache.M, cache.ITU.s_star - cache.stage, nodes - 1)
        @test size(MIRK.__mirk_collocation(cache)) == (cache.M, nodes - 1)
        @test size(MIRK.__mirk_jacobian(cache)) == (length(cache.residual), length(cache.y))

        next = solve!(cache)
        @test successful_retcode(next)
        @test all(getproperty(cache, name) === owner for (name, owner) in zip(fields, owners))
        @test saved.t == oldmesh
        @test map(t -> Array(saved(t)), times) == oldvalues
        @test map(t -> Array(saved(t, Val{1})), times) == oldderivs
        saved = next
    end
    return nothing
end
