using BoundaryValueDiffEqAscher, Test

function test_ascher_flat_buffers(upload, platform)
    AS = BoundaryValueDiffEqAscher
    f!(du, u, p, t) = (du[1] = u[2]; du[2] = -u[1]; nothing)
    bc!(r, u, p, t) = (r[1] = u[1]; r[2] = u[1] - sin(1); nothing)
    @testset "Flat Ascher $Alg" for Alg in (Ascher2, Ascher3)
        prob = BVProblem(f!, bc!, upload([0.1, 0.8]), (0.0, 1.0))
        cache = init(
            prob, Alg(; platform, device = true, zeta = [0.0, 1.0]);
            dt = 0.125, adaptive = false, abstol = 1.0e-10
        )
        @test cache isa AS.AscherCache
        names = (:x, :mesh, :residual, :locations)
        owners = map(name -> getproperty(cache, name), names)
        saved = solve!(cache)
        @test successful_retcode(saved)
        @test !ismutabletype(typeof(cache))
        @test all(x -> x isa AbstractVector, owners)
        @test all(x -> typeof(BoundaryValueDiffEqAscher.KernelAbstractions.get_backend(x)) === typeof(platform), owners)
        for nodes in (17, 17, 5, 33)
            times = [0.13, 0.43, 0.87]
            oldvalues = map(t -> Array(saved(t)), times)
            oldstages = Array(saved.interp.x)
            target = collect(range(0.0, 1.0; length = nodes)) .^ 1.1
            expected = reduce(hcat, map(t -> Array(saved(t)), target))
            AS.__ascher_device_jacobian!(AS.__ascher_jacobian(cache).matrix, cache.x, cache)
            work = copy(cache.work)
            coarse = AS.__ascher_refine!(cache, target)
            @test Array(coarse) ≈ expected atol = 1.0e-12
            @test all(getproperty(cache, name) === owner for (name, owner) in zip(names, owners))
            @test Array(cache.mesh) == cache.host_mesh == target
            @test Array(cache.locations) == cache.host_locations == [1, nodes]
            @test size(AS.__ascher_jacobian(cache).matrix) == (length(cache.residual), length(cache.x))
            next = solve!(cache)
            @test successful_retcode(next)
            @test all(cache.work[key] === buffers for (key, buffers) in work)
            @test all(buffer -> buffer isa AbstractVector, Iterators.flatten(values(cache.work)))
            @test map(t -> Array(saved(t)), times) == oldvalues
            @test Array(saved.interp.x) == oldstages
            @test Array(next(0.37)) ≈ [sin(0.37), cos(0.37)] atol = 3.0e-4
            saved = next
        end
    end
    return test_ascher_buffer_features(upload, platform)
end

function test_ascher_buffer_features(upload, platform)
    AS = BoundaryValueDiffEqAscher
    f!(du, u, p, t) = (du[1] = u[2]; du[2] = -u[1]; nothing)
    bc!(r, u, p, t) = (r[1] = u[1]; r[2] = u[1] - sin(1); nothing)
    prob = BVProblem(f!, bc!, upload([0.1, 0.8]), (0.0, 1.0))
    cache = init(prob, Ascher3(; platform, device = true, zeta = [0.0, 1.0]); dt = 0.25, abstol = 1.0e-8)
    owner = cache.x
    sol = solve!(cache)
    @test successful_retcode(sol)
    @test length(cache.host_mesh) == length(sol.t) > 5
    @test cache.x === owner
    interior_bc!(r, u, p, t) = (r[1] = u[1] - sin(0.3); r[2] = u[1] - sin(1); nothing)
    prob = BVProblem(f!, interior_bc!, upload([0.1, 0.8]), (0.0, 1.0))
    cache = init(prob, Ascher2(; platform, device = true, zeta = [0.3, 1.0]); dt = 0.25, abstol = 2.0e-6)
    sol = solve!(cache)
    @test successful_retcode(sol)
    @test Array(sol(0.47)) ≈ [sin(0.47), cos(0.47)] atol = 1.0e-5
    @test cache.host_locations == [searchsortedfirst(cache.host_mesh, 0.3), length(cache.host_mesh)]
    @test Array(cache.locations) == cache.host_locations

    return @testset "Ascher adaptive DAE buffers" begin
        function dae_rhs!(du, u, p, t)
            du[1] = u[2]
            du[2] = u[3]
            du[3] = u[3] + u[1] + (u[1]^2 - sin(t)^2) / 10
            return nothing
        end
        fun = BVPFunction(dae_rhs!, bc!; mass_matrix = AS.LinearAlgebra.Diagonal([1.0, 1.0, 0.0]), bcresid_prototype = zeros(2))
        prob = BVProblem(fun, upload([0.1, 0.8, -0.1]), (0.0, 1.0))
        cache = init(prob, Ascher3(; device = true, platform, zeta = [0.0, 1.0]); dt = 0.25, abstol = 1.0e-7)
        owner = cache.x
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test cache.ncomp == 2 && cache.M == 3
        @test length(cache.host_mesh) > 5
        @test length(cache.x) == (length(cache.host_mesh) - 1) * (2 + 3cache.k) + 2
        @test cache.x === owner
        @test Array(sol(0.47)) ≈ [sin(0.47), cos(0.47), -sin(0.47)] atol = 1.0e-6
    end

end
