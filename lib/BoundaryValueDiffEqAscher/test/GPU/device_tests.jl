using BoundaryValueDiffEqAscher, SciMLBase, Test, ForwardDiff, SparseArrays, LinearAlgebra

const ASCHER = BoundaryValueDiffEqAscher

function ascher_test_rhs!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1] + p[1] * (u[1]^2 - sin(t)^2)
    return nothing
end
function ascher_test_bc!(r, u, p, t)
    r[1] = u(first(t))[1]
    r[2] = u(last(t))[1] - sin(one(eltype(t)))
    return nothing
end

function ascher_dae_rhs!(du, u, p, t)
    du[1] = u[2]
    du[2] = u[3]
    du[3] = u[3] + u[1] + (u[1]^2 - sin(t)^2) / 10
    return nothing
end

function ascher_reference_jacobian(cache, x = cache.x)
    return ForwardDiff.jacobian(x) do u
        r = similar(u)
        ASCHER.__ascher_device_residual!(r, u, cache)
        return r
    end
end

function test_ascher_device_jacobians(to_device, platform)
    @testset "Colored Jacobian agrees with independent dense AD" begin
        prob = BVProblem(ascher_test_rhs!, ascher_test_bc!, to_device([0.1, 0.8]), (0.0, 1.0), [0.2])
        hostprob = remake(prob; u0 = [0.1, 0.8])
        hostcache = init(hostprob, Ascher3(device = true); dt = 0.25, adaptive = false)
        x = collect(range(0.2, 0.7; length = length(hostcache.x)))
        dense = ascher_reference_jacobian(hostcache, x)
        for mode in (AutoSparse(AutoForwardDiff(chunksize = 3)), AutoFiniteDiff(), AutoFiniteDiff(fdjtype = Val(:central)))
            cache = init(prob, Ascher3(; device = true, platform, jac_alg = BVPJacobianAlgorithm(mode)); dt = 0.25, adaptive = false)
            copyto!(cache.x, x)
            plan = ASCHER.__ascher_jacobian(cache)
            ASCHER.__ascher_device_jacobian!(plan.matrix, cache.x, cache)
            @test Matrix(SparseMatrixCSC(plan.matrix)) ≈ dense atol = 2.0e-7
            @test plan.ncolors < length(x)
            # Repeated evaluations must not mutate unknowns (the legacy residual
            # includes a Newton update, which cannot be differentiated this way).
            ASCHER.__ascher_device_residual!(cache.residual, cache.x, cache)
            @test Array(cache.x) == x
        end
    end
    @testset "Colored Jacobian with coupled interior boundaries" begin
        function coupled_bc!(r, u, p, t)
            r[1] = u(0.17)[1] + u(0.83)[1] - sin(0.17) - sin(0.83)
            r[2] = u(0.17)[2] - u(0.83)[1] - cos(0.17) + sin(0.83)
            return nothing
        end
        hostprob = BVProblem(ascher_test_rhs!, coupled_bc!, [0.1, 0.8], (0.0, 1.0), [0.2])
        hostcache = init(hostprob, Ascher3(device = true); dt = 0.25, adaptive = false)
        x = collect(range(0.2, 0.7; length = length(hostcache.x)))
        expected = ascher_reference_jacobian(hostcache, x)
        for mode in (AutoSparse(AutoForwardDiff(chunksize = 3)), AutoFiniteDiff(), AutoFiniteDiff(fdjtype = Val(:central)))
            prob = remake(hostprob; u0 = to_device([0.1, 0.8]))
            cache = init(prob, Ascher3(; device = true, platform, jac_alg = BVPJacobianAlgorithm(mode)); dt = 0.25, adaptive = false)
            copyto!(cache.x, x)
            J = ASCHER.__ascher_jacobian(cache).matrix
            ASCHER.__ascher_device_jacobian!(J, cache.x, cache)
            @test Matrix(SparseMatrixCSC(J)) ≈ expected atol = 2.0e-7
        end
    end
    @testset "Colored Jacobian with two-point DAE boundaries" begin
        bca!(r, u, p) = (r[1] = u[3]; nothing)
        bcb!(r, u, p) = (r[1] = u[1] - sin(1.0); nothing)
        fun = BVPFunction(ascher_dae_rhs!, (bca!, bcb!); twopoint = Val(true), mass_matrix = Diagonal([1.0, 1.0, 0.0]), bcresid_prototype = (zeros(1), zeros(1)))
        prob = TwoPointBVProblem(fun, to_device([0.1, 0.8, -0.3]), (0.0, 1.0))
        hostcache = init(remake(prob; u0 = [0.1, 0.8, -0.3]), Ascher3(device = true); dt = 0.25, adaptive = false)
        expected = ascher_reference_jacobian(hostcache)
        for mode in (AutoSparse(AutoForwardDiff(chunksize = 3)), AutoFiniteDiff(fdjtype = Val(:central)))
            cache = init(prob, Ascher3(; device = true, platform, jac_alg = BVPJacobianAlgorithm(mode)); dt = 0.25, adaptive = false)
            J = ASCHER.__ascher_jacobian(cache).matrix
            ASCHER.__ascher_device_jacobian!(J, cache.x, cache)
            @test Matrix(SparseMatrixCSC(J)) ≈ expected atol = 2.0e-7
            @test !ASCHER.__ascher_jacobian(cache).separate_boundary
        end
    end
    @testset "Colored Jacobian with algebraic interior boundaries" begin
        function bc!(r, u, p, t)
            r[1] = u(0.21)[3] + sin(0.21)
            r[2] = u(0.73)[2] + u(0.21)[3] - cos(0.73) + sin(0.21)
            return nothing
        end
        fun = BVPFunction(ascher_dae_rhs!, bc!; mass_matrix = Diagonal([1.0, 1.0, 0.0]), bcresid_prototype = zeros(2))
        prob = BVProblem(fun, to_device([0.1, 0.8, -0.3]), (0.0, 1.0))
        hostcache = init(remake(prob; u0 = [0.1, 0.8, -0.3]), Ascher3(device = true); dt = 0.25, adaptive = false)
        expected = ascher_reference_jacobian(hostcache)
        cache = init(prob, Ascher3(; device = true, platform); dt = 0.25, adaptive = false)
        J = ASCHER.__ascher_jacobian(cache).matrix
        ASCHER.__ascher_device_jacobian!(J, cache.x, cache)
        @test Matrix(SparseMatrixCSC(J)) ≈ expected atol = 1.0e-12
    end
    return @testset "Colored Jacobian with wide blocks" begin
        # A coupled 16-state problem exercises multiple AD chunks.
        function wide_rhs!(du, u, p, t)
            h = length(u) ÷ 2
            coupling = zero(eltype(u))
            for j in 1:h
                coupling += u[j] - sin((1 + j / (2h)) * t)
            end
            for j in 1:h
                w = 1 + j / (2h)
                du[j] = u[h + j]
                du[h + j] = -w^2 * u[j] + p * (u[j]^2 - sin(w * t)^2) + coupling / (10h)
            end
        end
        function wide_bc!(r, u, p, t)
            h = length(u(first(t))) ÷ 2
            for j in 1:h
                r[j] = u(first(t))[j]
                r[h + j] = u(last(t))[j] - sin(1 + j / (2h))
            end
        end
        initial = vcat(fill(0.1, 8), fill(0.8, 8))
        prob = BVProblem(wide_rhs!, wide_bc!, to_device(initial), (0.0, 1.0), 0.2)
        alg = Ascher3(;
            device = true, platform,
            jac_alg = BVPJacobianAlgorithm(AutoSparse(AutoForwardDiff(chunksize = 8)))
        )
        cache = init(prob, alg; dt = 0.5, adaptive = false)
        reference = init(
            remake(prob; u0 = initial),
            Ascher3(device = true); dt = 0.5, adaptive = false
        )
        expected = ascher_reference_jacobian(reference)
        J = ASCHER.__ascher_jacobian(cache).matrix
        ASCHER.__ascher_device_jacobian!(J, cache.x, cache)
        @test Matrix(SparseMatrixCSC(J)) ≈ expected atol = 1.0e-12
    end
end

function test_ascher_cpu()
    test_ascher_device_jacobians(identity, CPU())
    return @testset "Device validation and sparse scaling" begin
        prob = BVProblem(ascher_test_rhs!, ascher_test_bc!, [0.1, 0.8], (0.0, 1.0), [0.2])
        badfun = BVPFunction(ascher_test_rhs!, ascher_test_bc!; bcresid_prototype = zeros(3))
        @test_throws DimensionMismatch init(BVProblem(badfun, [0.1, 0.8], (0.0, 1.0)), Ascher2(device = true); dt = 0.1)
        @test_throws ArgumentError init(prob, Ascher2(device = true); dt = 0)
        @test_throws ArgumentError init(remake(prob; tspan = (1.0, 0.0)), Ascher2(device = true); dt = 0.1)
        @test_throws ArgumentError init(prob, Ascher2(device = true, jac_alg = BVPJacobianAlgorithm(AutoEnzyme())); dt = 0.1)
        small = init(prob, Ascher3(device = true); dt = 0.1)
        big = init(prob, Ascher3(device = true); dt = 0.01)
        @test ASCHER.__ascher_jacobian(big).ncolors <= ASCHER.__ascher_jacobian(small).ncolors + 2
        @test nnz(ASCHER.__ascher_jacobian(big).matrix) < 11nnz(ASCHER.__ascher_jacobian(small).matrix)
    end
end
