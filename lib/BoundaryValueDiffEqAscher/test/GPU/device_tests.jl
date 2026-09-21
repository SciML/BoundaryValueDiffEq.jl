using BoundaryValueDiffEqAscher, SciMLBase, Test, ForwardDiff, SparseArrays, LinearAlgebra
using StaticArrays: SVector

const ASCHER = BoundaryValueDiffEqAscher

function ascher_test_rhs!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1] + p[1] * (u[1]^2 - sin(t)^2)
    return nothing
end
function ascher_test_bc!(r, u, p, t)
    r[1] = u[1]
    r[2] = u[1] - sin(one(t))
    return nothing
end
ascher_test_rhs(u, p, t) = SVector(u[2], -u[1] + p[1] * (u[1]^2 - sin(t)^2))
ascher_test_bc(u, p, t) = SVector(u[1], u[1] - sin(one(t)))

function ascher_device_suite(to_device, platform; gpu = false)
    @testset "Gauss stages and polynomial interpolation" begin
        for Alg in (Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7)
            prob = BVProblem(ascher_test_rhs!, ascher_test_bc!, to_device([0.1, 0.8]), (0.0, 1.0), [0.2])
            alg = Alg(; device = true, zeta = [0.0, 1.0], platform)
            sol = solve(prob, alg; dt = 0.05, adaptive = false, abstol = 1.0e-10)
            @test successful_retcode(sol)
            tol = Alg === Ascher1 ? 3.0e-4 : 2.0e-7
            @test maximum(maximum(abs, Array(u) .- [sin(t), cos(t)]) for (t, u) in zip(sol.t, sol.u)) < tol
            @test Array(sol(0.37)) ≈ [sin(0.37), cos(0.37)] atol = (Alg === Ascher1 ? 5.0e-4 : 5.0e-6)
            gpu && @test !(ASCHER.__ascher_initial_backend(first(sol.u)) isa CPU)
        end
    end
    @testset "Colored Jacobian agrees with independent dense AD" begin
        prob = BVProblem(ascher_test_rhs!, ascher_test_bc!, to_device([0.1, 0.8]), (0.0, 1.0), [0.2])
        hostprob = remake(prob; u0 = [0.1, 0.8])
        hostcache = init(hostprob, Ascher3(device = true, zeta = [0.0, 1.0]); dt = 0.25, adaptive = false)
        x = collect(range(0.2, 0.7; length = length(hostcache.x)))
        dense = ForwardDiff.jacobian(x) do u
            r = similar(u)
            ASCHER.__ascher_device_residual!(r, u, hostcache)
            r
        end
        for mode in (AutoSparse(AutoForwardDiff(chunksize = 3)), AutoFiniteDiff(), AutoFiniteDiff(fdjtype = Val(:central)))
            cache = init(prob, Ascher3(; device = true, zeta = [0.0, 1.0], platform, jac_alg = BVPJacobianAlgorithm(mode)); dt = 0.25, adaptive = false)
            copyto!(cache.x, x)
            ASCHER.__ascher_device_jacobian!(BoundaryValueDiffEqAscher.__ascher_jacobian(cache).matrix, cache.x, cache)
            @test Matrix(SparseMatrixCSC(BoundaryValueDiffEqAscher.__ascher_jacobian(cache).matrix)) ≈ dense atol = 2.0e-7
            @test BoundaryValueDiffEqAscher.__ascher_jacobian(cache).ncolors < length(x)
            # Repeated evaluations must not mutate unknowns (the legacy residual
            # includes a Newton update, which cannot be differentiated this way).
            ASCHER.__ascher_device_residual!(cache.residual, cache.x, cache)
            @test Array(cache.x) == x
        end
    end
    @testset "Nonlinear DAE and endpoint conditions" begin
        function dae!(du, u, p, t)
            du[1] = u[2]
            du[2] = u[3]
            du[3] = u[3] + u[1] + (u[1]^2 - sin(t)^2) / 10
        end
        bca!(r, u, p) = (r[1] = u[1]; nothing)
        bcb!(r, u, p) = (r[1] = u[1] - sin(1.0); nothing)
        fun = BVPFunction(dae!, (bca!, bcb!); twopoint = Val(true), mass_matrix = Diagonal([1.0, 1.0, 0.0]), bcresid_prototype = (zeros(1), zeros(1)))
        prob = TwoPointBVProblem(fun, to_device([0.1, 0.8, -0.3]), (0.0, 1.0))
        sol = solve(prob, Ascher3(; device = true, platform); dt = 0.1, adaptive = false, abstol = 1.0e-10)
        @test successful_retcode(sol)
        @test Array(sol(0.37)) ≈ [sin(0.37), cos(0.37), -sin(0.37)] atol = 1.0e-5
        @test Array(last(sol.u)) ≈ [sin(1.0), cos(1.0), -sin(1.0)] atol = 2.0e-5
    end
    @testset "Out-of-place, Float32, and cache reuse" begin
        p = Float32[0.2]
        prob = BVProblem(ascher_test_rhs, ascher_test_bc, to_device(Float32[0.1, 0.8]), (0.0f0, 1.0f0), p)
        cache = init(prob, Ascher3(; device = true, zeta = [0.0, 1.0], platform); dt = 0.1f0, adaptive = false, abstol = 2.0f-6)
        firstsol = solve!(cache)
        @test successful_retcode(firstsol)
        preserved = Array(firstsol(0.37f0))
        p[1] = 0.3f0
        second = solve!(cache)
        @test successful_retcode(second)
        @test Array(second(0.37f0)) ≈ Float32[sin(0.37), cos(0.37)] atol = 2.0f-5
        @test Array(firstsol(0.37f0)) == preserved
        @test firstsol(0.37f0; idxs = 1) ≈ sin(0.37f0) atol = 2.0f-5
        @test Array(firstsol([0.2f0, 0.7f0]).u[2]) ≈ Float32[sin(0.7), cos(0.7)] atol = 2.0f-5
        @test_throws BoundsError firstsol(0.5f0; idxs = 0)
    end
    @testset "Scaled mass and function initial guess" begin
        f!(du, u, p, t) = (du[1] = 2u[1]; nothing)
        bc!(r, u, p, t) = (r[1] = u[1] - 1; nothing)
        fun = BVPFunction(f!, bc!; mass_matrix = Diagonal([2.0]))
        prob = BVProblem(fun, (p, t) -> to_device([1 + t]), (0.0, 1.0))
        sol = solve(prob, Ascher3(; device = true, platform, zeta = [0.0]); dt = 0.1, adaptive = false, abstol = 1.0e-10)
        @test successful_retcode(sol)
        @test Array(last(sol.u)) ≈ [exp(1)] atol = 1.0e-8
    end
    @testset "Wide blocks and numerical pivot matching" begin
        # Boundary/continuity rows have structural zeros on the diagonal. This
        # coupled 16-state problem exercises multiple AD chunks and cuDSS's
        # numerical matching, which must wait for the first actual Jacobian.
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
            h = length(u) ÷ 2
            for j in 1:h
                r[j] = u[j]
                r[h + j] = u[j] - sin(1 + j / (2h))
            end
        end
        initial = vcat(fill(0.1, 8), fill(0.8, 8))
        prob = BVProblem(wide_rhs!, wide_bc!, to_device(initial), (0.0, 1.0), 0.2)
        alg = Ascher3(;
            device = true, platform, zeta = vcat(zeros(8), ones(8)),
            jac_alg = BVPJacobianAlgorithm(AutoSparse(AutoForwardDiff(chunksize = 8)))
        )
        cache = init(prob, alg; dt = 0.5, adaptive = false)
        reference = init(
            remake(prob; u0 = initial),
            Ascher3(device = true, zeta = alg.zeta); dt = 0.5, adaptive = false
        )
        expected = ForwardDiff.jacobian(reference.x) do x
            r = similar(x)
            ASCHER.__ascher_device_residual!(r, x, reference)
            r
        end
        ASCHER.__ascher_device_jacobian!(BoundaryValueDiffEqAscher.__ascher_jacobian(cache).matrix, cache.x, cache)
        @test Matrix(SparseMatrixCSC(BoundaryValueDiffEqAscher.__ascher_jacobian(cache).matrix)) ≈ expected atol = 1.0e-12
        sol = solve(prob, alg; dt = 1 / 32, adaptive = false, abstol = 1.0e-9)
        @test successful_retcode(sol)
        @test Array(sol(0.37))[1:8] ≈ [sin((1 + j / 16) * 0.37) for j in 1:8] atol = 1.0e-8
    end
    return @testset "Interior side conditions and adaptive refinement" begin
        function bc!(r, u, p, t)
            r[1] = u[1] - sin(0.3)
            r[2] = u[1] - sin(1.0)
        end
        prob = BVProblem(ascher_test_rhs!, bc!, to_device([0.1, 0.8]), (0.0, 1.0), [0.2])
        sol = solve(prob, Ascher2(; device = true, zeta = [0.3, 1.0], platform); dt = 0.25, abstol = 2.0e-6)
        @test successful_retcode(sol)
        @test 0.3 in sol.t
        @test length(sol.t) > 5
        @test Array(sol(0.47)) ≈ [sin(0.47), cos(0.47)] atol = 1.0e-5
        failed = solve(prob, Ascher2(; device = true, zeta = [0.3, 1.0], platform, max_num_subintervals = 5); dt = 0.25, abstol = 1.0e-12)
        @test failed.retcode == ReturnCode.MaxIters
        failednl = solve(prob, Ascher2(; device = true, zeta = [0.3, 1.0], platform); dt = 0.25, adaptive = false, nlsolve_kwargs = (; maxiters = 1, abstol = 1.0e-14))
        @test !successful_retcode(failednl)
    end
end

@testset "Ascher portable device formulation on CPU" begin
    ascher_device_suite(identity, CPU())
end

@testset "Device validation and sparse scaling" begin
    prob = BVProblem(ascher_test_rhs!, ascher_test_bc!, [0.1, 0.8], (0.0, 1.0), [0.2])
    @test_throws DimensionMismatch init(prob, Ascher2(device = true); dt = 0.1)
    @test_throws ArgumentError init(prob, Ascher2(device = true, zeta = [0.0, 1.0]); dt = 0)
    @test_throws ArgumentError init(prob, Ascher2(device = true, zeta = [0.0, 2.0]); dt = 0.1)
    @test_throws ArgumentError init(prob, Ascher2(device = true, zeta = [0.0, 1.0], jac_alg = BVPJacobianAlgorithm(AutoEnzyme())); dt = 0.1)
    small = init(prob, Ascher3(device = true, zeta = [0.0, 1.0]); dt = 0.1)
    big = init(prob, Ascher3(device = true, zeta = [0.0, 1.0]); dt = 0.01)
    @test BoundaryValueDiffEqAscher.__ascher_jacobian(big).ncolors <= BoundaryValueDiffEqAscher.__ascher_jacobian(small).ncolors + 2
    @test nnz(BoundaryValueDiffEqAscher.__ascher_jacobian(big).matrix) < 11nnz(BoundaryValueDiffEqAscher.__ascher_jacobian(small).matrix)
end
