using CUDA
using CUDSS
using LinearSolve
using Test

@static if isdefined(CUDA, :cuSPARSE)
    using CUDA: cuSPARSE
else
    using CUDA: CUSPARSE as cuSPARSE
end

include("device_backend_tests.jl")
include("resident_backend_tests.jl")

function cuda_sparse_pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.81 * sin(u[1])
    return nothing
end

function cuda_sparse_pendulum_bc!(residual, u, p, t)
    residual[1] = u(pi / 4)[1] + pi / 2
    residual[2] = u(pi / 2)[1] - pi / 2
    return nothing
end

function cuda_sparse_problem(to_device, nlls)
    if nlls
        bf = BVPFunction(
            resident_exponential!, resident_nlls_bc!;
            bcresid_prototype = to_device(zeros(3))
        )
        return BVProblem(
            bf, to_device([0.5, 0.5]), (0.0, 1.0), to_device([1.0]); nlls = Val(true)
        )
    end
    return BVProblem(
        cuda_sparse_pendulum!, cuda_sparse_pendulum_bc!,
        to_device([pi / 2, pi / 2]), (0.0, pi / 2); nlls = Val(false)
    )
end

function cuda_sparse_host_cache(prob; dt)
    MIRK = BoundaryValueDiffEqMIRK
    return MIRK.__init_mirk_device(
        prob, MIRK4(; platform = MIRK.CPU()), prob.u0;
        dt, abstol = 1.0e-8, adaptive = false, controller = DefectControl(),
        nlsolve_kwargs = (;), optimize_kwargs = (;), verbose = false
    )
end

function check_cuda_sparse_jacobian(cache, host_cache; atol)
    MIRK = BoundaryValueDiffEqMIRK
    J = MIRK.__mirk_jacobian(cache)
    @test J isa cuSPARSE.CuSparseMatrixCSR
    values = MIRK.SparseArrays.nonzeros(J)
    rowptr, colind = Array(J.rowPtr), Array(J.colVal)
    @test values isa CuArray
    @test MIRK.SparseArrays.nnz(J) < length(J)
    @test cache.host_mesh == host_cache.host_mesh
    x = vec(Array(MIRK.__mirk_states(cache)))
    nlprob = MIRK.__construct_problem(cache, copy(vec(MIRK.__mirk_states(cache))))
    @test nlprob.f.jvp !== nothing
    @test nlprob.f.vjp !== nothing
    direction = collect(range(-0.3, 0.2; length = size(J, 2)))
    weights = collect(range(0.2, 0.6; length = size(J, 1)))
    product_cache = nothing
    for offset in (zeros(length(x)), collect(range(-0.2, 0.4; length = length(x))))
        input = x + offset
        device_input = CuArray(input)
        reference = MIRK.ForwardDiff.jacobian(input) do u
            residual = similar(u, length(host_cache.residual))
            MIRK.__device_residual!(residual, u, host_cache)
        end
        fill!(values, NaN)
        MIRK.__device_jacobian!(J, device_input, cache)
        @test MIRK.SparseArrays.nonzeros(J) === values
        @test Array(J.rowPtr) == rowptr
        @test Array(J.colVal) == colind
        @test collect(J) ≈ reference atol = atol rtol = 2.0e-6
        # Check the CSR storage order independently of conversion to a dense
        # matrix, including structural entries whose derivative is exactly zero.
        ordered_reference = [
            reference[row, colind[k]] for row in axes(J, 1)
                for k in rowptr[row]:(rowptr[row + 1] - 1)
        ]
        @test Array(values) ≈ ordered_reference atol = atol rtol = 2.0e-6

        jvp, vjp = similar(cache.residual), similar(device_input)
        nlprob.f.jvp(jvp, CuArray(direction), device_input, nlprob.p)
        nlprob.f.vjp(vjp, CuArray(weights), device_input, nlprob.p)
        @test Array(jvp) ≈ reference * direction atol = 5atol rtol = 2.0e-6
        @test Array(vjp) ≈ reference' * weights atol = 5atol rtol = 2.0e-6
        plan = MIRK.__mirk_jacobian_plan(cache)
        product = plan.product
        @test product isa cuSPARSE.CuSparseMatrixCSR
        @test product !== J
        @test MIRK.SparseArrays.nonzeros(product) !== values
        product_cache === nothing || @test product === product_cache
        product_cache = product
    end
    return nothing
end

@testset "CUDA MIRK collocation" begin
    if CUDA.functional()
        CUDA.allowscalar(false)
        @testset "CUDSS sparse LU is active" begin
            A = cuSPARSE.CuSparseMatrixCSR(
                BoundaryValueDiffEqMIRK.SparseArrays.sparse([4.0 1.0; 2.0 3.0])
            )
            b = CuArray([6.0, 8.0])
            @test Base.get_extension(LinearSolve, :LinearSolveCUDSSExt) !== nothing
            for linear_algorithm in (nothing, LUFactorization())
                problem = LinearProblem(A, b)
                result = linear_algorithm === nothing ? solve(problem) : solve(problem, linear_algorithm)
                @test successful_retcode(result)
                @test result.u isa CuArray
                @test Array(result.u) ≈ [1.0, 2.0]
            end
        end
        test_device_backend(CUDABackend())
        test_resident_backend(
            CuArray, u -> u isa CuArray, CUDABackend();
            is_jacobian_device = J -> J isa cuSPARSE.CuSparseMatrixCSR
        )

        @testset "CUDA compressed sparse Jacobians / nlls=$nlls" for nlls in (false, true)
            MIRK = BoundaryValueDiffEqMIRK
            prob = cuda_sparse_problem(CuArray, nlls)
            host_cache = cuda_sparse_host_cache(cuda_sparse_problem(identity, nlls); dt = 0.1)
            for jac_alg in (
                    BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 1)),
                    BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 8)),
                    BVPJacobianAlgorithm(AutoFiniteDiff()),
                    BVPJacobianAlgorithm(AutoFiniteDiff(; fdjtype = Val(:central))),
                    BVPJacobianAlgorithm(;
                        bc_diffmode = AutoFiniteDiff(),
                        nonbc_diffmode = AutoForwardDiff(; chunksize = 2)
                    ),
                    BVPJacobianAlgorithm(;
                        bc_diffmode = AutoForwardDiff(; chunksize = 2),
                        nonbc_diffmode = AutoFiniteDiff(; fdjtype = Val(:central))
                    ),
                )
                cache = init(prob, MIRK4(; jac_alg); dt = 0.1, adaptive = false)
                check_cuda_sparse_jacobian(cache, host_cache; atol = 2.0e-6)
                plan = MIRK.__mirk_jacobian_plan(cache)
                @test !plan.boundary_fallback
                @test maximum(group.ncolors for group in plan.groups) <= 8
                if nlls
                    @test size(MIRK.__mirk_jacobian(cache), 1) == size(MIRK.__mirk_jacobian(cache), 2) + 1
                    # Use the default nonlinear algorithm and its sparse
                    # rectangular linear solve, without a user linsolve override.
                    sol = solve!(cache)
                    @test successful_retcode(sol)
                    @test Array(sol.u[end]) ≈ fill(exp(1.0), 2) rtol = 1.0e-4
                end
            end
        end

        @testset "CUDA pendulum mesh scaling" begin
            MIRK = BoundaryValueDiffEqMIRK
            prob = cuda_sparse_problem(CuArray, false)
            for dt in (0.05, 0.001)
                cache = init(prob, MIRK4(); dt, adaptive = false)
                J = MIRK.__mirk_jacobian(cache)
                M, nodes = size(MIRK.__mirk_states(cache))
                @test J isa cuSPARSE.CuSparseMatrixCSR
                @test MIRK.SparseArrays.nnz(J) <= 2M^2 * (nodes - 1) + 4M
                plan = MIRK.__mirk_jacobian_plan(cache)
                @test !plan.boundary_fallback
                @test maximum(group.ncolors for group in plan.groups) <= 8
            end
        end

        @testset "Configured CUDA backend" begin
            platform = CUDABackend(; always_inline = true)
            prob = resident_problem(CuArray, Float64, true, false)
            cache = init(prob, MIRK4(; platform); dt = 0.2, adaptive = false)
            @test cache.alg.platform === platform
        end

        @testset "Dense device buffer resizing" begin
            MIRK = BoundaryValueDiffEqMIRK
            cache = init(
                resident_problem(CuArray, Float64, true, false),
                MIRK4(; nlsolve = NewtonRaphson(; linsolve = LUFactorization()));
                dt = 0.25, adaptive = false, abstol = 1.0e-11
            )
            # Exercise the dense fallback with real CUDA storage, without
            # changing the backend's sparse adapter or mutating the cache.
            fields = map(fieldnames(typeof(cache))) do name
                name === :jacobian_cache && return nothing
                name === :jac_prototype && return similar(cache.y, length(cache.residual) * length(cache.y))
                return getfield(cache, name)
            end
            dense_cache = MIRK.MIRKCache{true, Float64, false, MIRK.NoDiffCacheNeeded, false}(fields...)
            test_resident_resizing(dense_cache, x -> x isa CuArray)
        end

        include("adaptive_sparse_tests.jl")
    else
        @test_skip CUDA.functional()
    end
end
