using CUDA, CUDSS, Test
using BoundaryValueDiffEqCore, BoundaryValueDiffEqFIRK, SparseArrays, LinearAlgebra
import LinearSolve
CUDA.functional() || error("The FIRK GPU test group requires a functional CUDA device.")
CUDA.allowscalar(false)
const FIRK_GPU_TESTS = true
include("device_backend_tests.jl")
include("device_regression_tests.jl")
include("device_features_tests.jl")
include("device_nested_tests.jl")

@testset "FIRK CUDA" begin
    @info "Checking all FIRK methods on CUDA"
    test_device_backend(CuArray, x -> x isa CuArray, CUDABackend(); gpu = true)
    @info "Checking CUDA nonlinear solves and state shapes"
    test_device_regressions(CuArray, CUDABackend())
    @info "Checking CUDA adaptivity, parameter fitting and DAEs"
    test_device_features(CuArray, CUDABackend())
    @info "Checking resident nested FIRK on CUDA"
    test_nested_device(CuArray, CUDABackend())
    @info "Checking host solves with CUDA collocation"
    test_host_offload(CUDABackend())
    test_offload_features(CUDABackend())
    test_device_singular(CuArray, CUDABackend())
    prob = device_problem(CuArray, Float64, true, false)
    cache = init(prob, RadauIIa3(); dt = 0.1, adaptive = false)
    @test cache.alg.platform isa CUDABackend
    @test SparseArrays.nonzeros(BoundaryValueDiffEqFIRK.__firk_jacobian(cache)) isa CuArray
    @test size(BoundaryValueDiffEqFIRK.__firk_jacobian(cache), 1) == length(cache.residual)
    @test Base.get_extension(BoundaryValueDiffEqCore, :BoundaryValueDiffEqCoreCUDSSExt) !== nothing
    @test BoundaryValueDiffEqCore.__device_sparse_linsolve(BoundaryValueDiffEqFIRK.__firk_jacobian(cache)) !== nothing

    @testset "Sparse derivatives and matched factorization reuse" begin
        # High-stage two-point systems have zero diagonal entries. A small
        # nonlinear residual alone must not hide an inaccurate linear solve.
        prob = device_problem(CuArray, Float64, false, true)
        cache = init(prob, RadauIIa7(); dt = 0.05, adaptive = false)
        host = packed_init(
            device_problem(identity, Float64, false, true), RadauIIa7();
            dt = 0.05, adaptive = false
        )
        u, J, rhs = vec(BoundaryValueDiffEqFIRK.__firk_states(cache)), BoundaryValueDiffEqFIRK.__firk_jacobian(cache), cache.residual
        u .+= CuArray(collect(range(0.01, 0.1; length = length(u))))
        DF.__device_jacobian!(J, u, cache)
        reference = ForwardDiff.jacobian(Array(u)) do x
            residual = similar(x, length(rhs))
            DF.__device_residual!(residual, x, host)
        end
        @test Matrix(SparseMatrixCSC(J)) ≈ reference atol = 1.0e-10
        factorize = BoundaryValueDiffEqCore.__device_sparse_linsolve(J).fact_alg
        factor = factorize(J)
        for parameter in (1.0, 2.0)
            copyto!(cache.p, [parameter])
            DF.__device_jacobian!(J, u, cache)
            DF.__device_residual!(rhs, u, cache)
            @test factorize(J) === factor
            step = factor \ rhs
            @test norm(J * step - rhs, Inf) < 1.0e-10
        end
        # NonlinearSolve allocates a Jacobian prototype before evaluating it.
        # Initializing from that storage must not determine numerical pivots.
        prototype = copy(J)
        fill!(nonzeros(prototype), 0)
        linear = LinearSolve.init(
            LinearSolve.LinearProblem(prototype, copy(rhs)), BoundaryValueDiffEqCore.__device_sparse_linsolve(J)
        )
        linear.A = J
        step = LinearSolve.solve!(linear).u
        @test norm(J * step - rhs, Inf) < 1.0e-10
    end
end

include("resizing_tests.jl")
@testset "FIRK CUDA flat buffer resizing" test_firk_flat_buffers(CuArray, CUDA.CUDABackend())
