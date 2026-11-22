using BoundaryValueDiffEqCore, CUDA, CUDSS, LinearSolve
using BoundaryValueDiffEqCore: __device_sparse_linsolve
using LinearAlgebra, SparseArrays, Test

CUDA.functional() || error("Core CUDSS tests require a functional CUDA device.")
CUDA.allowscalar(false)
const CUDSSExt = Base.get_extension(BoundaryValueDiffEqCore, :BoundaryValueDiffEqCoreCUDSSExt)

@testset "Shared matched factorization" begin
    @test CUDSSExt !== nothing
    @test isempty(Test.detect_ambiguities(BoundaryValueDiffEqCore, CUDSSExt; recursive = true))
    for T in (Float32, Float64)
        host = sparse(T[0 2 0; 1 0 3; 0 4 5])
        A = CUDSSExt.CuCSR(host)
        b = CuArray(T[1, 2, 3])
        alg = __device_sparse_linsolve(A)
        @test alg.fact_alg !== __device_sparse_linsolve(A).fact_alg
        prototype = copy(A)
        fill!(nonzeros(prototype), 0)
        cache = init(LinearProblem(prototype, b), alg)
        # No numerical pivots may be chosen from the placeholder Jacobian.
        @test alg.fact_alg.solver === nothing
        cache.A = A
        tolerance = T === Float32 ? T(1.0e-4) : T(1.0e-11)
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test Array(sol.u) ≈ host \ Array(b) atol = tolerance
        factor = alg.fact_alg.solver
        @test factor !== nothing
        for scale in (T(1.3), T(0.7))
            updated = CUDSSExt.CuCSR(scale * host)
            cache.A = updated
            sol = solve!(cache)
            @test successful_retcode(sol)
            @test norm(updated * sol.u - b, Inf) < tolerance
            @test alg.fact_alg.solver === factor
        end
    end
end
