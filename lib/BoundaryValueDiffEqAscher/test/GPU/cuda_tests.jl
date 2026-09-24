using CUDA, CUDSS, LinearSolve, Test
CUDA.functional() || error("Ascher CUDA tests require a functional NVIDIA GPU.")
CUDA.allowscalar(false)

include("device_tests.jl")

@testset "Ascher CUDA with scalar indexing disabled" begin
    ascher_device_suite(CuArray, CUDA.CUDABackend(); gpu = true)
end

@testset "CUDA selection and cuDSS sparse Newton" begin
    for (state, platform) in ((CuArray([0.1, 0.8]), CPU()), ([0.1, 0.8], CUDA.CUDABackend()))
        prob = BVProblem(ascher_test_rhs!, ascher_test_bc!, state, (0.0, 1.0), [0.2])
        alg = Ascher3(; zeta = [0.0, 1.0], platform, nlsolve = NewtonRaphson(linsolve = LUFactorization()))
        cache = init(prob, alg; dt = 0.1, adaptive = false, abstol = 1.0e-10)
        @test cache.x isa CuArray
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test Array(sol(0.37)) ≈ [sin(0.37), cos(0.37)] atol = 1.0e-6
    end
end

include("resizing_tests.jl")
@testset "Ascher CUDA flat buffer resizing" test_ascher_flat_buffers(CuArray, CUDA.CUDABackend())
