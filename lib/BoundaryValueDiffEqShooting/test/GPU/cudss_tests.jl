using BoundaryValueDiffEqCore, BoundaryValueDiffEqShooting, CUDA, CUDSS, LinearSolve
using LinearAlgebra, SparseArrays, OrdinaryDiffEqTsit5, Test

const ShootingCUDA = Base.get_extension(BoundaryValueDiffEqShooting, :BoundaryValueDiffEqShootingCUDSSExt)
const ShootingModule = BoundaryValueDiffEqShooting
CUDA.allowscalar(false)

@testset "Lazy cuDSS cache and symbolic reuse" begin
    @test ShootingCUDA !== nothing
    @test isempty(Test.detect_ambiguities(ShootingModule, ShootingCUDA; recursive = true))
    for T in (Float32, Float64)
        tolerance = T === Float32 ? T(2.0e-5) : T(1.0e-11)
        host = sparse(T[0 2 0; 1 0 3; 0 4 5])
        A = ShootingCUDA.CuCSR(host)
        b = CuArray(T[1, 2, 3])
        cache = init(LinearProblem(A, b), ShootingCUDA.CachedFactorization(nothing); abstol = tolerance, reltol = tolerance)
        stats = cache.cacheval.stats
        @test cache.cacheval.full.factor === nothing
        @test stats.analyses == stats.factorizations == stats.solves == 0
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test Array(sol.u) ≈ host \ Array(b) atol = 10tolerance
        @test stats.analyses == stats.factorizations == 1
        factor = cache.cacheval.full.factor

        cache.b = CuArray(T[3, -1, 2])
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test Array(sol.u) ≈ host \ Array(cache.b) atol = 10tolerance
        @test stats.analyses == stats.factorizations == 1
        @test cache.cacheval.full.factor === factor

        host.nzval .*= T(1.3)
        host[1, 2] = T(200)
        host[3, 2] = T(0.04)
        A2 = ShootingCUDA.CuCSR(host)
        cache.A = A2
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test Array(sol.u) ≈ host \ Array(cache.b) atol = 10tolerance
        @test stats.analyses == 1 && stats.factorizations == 2
        @test cache.cacheval.full.factor === factor

        # Same dimensions AND nnz, but different row/column indices.
        changed = sparse(T[2 1 0; 0 3 4; 5 0 0])
        @test nnz(changed) == nnz(host)
        cache.A = ShootingCUDA.CuCSR(changed)
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test Array(sol.u) ≈ changed \ Array(cache.b) atol = 10tolerance
        @test stats.analyses == 2 && stats.factorizations == 3
        @test cache.cacheval.full.factor !== factor

        # Reinitializing with a different dimension replaces the analysis and
        # resizes residual/refinement workspaces as well as the external vectors.
        larger = sparse(T[3 1 0 0; 1 4 1 0; 0 1 5 1; 0 0 1 2])
        cache.A = ShootingCUDA.CuCSR(larger)
        cache.b = CUDA.ones(T, 4)
        cache.u = CUDA.zeros(T, 4)
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test Array(sol.u) ≈ larger \ ones(T, 4) atol = 10tolerance
        @test stats.analyses == 3 && stats.factorizations == 4

        nonzeros(cache.A) .= T(NaN)
        cache.A = cache.A
        @test !successful_retcode(solve!(cache))
        @test stats.factorizations == 4
    end
end

function condensation_test_rhs!(du, u, p, t)
    @inbounds for j in eachindex(u)
        du[j] = zero(eltype(u))
        for k in eachindex(u)
            if !iszero(p[j, k])
                du[j] += p[j, k] * u[k]
            end
        end
    end
    return nothing
end
function condensation_test_left!(r, u, p)
    r[1] = u[1] + u[2]
    r[2] = u[1] - u[2]
    return nothing
end
function condensation_test_right!(r, u, p)
    r[1] = u[3] + u[4]
    r[2] = u[3] - u[4]
    return nothing
end

function condensation_test_setup(T, steps, segment_length, mode = AutoForwardDiff(); dense = false)
    # Non-contiguous independent flow components (1, 3) and (2, 4), with
    # boundary conditions coupling them. No oscillator/pair indexing shortcuts.
    matrix = dense ? T[-1 0.1 0.2 0.3; 0.3 -2 0.1 0.2; 0.1 0.2 -1 0.3; 0.2 0.1 0.3 -2] :
        T[0 0 1 0; 0 0 0 2; -1 0 0 0; 0 -2 0 0]
    prob = TwoPointBVProblem(
        condensation_test_rhs!, (condensation_test_left!, condensation_test_right!),
        T[0.3, 0.4, 0.5, 0.6], (T(0), T(1)), matrix;
        bcresid_prototype = (zeros(T, 2), zeros(T, 2)), nlls = Val(false)
    )
    alg = MultipleShooting(
        steps, Tsit5(); platform = CUDA.CUDABackend(), device_steps = 8,
        jac_alg = BVPJacobianAlgorithm(mode)
    )
    setup = ShootingModule.__shooting_device_setup(prob, alg)
    (; u, cache, plan) = setup
    ShootingModule.__shooting_jacobian!(plan.matrix, u, cache, plan)
    linsolve = ShootingCUDA.default_linsolve(u, cache, plan; segment_length)
    return setup, linsolve
end

@testset "Segment condensation solves the original system" begin
    for T in (Float32, Float64), segment_length in (2, 7, 32)
        setup, alg = condensation_test_setup(T, 37, segment_length)
        A = setup.plan.matrix
        truth = CuArray(T[sin(i / 10) for i in axes(A, 2)])
        b = A * truth
        tolerance = T === Float32 ? T(1.0e-5) : T(1.0e-11)
        cache = init(LinearProblem(A, b), alg; abstol = tolerance, reltol = tolerance)
        state = cache.cacheval
        @test state.active
        @test state.work.components == 2
        @test size(state.work.matrix, 1) == 4 * (cld(37, segment_length) + 1)
        @test state.reduced.factor === nothing && state.full.factor === nothing
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test norm(A * sol.u - b, Inf) <= 2tolerance
        @test Array(sol.u) ≈ Array(truth) atol = (T === Float32 ? 2.0e-4 : 1.0e-9)
        @test state.stats.fallbacks == 0
        @test state.full.factor === nothing
        @test state.stats.analyses == state.stats.factorizations == 1
        factor = state.reduced.factor

        # Updating just b must neither recompose nor refactor the matrix.
        cache.b = 2b
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test norm(A * sol.u - cache.b, Inf) <= 2tolerance
        @test state.stats.analyses == state.stats.factorizations == 1
        # Scale ALL coefficients, including the matching-node diagonal. The
        # reduction must use D_i, rather than assuming an exact constant -I.
        nonzeros(A) .*= T(1.25)
        cache.A = A
        cache.b = A * truth
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test norm(A * sol.u - cache.b, Inf) <= 2tolerance
        @test state.stats.analyses == 1 && state.stats.factorizations == 2
        @test state.reduced.factor === factor
        @test state.stats.fallbacks == 0
    end

    @testset "Finite-difference Jacobian" begin
        setup, alg = condensation_test_setup(Float64, 13, 5, AutoFiniteDiff())
        A = setup.plan.matrix
        truth = CUDA.ones(Float64, size(A, 2))
        cache = init(LinearProblem(A, A * truth), alg; abstol = 1.0e-11, reltol = 1.0e-11)
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test norm(sol.u - truth, Inf) < 1.0e-9
        @test cache.cacheval.stats.fallbacks == 0
    end

    @testset "Changed structure invalidates condensation" begin
        setup, alg = condensation_test_setup(Float64, 13, 5)
        A = setup.plan.matrix
        truth = CUDA.ones(Float64, size(A, 2))
        cache = init(LinearProblem(A, A * truth), alg; abstol = 1.0e-11, reltol = 1.0e-11)
        @test successful_retcode(solve!(cache))
        changed = SparseMatrixCSC(A)
        changed[4, 9] = 0.1 # introduce a coupling beyond adjacent nodes
        cache.A = ShootingCUDA.CuCSR(changed)
        cache.b = cache.A * truth
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test norm(sol.u - truth, Inf) < 1.0e-9
        @test !cache.cacheval.active
        @test cache.cacheval.stats.fallbacks == 1
    end

    @testset "Fully coupled flow" begin
        setup, alg = condensation_test_setup(Float64, 19, 6; dense = true)
        A = setup.plan.matrix
        truth = CuArray([cos(i / 10) for i in axes(A, 2)])
        cache = init(LinearProblem(A, A * truth), alg; abstol = 1.0e-11, reltol = 1.0e-11)
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test norm(sol.u - truth, Inf) < 1.0e-9
        @test cache.cacheval.work.components == 1
        @test cache.cacheval.stats.fallbacks == 0
    end
end

@testset "Unsafe transfers fall back to full sparse solve" begin
    rhs!(du, u, p, t) = (du[1] = 100u[1]; du[2] = -100u[2]; nothing)
    left!(r, u, p) = (r[1] = u[2]; nothing)
    right!(r, u, p) = (r[1] = u[1]; nothing)
    prob = TwoPointBVProblem(
        rhs!, (left!, right!), ones(2), (0.0, 1.0);
        bcresid_prototype = (zeros(1), zeros(1)), nlls = Val(false)
    )
    alg = MultipleShooting(17, Tsit5(); platform = CUDA.CUDABackend(), device_steps = 8)
    (; u, cache, plan) = ShootingModule.__shooting_device_setup(prob, alg)
    ShootingModule.__shooting_jacobian!(plan.matrix, u, cache, plan)
    linear_alg = ShootingCUDA.default_linsolve(u, cache, plan; segment_length = 8)
    A = plan.matrix
    truth = CUDA.ones(Float64, size(A, 2))
    linear_cache = init(LinearProblem(A, A * truth), linear_alg; abstol = 1.0e-10, reltol = 1.0e-12)
    sol = solve!(linear_cache)
    @test successful_retcode(sol)
    @test norm(sol.u - truth, Inf) < 1.0e-9
    @test linear_cache.cacheval.stats.fallbacks == 1
    @test linear_cache.cacheval.full.factor !== nothing
    @test linear_cache.cacheval.reduced.factor === nothing
end

@testset "Device linear solver keyword validation" begin
    @test_throws ArgumentError MultipleShooting(5, Tsit5(); device_linsolve = LUFactorization())
    @test_throws ArgumentError MultipleShooting(
        5, Tsit5(); device_steps = 4,
        device_linsolve = LUFactorization(), nlsolve = NewtonRaphson()
    )
end

@testset "One-sided and nonlinear boundary solves" begin
    rhs!(du, u, p, t) = (du[1] = -u[1]; du[2] = -2u[2]; nothing)
    boundary!(r, u, p) = (r[1] = u[1] - 1; r[2] = u[2] - 1; nothing)
    empty!(r, u, p) = nothing
    for left in (true, false)
        bc = left ? (boundary!, empty!) : (empty!, boundary!)
        prototype = left ? (zeros(2), zeros(0)) : (zeros(0), zeros(2))
        prob = TwoPointBVProblem(rhs!, bc, ones(2), (0.0, 1.0); bcresid_prototype = prototype, nlls = Val(false))
        alg = MultipleShooting(37, Tsit5(); platform = CUDA.CUDABackend(), device_steps = 8)
        sol = solve(prob, alg; abstol = 1.0e-10)
        @test successful_retcode(sol)
        @test maximum(abs, sol.resid) < 1.0e-10
        expected = left ? [exp(-1), exp(-2)] : [1.0, 1.0]
        @test Array(sol.u[end]) ≈ expected atol = 1.0e-9
    end
    # Force repeated Jacobian updates through the public NonlinearSolve API.
    nonlinear_left!(r, u, p) = (r[1] = u[1]^2 - 1; nothing)
    nonlinear_right!(r, u, p) = (r[1] = u[2]^2 - 1; nothing)
    prob = TwoPointBVProblem(
        rhs!, (nonlinear_left!, nonlinear_right!), [0.8, 1.4], (0.0, 1.0);
        bcresid_prototype = (zeros(1), zeros(1)), nlls = Val(false)
    )
    nlsolve = NewtonRaphson(; linsolve = LUFactorization())
    if hasproperty(nlsolve, :jacobian_reuse)
        nlsolve = NewtonRaphson(; linsolve = LUFactorization(), jacobian_reuse = false)
    end
    alg = MultipleShooting(37, Tsit5(); platform = CUDA.CUDABackend(), device_steps = 8, nlsolve)
    sol = solve(prob, alg; abstol = 1.0e-10)
    @test successful_retcode(sol)
    @test maximum(abs, sol.resid) < 1.0e-10
    @test sol.original.stats.njacs > 1
end

# Exercise default selection with CUDSS loaded, including multipoint boundaries,
# least squares, Float32/64, and out-of-place functions.
device_shooting_tests(CUDA.CUDABackend(); gpu = true)
