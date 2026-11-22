using BoundaryValueDiffEqCore
using BoundaryValueDiffEqCore: AbstractBoundaryValueDiffEqCache, SparseJacobianCache,
    __concrete_device_solve_algorithm, __concrete_solve_algorithm, __concrete_kwargs,
    __default_linsolve, __default_sparse_linsolve, __device_sparse_linsolve, __internal_solve,
    __needs_sparse_damping
using KernelAbstractions: CPU
using LinearSolve: KrylovJL_GMRES, KrylovJL_LSMR, LUFactorization, UMFPACKFactorization
using NonlinearSolveBase: NonlinearSolveBase, NonlinearSolvePolyAlgorithm
using SparseArrays: sparse
using Test

struct InternalSolverCache{A, J} <: AbstractBoundaryValueDiffEqCache
    alg::A
    jacobian_cache::J
end

@testset "Shared linear solver defaults" begin
    @test __default_linsolve(zeros(2)) === nothing
    # A non-Array storage token exercises the backend fallback without a GPU.
    @test __default_linsolve(nothing) == KrylovJL_GMRES()
    A = sparse([2.0 1.0; 1.0 3.0])
    @test __default_sparse_linsolve(A) isa UMFPACKFactorization
    @test __device_sparse_linsolve(A) === nothing
    @test __default_sparse_linsolve(sparse(Float32[2 1; 1 3])) == KrylovJL_GMRES()
end

@testset "Shared internal nonlinear solvers" begin
    square = NonlinearProblem(
        NonlinearFunction((u, p) -> u .- [1, 2]; jac = (u, p) -> [1.0 0.0; 0.0 1.0]),
        zeros(2)
    )
    rectangular = NonlinearLeastSquaresProblem(
        NonlinearFunction(
            (u, p) -> [u[1] - 1, 2u[1] - 2, u[2] - 2];
            jac = (u, p) -> [1.0 0.0; 2.0 0.0; 0.0 1.0]
        ), zeros(2)
    )
    for prob in (square, rectangular), linesearch_fallback in (false, true)
        algorithm = __concrete_device_solve_algorithm(
            prob, nothing, nothing; concrete_jac = true, linesearch_fallback
        )
        sol = __internal_solve(prob, algorithm; abstol = 1.0e-10)
        @test successful_retcode(sol)
        @test sol.u ≈ [1, 2]
        members = linesearch_fallback ? algorithm.algs : (algorithm,)
        @test length(members) == (linesearch_fallback ? 2 : 1)
        @test all(a -> !__needs_sparse_damping(a, prob.u0), members)
        if prob === rectangular
            @test all(a -> a.descent.linsolve == KrylovJL_LSMR(), members)
        end
    end

    chosen = LUFactorization()
    explicit = NewtonRaphson(; linsolve = chosen)
    optimizer = :explicit_optimizer
    sparse_plan = SparseJacobianCache((), false, nothing, nothing)
    @test __concrete_device_solve_algorithm(square, explicit, nothing) === explicit
    @test __concrete_device_solve_algorithm(square, nothing, optimizer) === optimizer
    @test_throws ErrorException __concrete_device_solve_algorithm(square, explicit, optimizer)
    algorithm = __concrete_device_solve_algorithm(square, nothing, nothing; linsolve = chosen)
    @test algorithm.descent.linsolve === chosen

    for prob in (square, rectangular)
        for (nlsolve, optimize) in ((explicit, nothing), (nothing, optimizer))
            cache = InternalSolverCache((; nlsolve, optimize, platform = :device), Ref(sparse_plan))
            @test __concrete_solve_algorithm(prob, cache) === something(nlsolve, optimize)
        end

        default = (; nlsolve = nothing, optimize = nothing)
        expected_cpu_members = prob === square ? 3 : 5
        for (platform, plan, members) in (
                (CPU(), nothing, expected_cpu_members),
                (CPU(), sparse_plan, expected_cpu_members),
                (:device, sparse_plan, 2),
            )
            cache = InternalSolverCache((; default..., platform), Ref(plan))
            algorithm = __concrete_solve_algorithm(prob, cache)
            @test algorithm isa NonlinearSolvePolyAlgorithm
            @test length(algorithm.algs) == members
            if platform === :device
                @test !__needs_sparse_damping(algorithm, prob.u0)
            elseif prob === rectangular && plan !== nothing
                @test first(algorithm.algs).descent.linsolve == KrylovJL_LSMR()
            end
        end
    end

    complex_prob = NonlinearProblem((u, p) -> u .- 1, ComplexF64[0])
    @test length(__concrete_solve_algorithm(complex_prob, nothing).algs) == 1
    @test !__needs_sparse_damping(nothing, complex_prob.u0)
    damping = isdefined(NonlinearSolveBase, :MoreTrustRegionDescent)
    @test __needs_sparse_damping(nothing, square.u0) == damping
    @test __needs_sparse_damping(
        NonlinearSolvePolyAlgorithm((NewtonRaphson(), TrustRegion())), square.u0
    ) == damping
end

@testset "Internal solver keyword precedence" begin
    verbose = BVPVerbosity()
    nonlinear = (; maxiters = 7)
    explicit = (; maxiters = 9, verbose = false)
    @test __concrete_kwargs(nothing, nothing, nonlinear, (;), verbose).maxiters == 7
    @test __concrete_kwargs(nothing, nothing, explicit, (;)) == explicit
    @test __concrete_kwargs(NewtonRaphson(), nothing, explicit, (;), verbose) == explicit
    @test __concrete_kwargs(nothing, :optimizer, nonlinear, explicit, verbose) == explicit
end
