using BoundaryValueDiffEqCore
using BoundaryValueDiffEqCore: AbstractBoundaryValueDiffEqCache, SparseJacobianCache,
    __concrete_device_solve_algorithm, __concrete_solve_algorithm, __concrete_kwargs,
    __default_linsolve, __default_sparse_linsolve, __internal_solve, __needs_sparse_damping
using KernelAbstractions: CPU
using LinearSolve: KrylovJL_GMRES, KrylovJL_LSMR, LUFactorization, UMFPACKFactorization
using NonlinearSolveBase: NonlinearSolveBase, NonlinearSolvePolyAlgorithm, NonlinearVerbosity
using OptimizationBase: OptimizationVerbosity
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
    @test __default_sparse_linsolve(sparse([2.0 1.0; 1.0 3.0])) isa UMFPACKFactorization
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

    explicit = NewtonRaphson(; linsolve = LUFactorization())
    optimizer = :explicit_optimizer
    for prob in (square, rectangular)
        @test __concrete_device_solve_algorithm(prob, explicit, nothing) === explicit
        @test __concrete_device_solve_algorithm(prob, nothing, optimizer) === optimizer
        @test_throws ErrorException __concrete_device_solve_algorithm(prob, explicit, optimizer)
        for platform in (CPU(), :device), plan in (nothing, SparseJacobianCache((), false, nothing, nothing))
            cache = InternalSolverCache((; nlsolve = explicit, optimize = nothing, platform), Ref(plan))
            @test __concrete_solve_algorithm(prob, cache) === explicit
            cache = InternalSolverCache((; nlsolve = nothing, optimize = optimizer, platform), Ref(plan))
            @test __concrete_solve_algorithm(prob, cache) === optimizer

            cache = InternalSolverCache((; nlsolve = nothing, optimize = nothing, platform), Ref(plan))
            algorithm = __concrete_solve_algorithm(prob, cache)
            @test algorithm isa NonlinearSolvePolyAlgorithm
            if plan !== nothing && platform === :device
                @test length(algorithm.algs) == 2
                @test all(a -> !__needs_sparse_damping(a, prob.u0), algorithm.algs)
            elseif prob === square
                @test length(algorithm.algs) == 3
            else
                @test length(algorithm.algs) == 5
                if plan !== nothing
                    @test first(algorithm.algs).descent.linsolve == KrylovJL_LSMR()
                end
            end
        end
    end
    chosen = LUFactorization()
    algorithm = __concrete_device_solve_algorithm(square, nothing, nothing; linsolve = chosen)
    @test algorithm.descent.linsolve === chosen

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
    for nonlinear in (nothing, NewtonRaphson())
        kwargs = __concrete_kwargs(nonlinear, nothing, (; maxiters = 7), (;), verbose)
        @test kwargs.maxiters == 7
        @test kwargs.verbose isa NonlinearVerbosity
        explicit = (; maxiters = 9, verbose = false)
        @test __concrete_kwargs(nonlinear, nothing, explicit, (;), verbose) == explicit
        @test __concrete_kwargs(nonlinear, nothing, explicit, (;)) == explicit
    end
    kwargs = __concrete_kwargs(nothing, :optimizer, (; maxiters = 1), (; maxiters = 8), verbose)
    @test kwargs.maxiters == 8
    @test kwargs.verbose isa OptimizationVerbosity
    explicit = (; maxiters = 4, verbose = false)
    @test __concrete_kwargs(nothing, :optimizer, (;), explicit, verbose) == explicit
end
