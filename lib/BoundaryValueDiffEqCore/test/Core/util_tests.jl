using BoundaryValueDiffEqCore
using SciMLBase
using Test

@testset "Singular terms with appended parameters" begin
    using BoundaryValueDiffEqCore: __device_singular!

    # Keep nonzero sentinels outside the physical-state block so an unchecked
    # read beyond S produces a deterministic failure rather than allocator noise.
    storage = fill(99.0, 4, 4)
    storage[1:2, 1:2] .= [1.0 2.0; 3.0 4.0]
    S = view(storage, 1:2, 1:2)
    u = [2.0, 3.0, 11.0, 13.0]
    initial = [5.0, 7.0, 0.0, 0.0]
    du = copy(initial)
    __device_singular!(du, S, u, 2.0)
    @test du == [9.0, 16.0, 0.0, 0.0]

    for t in (0.0, -1.0)
        du = copy(initial)
        __device_singular!(du, S, u, t)
        @test du == initial
    end
    du = copy(initial)
    __device_singular!(du, nothing, u, 2.0)
    @test du == initial
end

@testset "Resizable buffer views" begin
    using BoundaryValueDiffEqCore: __reshape_buffer, __device_sparse_supported,
        __device_sparse_matrix
    using SparseArrays: sparse

    buffer = collect(1.0:6.0)
    shaped = __reshape_buffer(buffer, 2, 3)
    @test size(shaped) == (2, 3)
    shaped[2, 2] = 42
    @test buffer[4] == 42
    @test __device_sparse_supported(shaped)
    pattern = sparse([1, 2], [1, 3], [1.0, 1.0], 2, 3)
    @test size(__device_sparse_matrix(shaped, pattern).matrix) == size(pattern)

    # A normal Array reshape permanently shares storage on Julia 1.10, even
    # when the shaped object is no longer retained by a solver cache.
    resize!(buffer, 12)
    fill!(buffer, 3)
    @test __reshape_buffer(buffer, (3, 4)) == fill(3, 3, 4)
    resize!(buffer, 3)
    @test vec(__reshape_buffer(buffer, 1, 3)) == fill(3, 3)
end

module ExternalBVPAlgorithmExtension
    using BoundaryValueDiffEqCore, SciMLBase

    struct ExternalBVPAlgorithm <: BoundaryValueDiffEqCore.AbstractBoundaryValueDiffEqAlgorithm end
    struct ExternalBVPCache{P} <: BoundaryValueDiffEqCore.AbstractBoundaryValueDiffEqCache
        prob::P
        init_arg::Symbol
        adaptive::Bool
    end

    SciMLBase.__init(
        prob::SciMLBase.AbstractBVProblem, ::ExternalBVPAlgorithm, init_arg::Symbol;
        adaptive = true, kwargs...
    ) = ExternalBVPCache(prob, init_arg, adaptive)

    SciMLBase.solve!(cache::ExternalBVPCache) =
        (; cache.prob, cache.init_arg, cache.adaptive)

    struct ExternalCombinedErrorControl <: BoundaryValueDiffEqCore.AbstractErrorControl end

    BoundaryValueDiffEqCore.__use_both_error_control(::ExternalCombinedErrorControl) = true
end

@testset "AbstractBoundaryValueDiffEqAlgorithm extension interface" begin
    @test ExternalBVPAlgorithmExtension.ExternalBVPAlgorithm <:
    BoundaryValueDiffEqCore.AbstractBoundaryValueDiffEqAlgorithm
    @test ExternalBVPAlgorithmExtension.ExternalBVPCache <:
    BoundaryValueDiffEqCore.AbstractBoundaryValueDiffEqCache

    f(u, p, t) = u
    bc(u, p, t) = u
    prob = SciMLBase.BVProblem(f, bc, [1.0], (0.0, 1.0))
    sol = SciMLBase.solve(
        prob, ExternalBVPAlgorithmExtension.ExternalBVPAlgorithm(), :from_solve;
        adaptive = false
    )

    @test sol.prob === prob
    @test sol.init_arg === :from_solve
    @test !sol.adaptive
    @test !SciMLBase.isinplace(
        ExternalBVPAlgorithmExtension.ExternalBVPCache(prob, :test, true)
    )
end

@testset "AbstractErrorControl extension interface" begin
    @test !BoundaryValueDiffEqCore.__use_both_error_control(DefectControl())
    @test BoundaryValueDiffEqCore.__use_both_error_control(
        ExternalBVPAlgorithmExtension.ExternalCombinedErrorControl()
    )
end

@testset "__extract_lcons_ucons length" begin
    # Regression test: the function must return vectors matching the actual
    # constraint vector length (= length(resid_prototype)), not a reconstruction
    # from (M, N, ...) which was wrong for several solvers.
    using BoundaryValueDiffEqCore: __extract_lcons_ucons
    using SciMLBase: BVProblem

    f!(du, u, p, t) = (du[1] = u[2]; du[2] = -u[1])
    bc!(res, u, p, t) = (res[1] = u(0.0)[1]; res[2] = u(1.0)[1])

    # Fallback path (isnothing(prob.lcons)): both vectors have length == constraint_length
    prob = BVProblem(f!, bc!, [0.0, 0.0], (0.0, 1.0); bcresid_prototype = zeros(2))
    lc, uc = __extract_lcons_ucons(prob, Float64, 42)
    @test length(lc) == 42
    @test length(uc) == 42
    @test all(iszero, lc)
    @test all(iszero, uc)

    # User-provided lcons/ucons: values preserved, padded with zeros to constraint_length
    prob2 = BVProblem(
        f!, bc!, [0.0, 0.0], (0.0, 1.0);
        bcresid_prototype = zeros(2),
        lcons = [-1.0, -2.0], ucons = [1.0, 2.0]
    )
    lc2, uc2 = __extract_lcons_ucons(prob2, Float64, 10)
    @test length(lc2) == 10
    @test length(uc2) == 10
    @test lc2[1:2] == [-1.0, -2.0]
    @test uc2[1:2] == [1.0, 2.0]
    @test all(iszero, lc2[3:end])
    @test all(iszero, uc2[3:end])
end

@testset "_process_verbose_param foreign AbstractVerbositySpecifier" begin
    # DiffEqBase.DEVerbosity is a foreign AbstractVerbositySpecifier that
    # can flow in via DiffEqBase's `solve`/`init` default `verbose` kwarg.
    # It must not hit a MethodError at precompile time; it should fall
    # back to BVP's own DEFAULT_VERBOSE (a BVPVerbosity).
    using BoundaryValueDiffEqCore, DiffEqBase
    result = BoundaryValueDiffEqCore._process_verbose_param(DiffEqBase.DEFAULT_VERBOSE)
    @test result isa BoundaryValueDiffEqCore.BVPVerbosity
    @test result === BoundaryValueDiffEqCore.DEFAULT_VERBOSE
end

module SecondOrderExternalAlgorithmExtension
    using BoundaryValueDiffEqCore, SciMLBase

    struct FirstOrderAlg <: BoundaryValueDiffEqCore.AbstractBoundaryValueDiffEqAlgorithm end
    struct InitOnlyAlg <: BoundaryValueDiffEqCore.AbstractBoundaryValueDiffEqAlgorithm end
    struct SolveOnlyAlg <: BoundaryValueDiffEqCore.AbstractBoundaryValueDiffEqAlgorithm end
    struct ExtCache{P} <: BoundaryValueDiffEqCore.AbstractBoundaryValueDiffEqCache
        prob::P
    end

    SciMLBase.__init(prob::SciMLBase.BVProblem, ::FirstOrderAlg; kwargs...) = ExtCache(prob)
    SciMLBase.__init(prob::SciMLBase.AbstractBVProblem, ::InitOnlyAlg; kwargs...) =
        ExtCache(prob)
    SciMLBase.solve!(cache::ExtCache) = (; cache.prob, reached = :ext_init)

    SciMLBase.__solve(
        prob::SciMLBase.AbstractBVProblem, ::SolveOnlyAlg, args...; kwargs...
    ) = (; prob, reached = :ext_solve)
end

const SO_EXT = SecondOrderExternalAlgorithmExtension
const SO_PROB = SecondOrderBVProblem(
    (ddu, du, u, p, t) -> (ddu .= 0; nothing), (res, du, u, p, t) -> (res .= 0; nothing),
    [1.0, -1.0], (0.0, 1.0)
)

@testset "SecondOrderBVProblem with a first-order-only algorithm" begin
    err = @test_throws ArgumentError SciMLBase.solve(SO_PROB, SO_EXT.FirstOrderAlg(); dt = 0.2)
    @test occursin(
        "SecondOrderBVProblem is only supported by MIRKN solvers (MIRKN4, MIRKN6). " *
            "Got FirstOrderAlg.", sprint(showerror, err.value)
    )
end

@testset "SecondOrderBVProblem with external `$(nameof(typeof(alg)))`" for (alg, reached) in (
        (SO_EXT.InitOnlyAlg(), :ext_init), (SO_EXT.SolveOnlyAlg(), :ext_solve),
    )
    @test SciMLBase.solve(SO_PROB, alg; dt = 0.2) == (; prob = SO_PROB, reached)
end

@testset "No ambiguities with external `__init`/`__solve` methods" begin
    @test isempty(Test.detect_ambiguities(BoundaryValueDiffEqCore, SO_EXT))
end
