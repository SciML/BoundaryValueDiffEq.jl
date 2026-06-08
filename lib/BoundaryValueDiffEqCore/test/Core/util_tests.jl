using BoundaryValueDiffEqCore
using Test

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
