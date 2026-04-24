using InteractiveUtils, Test

@info sprint(InteractiveUtils.versioninfo)

@testset "BoundaryValueDiffEqCore.jl" begin
    @testset "Aqua" begin
        using Aqua, BoundaryValueDiffEqCore

        Aqua.test_all(BoundaryValueDiffEqCore; piracies = false, ambiguities = false, stale_deps = false)
        Aqua.test_stale_deps(BoundaryValueDiffEqCore; ignore = [:TimerOutputs])
        Aqua.test_piracies(BoundaryValueDiffEqCore)
        Aqua.test_ambiguities(BoundaryValueDiffEqCore; recursive = false)
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
end
