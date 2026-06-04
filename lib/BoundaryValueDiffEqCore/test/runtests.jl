using InteractiveUtils, Test

@info sprint(InteractiveUtils.versioninfo)

# Standard sublibrary test groups (Core / QA). The root test/runtests.jl
# activates this sublibrary and sets BVDE_TEST_GROUP to the standard group name.
# Core runs functional/correctness tests; QA runs the Aqua quality checks.
const GROUP = get(ENV, "BVDE_TEST_GROUP", "All")

@testset "BoundaryValueDiffEqCore.jl" begin
    if GROUP in ("QA", "All")
        @testset "Aqua" begin
            using Aqua, BoundaryValueDiffEqCore

            Aqua.test_all(BoundaryValueDiffEqCore; piracies = false, ambiguities = false, stale_deps = false)
            Aqua.test_stale_deps(BoundaryValueDiffEqCore; ignore = [:TimerOutputs])
            Aqua.test_piracies(BoundaryValueDiffEqCore)
            Aqua.test_ambiguities(BoundaryValueDiffEqCore; recursive = false)
        end
    end

    if GROUP in ("Core", "All")
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
end
