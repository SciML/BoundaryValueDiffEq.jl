using InteractiveUtils, Test

@info sprint(InteractiveUtils.versioninfo)

# Standard sublibrary test groups (Core / QA). The centralized sublibrary CI
# routes the emitted matrix `group` here via BVDE_TEST_GROUP (Core / QA). When
# run directly with GROUP set, honor the standard `<pkg>` / `<pkg>_<GROUP>`
# naming via the prefix-strip shim. Core runs functional/correctness tests; QA
# runs the Aqua quality checks.
const _SUB = "BoundaryValueDiffEqCore"
const _G = get(ENV, "GROUP", "All")
const GROUP = get(
    ENV, "BVDE_TEST_GROUP",
    _G == _SUB ? "Core" :
        (startswith(_G, _SUB * "_") ? _G[(length(_SUB) + 2):end] : _G)
)

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
