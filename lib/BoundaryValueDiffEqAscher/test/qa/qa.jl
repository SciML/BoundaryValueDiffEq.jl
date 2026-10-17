using SciMLTesting
using BoundaryValueDiffEqAscher
using Test

include(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "reexports.jl"))

run_qa(
    BoundaryValueDiffEqAscher;
    ei_kwargs = (;
        # StandardBVProblem is a SciMLBase-owned BVP problem type that this solver
        # legitimately dispatches on but which SciMLBase does not mark public.
        all_explicit_imports_are_public = (; ignore = (:StandardBVProblem,)),
        # ForwardDiff.Dual / ForwardDiff.jacobian! are ForwardDiff internals with
        # no public replacement.
        all_qualified_accesses_are_public = (; ignore = (:Dual, :jacobian!)),
    ),
    reexports_allow = ASCHER_REEXPORTS,
)

test_reexport_surface(BoundaryValueDiffEqAscher, ASCHER_REEXPORTS, @__MODULE__)
