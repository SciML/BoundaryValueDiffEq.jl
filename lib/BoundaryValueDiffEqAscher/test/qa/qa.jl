using SciMLTesting
using BoundaryValueDiffEqAscher
using Test

const DOCS_SRC = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src"))
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
    api_docs_kwargs = (;
        docs_src = DOCS_SRC, ignore = ASCHER_REEXPORTS,
        rendered_ignore = ASCHER_REEXPORTS,
    ),
)
