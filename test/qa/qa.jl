using SciMLTesting
using BoundaryValueDiffEq
using SciMLBase
using JET
using Test

include("reexports.jl")

run_qa(
    BoundaryValueDiffEq;
    aqua_kwargs = (;
        ambiguities = false,
        piracies = (; treat_as_own = [SciMLBase.BVProblem]),
    ),
    reexports_allow = ROOT_REEXPORTS,
    api_docs_kwargs = (; ignore = ROOT_REEXPORTS, rendered_ignore = ROOT_REEXPORTS),
)
