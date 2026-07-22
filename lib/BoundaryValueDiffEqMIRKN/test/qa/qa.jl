using SciMLTesting
using BoundaryValueDiffEqMIRKN
using Test

const DOCS_SRC = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "reexports.jl"))

run_qa(
    BoundaryValueDiffEqMIRKN;
    ei_kwargs = (;
        # External internals with no public replacement:
        #   - StandardSecondOrderBVProblem: SciMLBase-owned problem type, not public.
        #   - pickchunksize: ForwardDiff internal.
        all_explicit_imports_are_public = (;
            ignore = (:StandardSecondOrderBVProblem, :pickchunksize),
        ),
    ),
    reexports_allow = MIRKN_REEXPORTS,
    api_docs_kwargs = (;
        docs_src = DOCS_SRC, ignore = MIRKN_REEXPORTS,
        rendered_ignore = MIRKN_REEXPORTS,
    ),
)
