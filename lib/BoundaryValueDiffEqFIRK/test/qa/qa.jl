using SciMLTesting
using BoundaryValueDiffEqFIRK
using Test

const DOCS_SRC = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "reexports.jl"))

run_qa(
    BoundaryValueDiffEqFIRK;
    ei_kwargs = (;
        # External internals with no public replacement:
        #   - StandardBVProblem: SciMLBase-owned problem type, not public.
        #   - pickchunksize: ForwardDiff internal.
        all_explicit_imports_are_public = (;
            ignore = (:StandardBVProblem, :pickchunksize),
        ),
        # SciMLStructures interface (Tunable/canonicalize/isscimlstructure) is not
        # marked public.
        all_qualified_accesses_are_public = (;
            ignore = (:Tunable, :canonicalize, :isscimlstructure),
        ),
    ),
    reexports_allow = FIRK_REEXPORTS,
    api_docs_kwargs = (;
        docs_src = DOCS_SRC, ignore = FIRK_REEXPORTS,
        rendered_ignore = FIRK_REEXPORTS,
    ),
)
