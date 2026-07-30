using SciMLTesting
using BoundaryValueDiffEqShooting
using Test

const DOCS_SRC = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src"))
# `SHOOTING_REEXPORTS` is the audited set of ADTypes / BoundaryValueDiffEqCore /
# SciMLBase names that BoundaryValueDiffEqShooting deliberately re-exports; it is kept
# in sync with the re-export `export` block in src/BoundaryValueDiffEqShooting.jl.
include(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "reexports.jl"))

run_qa(
    BoundaryValueDiffEqShooting;
    aqua_kwargs = (; persistent_tasks = false),
    ei_kwargs = (;
        # All external internals with no public replacement:
        #   - StandardBVProblem: SciMLBase-owned BVP problem type, not public.
        #   - overloaded_input_type: DifferentiationInterface internal.
        #   - pickchunksize: ForwardDiff internal.
        all_explicit_imports_are_public = (;
            ignore = (:StandardBVProblem, :overloaded_input_type, :pickchunksize),
        ),
    ),
    reexports_allow = SHOOTING_REEXPORTS,
    api_docs_kwargs = (;
        docs_src = DOCS_SRC, ignore = SHOOTING_REEXPORTS,
        rendered_ignore = SHOOTING_REEXPORTS,
    ),
)
