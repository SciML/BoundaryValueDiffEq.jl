using SciMLTesting
using BoundaryValueDiffEqShooting
using Test

const DOCS_SRC = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src"))
const UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP = (
    :AutoModelingToolkit,
    :AutoSparseFastDifferentiation,
    :AutoSparseFiniteDiff,
    :AutoSparseForwardDiff,
    :AutoSparsePolyesterForwardDiff,
    :AutoSparseReverseDiff,
    :AutoSparseZygote,
    :pickchunksize,
)

run_qa(
    BoundaryValueDiffEqShooting;
    explicit_imports = true,
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
    api_docs_kwargs = (;
        rendered = true,
        docs_src = DOCS_SRC,
        ignore = UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP,
        rendered_ignore = UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP,
    ),
)
