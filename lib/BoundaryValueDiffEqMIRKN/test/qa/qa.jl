using SciMLTesting
using BoundaryValueDiffEqMIRKN
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
    BoundaryValueDiffEqMIRKN;
    explicit_imports = true,
    ei_kwargs = (;
        # External internals with no public replacement:
        #   - StandardSecondOrderBVProblem: SciMLBase-owned problem type, not public.
        #   - pickchunksize: ForwardDiff internal.
        all_explicit_imports_are_public = (;
            ignore = (:StandardSecondOrderBVProblem, :pickchunksize),
        ),
    ),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = DOCS_SRC,
        ignore = UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP,
        rendered_ignore = UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP,
    ),
)
