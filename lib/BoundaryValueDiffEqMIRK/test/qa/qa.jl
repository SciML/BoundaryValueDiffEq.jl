using SciMLTesting
using BoundaryValueDiffEqMIRK
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
    BoundaryValueDiffEqMIRK;
    explicit_imports = true,
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
    api_docs_kwargs = (;
        rendered = true,
        docs_src = DOCS_SRC,
        ignore = UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP,
        rendered_ignore = UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP,
    ),
)
