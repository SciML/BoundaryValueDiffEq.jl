using SciMLTesting
using BoundaryValueDiffEq
using SciMLBase
using JET
using Test

const DOCS_SRC = normpath(joinpath(@__DIR__, "..", "..", "docs", "src"))

const ADTYPES_REEXPORTS_WITH_UPSTREAM_DOC_OWNERSHIP = (
    :AutoModelingToolkit,
    :AutoSparseFastDifferentiation,
    :AutoSparseFiniteDiff,
    :AutoSparseForwardDiff,
    :AutoSparsePolyesterForwardDiff,
    :AutoSparseReverseDiff,
    :AutoSparseZygote,
)

run_qa(
    BoundaryValueDiffEq;
    explicit_imports = true,
    aqua_kwargs = (;
        ambiguities = false,
        piracies = (; treat_as_own = [SciMLBase.BVProblem]),
    ),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = DOCS_SRC,
        ignore = ADTYPES_REEXPORTS_WITH_UPSTREAM_DOC_OWNERSHIP,
        rendered_ignore = ADTYPES_REEXPORTS_WITH_UPSTREAM_DOC_OWNERSHIP,
    ),
)
