using SciMLTesting
using BoundaryValueDiffEq
using SciMLBase
using JET
using Test

const DOCS_SRC = normpath(joinpath(@__DIR__, "..", "..", "docs", "src"))

function upstream_reexports_with_doc_ownership(pkg, owners, extra = ())
    names = Set{Symbol}(extra)
    for owner in owners
        isdefined(pkg, owner) || continue
        union!(names, SciMLTesting.public_api_names(getproperty(pkg, owner)))
        push!(names, owner)
    end
    return Tuple(sort!(collect(names)))
end

const UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP = upstream_reexports_with_doc_ownership(
    BoundaryValueDiffEq,
    (:ADTypes, :SciMLBase, :SciMLOperators),
    (:AllObserved, :deleteat!, :init, :solve, :solve!, :step!)
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
        ignore = UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP,
        rendered_ignore = UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP,
    ),
)
