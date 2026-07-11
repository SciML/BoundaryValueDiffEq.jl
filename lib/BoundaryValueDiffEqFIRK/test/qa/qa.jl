using SciMLTesting
using BoundaryValueDiffEqFIRK
using Test

const DOCS_SRC = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src"))

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
    BoundaryValueDiffEqFIRK,
    (:ADTypes, :NonlinearSolveFirstOrder, :SciMLBase, :SciMLOperators),
    (
        :AllObserved, :BoundaryValueDiffEqCore, :deleteat!, :init, :pickchunksize, :solve,
        :solve!, :step!,
    )
)

run_qa(
    BoundaryValueDiffEqFIRK;
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
