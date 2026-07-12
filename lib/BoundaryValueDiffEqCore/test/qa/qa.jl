using SciMLTesting
using BoundaryValueDiffEqCore
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
    BoundaryValueDiffEqCore,
    (:NonlinearSolveFirstOrder, :SciMLBase, :SciMLOperators),
    (:AllObserved, :deleteat!, :init, :pickchunksize, :solve, :solve!, :step!)
)

run_qa(
    BoundaryValueDiffEqCore;
    explicit_imports = true,
    aqua_kwargs = (;
        ambiguities = (; recursive = false),
        stale_deps = (; ignore = [:TimerOutputs]),
    ),
    ei_kwargs = (;
        # All remaining entries are external internals with no public replacement:
        #   - SciMLBase: BVP problem/algorithm abstract types that BVP solvers
        #     legitimately subtype/extend, and solution_new_original_retcode (no
        #     public counterpart; solution_new_retcode does not preserve the
        #     original retcode).
        #   - ForwardDiff Dual/value/can_dual/pickchunksize: not marked public.
        #   - ArrayInterface.parameterless_type: not marked public.
        #   - SciMLStructures Tunable/canonicalize/isscimlstructure/replace: the
        #     SciMLStructures interface is not marked public.
        #   - SparseConnectivityTracer Dual/primal: internal tracer types.
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractBVProblem, :StandardBVProblem, :StandardSecondOrderBVProblem,
                :parameterless_type, :pickchunksize,
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractBVPAlgorithm, :AbstractBVProblem, :solution_new_original_retcode,
                :Dual, :value, :can_dual, :primal,
                :Tunable, :canonicalize, :isscimlstructure, :replace,
            ),
        ),
    ),
    api_docs_kwargs = (;
        rendered = true,
        docs_src = DOCS_SRC,
        ignore = UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP,
        rendered_ignore = UPSTREAM_REEXPORTS_WITH_DOC_OWNERSHIP,
    ),
)
