using SciMLTesting
using BoundaryValueDiffEq
using SciMLBase
using JET
using Test

# ExplicitImports only analyzes an extension module once it has been materialized,
# which requires its trigger weakdep to be loaded. Without this, the QA checks
# silently skip BoundaryValueDiffEqODEInterfaceExt entirely.
using ODEInterface

include("reexports.jl")

run_qa(
    BoundaryValueDiffEq;
    aqua_kwargs = (;
        piracies = (; treat_as_own = [SciMLBase.BVProblem]),
    ),
    reexports_allow = ROOT_REEXPORTS,
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                # ForwardDiff declares no `public` names at all (it exports only
                # `DiffResults`), so its documented in-place AD entry point has no
                # public spelling to switch to.
                :jacobian!,
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                # ODEInterface has no `export` statements and no `public`
                # declarations anywhere; its entire documented API -- solver
                # entry points, option keys, and solution accessors -- is
                # non-public by ExplicitImports' definition. There is no public
                # spelling for any of these names.
                :Bvpm2,
                :OPT_ADDGRIDPOINTS,
                :OPT_BVPCLASS,
                :OPT_COLLOCATIONPTS,
                :OPT_DIAGNOSTICOUTPUT,
                :OPT_ERRORCONTROL,
                :OPT_MAXSTEPS,
                :OPT_MAXSUBINTERVALS,
                :OPT_METHODCHOICE,
                :OPT_RHS_CALLMODE,
                :OPT_RTOL,
                :OPT_SINGULARTERM,
                :OPT_SOLMETHOD,
                :OptionsODE,
                :RHS_CALL_INSITU,
                :bvpm2_destroy,
                :bvpm2_get_x,
                :bvpm2_init,
                :bvpm2_solve,
                :bvpsol,
                :colnew,
                :evalSolution,
            ),
        ),
    ),
)

test_reexport_surface(BoundaryValueDiffEq, ROOT_REEXPORTS, @__MODULE__)
