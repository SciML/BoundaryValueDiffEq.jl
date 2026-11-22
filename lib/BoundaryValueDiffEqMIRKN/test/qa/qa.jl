using SciMLTesting
using BoundaryValueDiffEqMIRKN
using Test

include(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "reexports.jl"))

run_qa(
    BoundaryValueDiffEqMIRKN;
    ei_kwargs = (;
        # External internals with no public replacement:
        #   - StandardSecondOrderBVProblem: SciMLBase-owned problem type, not public.
        #   - pickchunksize: ForwardDiff internal.
        all_explicit_imports_are_public = (;
            ignore = (:StandardSecondOrderBVProblem, :pickchunksize),
        ),
    ),
    reexports_allow = MIRKN_REEXPORTS,
)

test_reexport_surface(BoundaryValueDiffEqMIRKN, MIRKN_REEXPORTS, @__MODULE__)
