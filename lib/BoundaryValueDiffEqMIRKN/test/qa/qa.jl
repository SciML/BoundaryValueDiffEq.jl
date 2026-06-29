using SciMLTesting
using BoundaryValueDiffEqMIRKN
using Test

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
)
