using SciMLTesting
using BoundaryValueDiffEqShooting
using Test

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
)
