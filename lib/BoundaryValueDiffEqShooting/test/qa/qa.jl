using SciMLTesting
using BoundaryValueDiffEqShooting
using Test

run_qa(
    BoundaryValueDiffEqShooting;
    aqua_kwargs = (; persistent_tasks = false),
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            ignore = (:overloaded_input_type, :pickchunksize),
        ),
    ),
)
