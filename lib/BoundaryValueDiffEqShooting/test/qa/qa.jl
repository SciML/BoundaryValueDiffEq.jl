using SciMLTesting
using BoundaryValueDiffEqShooting
using Test

include(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "reexports.jl"))

run_qa(
    BoundaryValueDiffEqShooting;
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            ignore = (:overloaded_input_type, :pickchunksize),
        ),
    ),
    reexports_allow = SHOOTING_REEXPORTS,
)

test_reexport_surface(BoundaryValueDiffEqShooting, SHOOTING_REEXPORTS, @__MODULE__)
