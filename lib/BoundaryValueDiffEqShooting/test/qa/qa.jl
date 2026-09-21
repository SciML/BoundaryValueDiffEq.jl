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
        # Device AD seeds/extracts ForwardDiff's documented
        # Dual/Partials representation directly; those names are not marked public.
        all_qualified_accesses_are_public = (;
            ignore = (:Dual, :Partials, :Tag, :partials),
        ),
    ),
    reexports_allow = SHOOTING_REEXPORTS,
)

test_reexport_surface(BoundaryValueDiffEqShooting, SHOOTING_REEXPORTS, @__MODULE__)
