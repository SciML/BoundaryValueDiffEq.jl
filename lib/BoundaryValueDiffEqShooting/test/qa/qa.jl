using SciMLTesting
using BoundaryValueDiffEqShooting
using DiffEqGPU
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
            # The optional extension calls and specializes its parent package
            # internals; these hooks are not part of the user-facing API.
            ignore = (
                :Dual, :Partials, :Tag, :partials,
                :__shooting_host, :__shooting_integrate!, :__shooting_odecache,
                :__shooting_validate_ode,
            ),
        ),
    ),
    reexports_allow = SHOOTING_REEXPORTS,
)

test_reexport_surface(BoundaryValueDiffEqShooting, SHOOTING_REEXPORTS, @__MODULE__)

@test Base.get_extension(BoundaryValueDiffEqShooting, :BoundaryValueDiffEqShootingDiffEqGPUExt) !== nothing
