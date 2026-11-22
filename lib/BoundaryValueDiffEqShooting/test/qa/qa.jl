using SciMLTesting
using BoundaryValueDiffEqShooting
using DiffEqGPU
using Test

include(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "reexports.jl"))

run_qa(
    BoundaryValueDiffEqShooting;
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            # ForwardDiff's nested dual construction uses non-public names.
            ignore = (:overloaded_input_type, :pickchunksize, :Dual, :Partials, :value, :partials),
        ),
        all_qualified_accesses_are_public = (;
            # The extension specializes its parent package's internal hooks.
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
