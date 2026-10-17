using SciMLTesting
using BoundaryValueDiffEqFIRK
using Test

include(joinpath(@__DIR__, "..", "..", "..", "..", "test", "qa", "reexports.jl"))

run_qa(
    BoundaryValueDiffEqFIRK;
    ei_kwargs = (;
        # External internals with no public replacement:
        #   - StandardBVProblem: SciMLBase-owned problem type, not public.
        #   - pickchunksize: ForwardDiff internal.
        all_explicit_imports_are_public = (;
            ignore = (:StandardBVProblem, :pickchunksize),
        ),
        # SciMLStructures interface (Tunable/canonicalize/isscimlstructure) is not
        # marked public. Nested-stage sensitivities also use ForwardDiff's
        # Dual representation, whose accessors are not marked public upstream.
        all_qualified_accesses_are_public = (;
            ignore = (:Tunable, :canonicalize, :isscimlstructure, :Dual, :Partials, :partials, :value),
        ),
    ),
    reexports_allow = FIRK_REEXPORTS,
)

test_reexport_surface(BoundaryValueDiffEqFIRK, FIRK_REEXPORTS, @__MODULE__)
