using SciMLTesting
using BoundaryValueDiffEqCore
using Test

run_qa(
    BoundaryValueDiffEqCore;
    explicit_imports = true,
    aqua_kwargs = (;
        ambiguities = (; recursive = false),
        stale_deps = (; ignore = [:TimerOutputs]),
    ),
    # Pre-existing ExplicitImports findings, tracked in SciML/BoundaryValueDiffEq.jl#519.
    ei_broken = (
        :no_implicit_imports, :no_stale_explicit_imports,
        :all_qualified_accesses_are_public, :all_explicit_imports_are_public,
    ),
)
