using SciMLTesting
using BoundaryValueDiffEqAscher
using Test

run_qa(
    BoundaryValueDiffEqAscher;
    explicit_imports = true,
    # Pre-existing ExplicitImports findings, tracked in SciML/BoundaryValueDiffEq.jl#519.
    ei_broken = (
        :no_implicit_imports, :no_stale_explicit_imports,
        :all_qualified_accesses_are_public, :all_explicit_imports_are_public,
    ),
)
