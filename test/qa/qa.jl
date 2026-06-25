using SciMLTesting
using BoundaryValueDiffEq
using SciMLBase
using JET
using Test

run_qa(
    BoundaryValueDiffEq;
    explicit_imports = true,
    aqua_kwargs = (;
        ambiguities = false,
        piracies = (; treat_as_own = [SciMLBase.BVProblem]),
    ),
    # BoundaryValueDiffEq is an umbrella that re-exports the solver algorithms
    # (Shooting, MIRK*, RadauIIa*, LobattoIII*, Ascher*, ...) from its sublibraries
    # via `using <Sublibrary>` + `export`, which ExplicitImports flags as implicit.
    # Tracked in SciML/BoundaryValueDiffEq.jl#519.
    ei_broken = (:no_implicit_imports,),
    ei_kwargs = (;
        # SciMLBase.__init/__solve (extending the solve interface) and
        # Base.get_extension (querying loaded weakdeps) are other-package internals.
        all_qualified_accesses_are_public = (; ignore = (:__init, :__solve, :get_extension)),
    ),
)
