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
)
