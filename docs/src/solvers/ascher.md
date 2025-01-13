# [BoundaryValueDiffEqAscher](@id ascher)

Gauss Legendre collocation methods with Ascher's error control adaptivity and mesh refinement routines. To be able to access the solvers in BoundaryValueDiffEqFIRK, you must first install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqAscher")
```

```julia
solve(prob::BVProblem, alg, dt; kwargs...)
solve(prob::TwoPointBVProblem, alg, dt; kwargs...)
```

## Full List of Methods

  - `Ascher1`: 1 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher2`: 2 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher3`: 3 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher4`: 4 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher5`: 5 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher6`: 6 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher7`: 7 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.

## Detailed Solvers Explanation

```@docs
Ascher1
Ascher2
Ascher3
Ascher4
Ascher5
Ascher6
Ascher7
```
