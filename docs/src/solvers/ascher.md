# [BoundaryValueDiffEqAscher.jl](@id ascher)

Gauss Legendre collocation methods with Ascher's error control adaptivity and mesh refinement routines. To be able to access the solvers in BoundaryValueDiffEqFIRK, you must first install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqAscher")
```

```julia
solve(prob::BVProblem, alg; kwargs...)
solve(prob::TwoPointBVProblem, alg; kwargs...)
```

## Full List of Methods

  - `Ascher1`: 1 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher2`: 1 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher3`: 1 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher4`: 1 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher5`: 1 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher6`: 1 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher7`: 1 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.

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
