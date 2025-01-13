# [BoundaryValueDiffEqFIRK](@id firk)

Fully Implicit Runge Kutta(FIRK) Methods. To be able to access the solvers in BoundaryValueDiffEqFIRK, you must first install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqFIRK")
```

```julia
solve(prob::BVProblem, alg, dt; kwargs...)
solve(prob::TwoPointBVProblem, alg, dt; kwargs...)
```

!!! note "Nested nonlinear solving in FIRK methods"
    
    When encountered with large BVP system, setting `nested_nlsolve` to `true` enables FIRK methods to use nested nonlinear solving for the implicit FIRK step instead of solving as a part of the global residual(when default as `nested_nlsolve=false`),

## Full List of Methods

### Radau IIA methods

  - `RadauIIa1`: 1 stage Radau IIA method, with defect control adaptivity
  - `RadauIIa2`: 2 stage Radau IIA method, with defect control adaptivity.
  - `RadauIIa3`: 3 stage Radau IIA method, with defect control adaptivity.
  - `RadauIIa5`: 5 stage Radau IIA method, with defect control adaptivity.
  - `RadauIIa7`: 7 stage Radau IIA method, with defect control adaptivity.

### Lobatto IIIA methods

  - `LobattoIIIa2`: 2 stage Lobatto IIIa method, with defect control adaptivity.
  - `LobattoIIIa3`: 3 stage Lobatto IIIa method, with defect control adaptivity.
  - `LobattoIIIa4`: 4 stage Lobatto IIIa method, with defect control adaptivity.
  - `LobattoIIIa5`: 5 stage Lobatto IIIa method, with defect control adaptivity.

### Lobatto IIIB methods

  - `LobattoIIIb2`: 2 stage Lobatto IIIb method, with defect control adaptivity.
  - `LobattoIIIb3`: 3 stage Lobatto IIIb method, with defect control adaptivity.
  - `LobattoIIIb4`: 4 stage Lobatto IIIb method, with defect control adaptivity.
  - `LobattoIIIb5`: 5 stage Lobatto IIIb method, with defect control adaptivity.

### Lobatto IIIC methods

  - `LobattoIIIc2`: 2 stage Lobatto IIIc method, with defect control adaptivity.
  - `LobattoIIIc3`: 3 stage Lobatto IIIc method, with defect control adaptivity.
  - `LobattoIIIc4`: 4 stage Lobatto IIIc method, with defect control adaptivity.
  - `LobattoIIIc5`: 5 stage Lobatto IIIc method, with defect control adaptivity.

## Detailed Solvers Explanation

```@docs
RadauIIa1
RadauIIa2
RadauIIa3
RadauIIa5
RadauIIa7
```

```@docs
LobattoIIIa2
LobattoIIIa3
LobattoIIIa4
LobattoIIIa5
```

```@docs
LobattoIIIb2
LobattoIIIb3
LobattoIIIb4
LobattoIIIb5
```

```@docs
LobattoIIIc2
LobattoIIIc3
LobattoIIIc4
LobattoIIIc5
```
