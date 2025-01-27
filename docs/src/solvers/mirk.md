# [BoundaryValueDiffEqMIRK](@id mirk)

Monotonic Implicit Runge Kutta(MIRK) Methods. To only use the MIRK methods form BoundaryVaueDiffEq.jl, you need to install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqFIRK")
```

```julia
solve(prob::BVProblem, alg, dt; kwargs...)
solve(prob::TwoPointBVProblem, alg, dt; kwargs...)
```

## Full List of Methods

  - `MIRK2`: 2 stage Monotonic Implicit Runge-Kutta method, with defect control adaptivity.
  - `MIRK3`: 3 stage Monotonic Implicit Runge-Kutta method, with defect control adaptivity.
  - `MIRK4`: 4 stage Monotonic Implicit Runge-Kutta method, with defect control adaptivity.
  - `MIRK5`: 5 stage Monotonic Implicit Runge-Kutta method, with defect control adaptivity.
  - `MIRK6`: 6 stage Monotonic Implicit Runge-Kutta method, with defect control adaptivity.

## Detailed Solvers Explanation

```@docs
MIRK2
MIRK3
MIRK4
MIRK5
MIRK6
```
