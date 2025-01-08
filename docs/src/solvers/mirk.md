# [BoundaryValueDiffEqMIRK.jl](@id mirk)

Monotonic Implicit Runge Kutta(MIRK) Methods. To only use the MIRK methods form BoundaryVaueDiffEq.jl, you need to install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqFIRK")
```

```julia
solve(prob::BVProblem, alg; kwargs...)
solve(prob::TwoPointBVProblem, alg; kwargs...)
```

## Full List of Methods

  - `MIRK2`: Monotonic Implicit Runge-Kutta methods with stage order of 2, with defect control adaptivity.
  - `MIRK3`: Monotonic Implicit Runge-Kutta methods with stage order of 3, with defect control adaptivity.
  - `MIRK4`: Monotonic Implicit Runge-Kutta methods with stage order of 4, with defect control adaptivity.
  - `MIRK5`: Monotonic Implicit Runge-Kutta methods with stage order of 5, with defect control adaptivity.
  - `MIRK6`: Monotonic Implicit Runge-Kutta methods with stage order of 6, with defect control adaptivity.

## Detailed Solvers Explanation

```@docs
MIRK2
MIRK3
MIRK4
MIRK5
MIRK6
```
