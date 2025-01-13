# [BoundaryValueDiffEqMIRKN](@id mirkn)

Monotonic Implicit Runge Kutta Nyström(MIRKN) Methods. To only use the MIRKN methods form BoundaryVaueDiffEq.jl, you need to install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqMIRKN")
```

```julia
solve(prob::SecondOrderBVProblem, alg, dt; kwargs...)
solve(prob::TwoPointSecondOrderBVProblem, alg, dt; kwargs...)
```

!!! note "Defect control adaptivity"
    
    MIRKN don't have defect control adaptivity

## Full List of Methods

  - `MIRKN4`: 4 stage Monotonic Implicit Runge-Kutta-Nyström method, with no error control adaptivity.
  - `MIRKN6`: 4 stage Monotonic Implicit Runge-Kutta-Nyström method, with no error control adaptivity.

## Detailed Solvers Explanation

```@docs
MIRKN4
MIRKN6
```
