# [BoundaryValueDiffEqMIRKN](@id mirkn)

Monotonic Implicit Runge Kutta Nyström(MIRKN) Methods. To only use the MIRKN methods form BoundaryVaueDiffEq.jl, you need to install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqMIRKN")
```

```julia
solve(prob::SecondOrderBVProblem, alg; kwargs...)
solve(prob::TwoPointSecondOrderBVProblem, alg; kwargs...)
```

!!! note "Defect control adaptivity"
    
    MIRKN don't have defect control adaptivity

## Full List of Methods

  - `MIRKN4`: Monotonic Implicit Runge-Kutta-Nyström methods with stage order of 4, without error control adaptivity.
  - `MIRKN6`: Monotonic Implicit Runge-Kutta-Nyström methods with stage order of 6, without error control adaptivity.

## Detailed Solvers Explanation

```@docs
MIRKN4
MIRKN6
```
