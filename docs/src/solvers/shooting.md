# [BoundaryValueDiffEqShooting](@id shooting)

Single shooting method and multiple shooting method. To only use the Shooting methods form BoundaryValueDiffEq.jl, you need to install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqShooting")
```

Shooting algorithms operate on problem definitions and solver functions owned by SciMLBase,
and require an ODE algorithm from its owning solver package:

```julia
using BoundaryValueDiffEqShooting: MultipleShooting, Shooting
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: BVProblem, TwoPointBVProblem, solve

# `prob` is a BVProblem or TwoPointBVProblem defined with SciMLBase.
sol = solve(prob, Shooting(Tsit5()))
```

## Full List of Methods

  - `Shooting`: Single shooting methods, reduces BVP to an initial value problem and solves the IVP.
  - `MultipleShooting`: Reduces BVP to an initial value problem and solves the IVP. Significantly more stable than Single Shooting.

## Detailed Solvers Explanation

```@docs
Shooting
MultipleShooting
```
