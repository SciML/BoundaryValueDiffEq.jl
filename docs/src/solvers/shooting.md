# [BoundaryValueDiffEqShooting](@id shooting)

Single shooting method and multiple shooting method. To only use the Shooting methods form BoundaryVaueDiffEq.jl, you need to install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqShooting")
```

!!! note "Require OrdinaryDiffEq"
    
    Shooting methods require OrdinaryDiffEq.jl loaded to use the ODE solvers

```julia
solve(prob::BVProblem, alg; kwargs...)
solve(prob::TwoPointBVProblem, alg; kwargs...)
```

Shooting methods should be use together with ODE solvers:

```
BoundaryValueDiffEqShooting.Shooting
BoundaryValueDiffEqShooting.MultipleShooting
```

## Full List of Methods

  - `Shooting`: Single shooting methods, reduces BVP to an initial value problem and solves the IVP.
  - `MultipleShooting`: Reduces BVP to an initial value problem and solves the IVP. Significantly more stable than Single Shooting.

## Detailed Solvers Explanation

```@docs
Shooting
MultipleShooting
```
