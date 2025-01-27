# [Wrapper Methods](@id wrapper)

!!! note "Require ODEInterface"
    
    Wrapper methods require ODEInterface.jl loaded

```julia
solve(prob::BVProblem, alg; kwargs...)
solve(prob::TwoPointBVProblem, alg; kwargs...)
```

## Full List of Methods

  - BVPM2: Single shooting methods, reduces BVP to an initial value problem and solves the IVP.
  - BVPSOL: Reduces BVP to an initial value problem and solves the IVP. Significantly more stable than Single Shooting.
  - COLNEW: Gauss-Legendre collocation methods for BVP with Ascher's error control adaptivity and mesh refinement.
