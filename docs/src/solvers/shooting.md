# [BoundaryValueDiffEqShooting](@id shooting)

Single shooting method and multiple shooting method. To only use the Shooting methods form BoundaryValueDiffEq.jl, you need to install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqShooting")
```

Shooting algorithms operate on problem definitions and solver functions owned by SciMLBase,
and require an ODE algorithm from its owning solver package:

```jldoctest
using BoundaryValueDiffEqShooting: MultipleShooting, Shooting
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: BVProblem, ReturnCode, solve

function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = 0
end

function bc!(residual, u, p, t)
    residual[1] = u(0.0)[1] - 1
    residual[2] = u(1.0)[1]
end

prob = BVProblem(f!, bc!, [1.0, -1.0], (0.0, 1.0); nlls = Val(false))
sol = solve(prob, Shooting(Tsit5()); abstol = 1e-8)

@assert sol.retcode == ReturnCode.Success
@assert isapprox(sol(0.0)[1], 1.0; atol = 1e-6)
@assert isapprox(sol(1.0)[1], 0.0; atol = 1e-6)
# output
```

## Full List of Methods

  - `Shooting`: Single shooting methods, reduces BVP to an initial value problem and solves the IVP.
  - `MultipleShooting`: Reduces BVP to an initial value problem and solves the IVP. Significantly more stable than Single Shooting.

## Detailed Solvers Explanation

```@docs
Shooting
MultipleShooting
```
