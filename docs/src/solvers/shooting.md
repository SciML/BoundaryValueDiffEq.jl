# [BoundaryValueDiffEqShooting](@id shooting)

Single shooting method and multiple shooting method. To only use the Shooting methods form BoundaryValueDiffEq.jl, you need to install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqShooting")
```

`BoundaryValueDiffEqShooting` reexports the problem constructors, `solve` and
`ReturnCode` its documented workflow uses (see [Reexported API](@ref reexports)); the ODE
algorithm has to come from its own solver package:

```jldoctest
using BoundaryValueDiffEqShooting
using OrdinaryDiffEqTsit5: Tsit5

function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = 0
    return
end

function bc!(residual, u, p, t)
    residual[1] = u(0.0)[1] - 1
    residual[2] = u(1.0)[1]
    return
end

prob = BVProblem(f!, bc!, [1.0, -1.0], (0.0, 1.0); nlls = Val(false))
sol = solve(prob, Shooting(Tsit5()); abstol = 1.0e-8)

@assert sol.retcode == ReturnCode.Success
@assert isapprox(sol(0.0)[1], 1.0; atol = 1.0e-6)
@assert isapprox(sol(1.0)[1], 0.0; atol = 1.0e-6)
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
