# [BoundaryValueDiffEqFIRK](@id firk)

Fully Implicit Runge Kutta(FIRK) Methods. To be able to access the solvers in BoundaryValueDiffEqFIRK, you must first install them use the Julia package manager:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqFIRK")
```

`BoundaryValueDiffEqFIRK` reexports the BVP problem constructors and `solve` used by its
documented solver workflow (see [Reexported API](@ref reexports)):

```jldoctest
using BoundaryValueDiffEqFIRK

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
sol = solve(prob, RadauIIa5(); dt = 0.2, abstol = 1.0e-8)

@assert isapprox(sol(0.0)[1], 1.0; atol = 1.0e-6)
@assert isapprox(sol(1.0)[1], 0.0; atol = 1.0e-6)

function bca!(residual, u, p)
    residual[1] = u[1] - 1
    return
end

function bcb!(residual, u, p)
    residual[1] = u[1]
    return
end

two_point_prob = TwoPointBVProblem(
    f!, (bca!, bcb!), [1.0, -1.0], (0.0, 1.0);
    bcresid_prototype = (zeros(1), zeros(1)), nlls = Val(false)
)
two_point_sol = solve(two_point_prob, RadauIIa5(); dt = 0.2, abstol = 1.0e-8)

@assert isapprox(two_point_sol(0.0)[1], 1.0; atol = 1.0e-6)
@assert isapprox(two_point_sol(1.0)[1], 0.0; atol = 1.0e-6)
# output
```

## Nested nonlinear solving in FIRK methods

When working with large boundary value problems, especially those involving stiff systems, computational efficiency and solver robustness become critical concerns. To improve the efficiency of FIRK methods on large BVPs, we can use nested nonlinear solving to obtain the implicit FIRK step instead of solving them as part of the global residual. In BoundaryValueDiffEq.jl, we can set `nested_nlsolve` as `true` to enable FIRK methods to compute the implicit FIRK steps using nested nonlinear solving(default option in FIRK methods is `nested_nlsolve=false`).

Moreover, the nested nonlinear problem solver can be finely tuned to meet specific accuracy requirements by providing detailed keyword arguments through the `nested_nlsolve_kwargs` option in any FIRK solver, for example, `RadauIIa5(; nested_nlsolve = true, nested_nlsolve_kwargs = (; abstol = 1e-6, reltol = 1e-6))`, where `nested_nlsolve_kwargs` can be any common keyword arguments in NonlinearSolve.jl, see [Common Solver Options in NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/basics/solve/).

## Full List of Methods

### Radau IIA methods

  - `RadauIIa1`: 1 stage Radau IIA method, without defect control adaptivity
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

  - `LobattoIIIb2`: 2 stage Lobatto IIIb method, without defect control adaptivity.
  - `LobattoIIIb3`: 3 stage Lobatto IIIb method, with defect control adaptivity.
  - `LobattoIIIb4`: 4 stage Lobatto IIIb method, with defect control adaptivity.
  - `LobattoIIIb5`: 5 stage Lobatto IIIb method, with defect control adaptivity.

### Lobatto IIIC methods

  - `LobattoIIIc2`: 2 stage Lobatto IIIc method, without defect control adaptivity.
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
