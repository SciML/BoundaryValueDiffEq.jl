# [SimpleBoundaryValueDiffEq](@id simpleboundaryvaluediffeq)

SimpleBoundaryValueDiffEq provides lightweight MIRK and single-shooting methods.
Install and load the solver package with:

```julia
using Pkg
Pkg.add("SimpleBoundaryValueDiffEq")
using SimpleBoundaryValueDiffEq
```

## Solver API

```julia
SimpleMIRK4(; nlsolve = SimpleNewtonRaphson())
SimpleMIRK5(; nlsolve = SimpleNewtonRaphson())
SimpleMIRK6(; nlsolve = SimpleNewtonRaphson())
SimpleShooting(; nlsolve = SimpleNewtonRaphson(), ode_alg = Tsit5())

solve(prob::BVProblem, alg; kwargs...)
solve(prob::TwoPointBVProblem, alg; kwargs...)
```

All simple MIRK constructors accept `nlsolve`; `SimpleShooting` also accepts
`ode_alg`.

| Keyword | Meaning |
|:--|:--|
| `nlsolve` | Nonlinear solver; defaults to `SimpleNewtonRaphson()` |
| `ode_alg` | Internal ODE solver for `SimpleShooting`; defaults to `Tsit5()` |

`SimpleNewtonRaphson()` and `Tsit5()` above describe constructor defaults; callers
can simply use `SimpleMIRK4()` or `SimpleShooting()`. To supply a different
nonlinear or ODE algorithm, import it from its owning solver package.

The simple MIRK methods use a fixed mesh. Pass a positive `dt` to `solve`, for
example `solve(prob, SimpleMIRK4(); dt = 0.05)`. `SimpleShooting` accepts `abstol`,
`reltol`, `odesolve_kwargs` and `nlsolve_kwargs` as solve keywords.

## GPU execution

These constructors do not expose `platform` or `device_steps`. For the GPU
collocation and multiple-shooting APIs, use the corresponding BoundaryValueDiffEq
subpackages described in [Solving Boundary Value Problems on GPUs](@ref gpu).

## Full List of Methods

  - `SimpleMIRK4`: fourth-order MIRK method.
  - `SimpleMIRK5`: fifth-order MIRK method.
  - `SimpleMIRK6`: sixth-order MIRK method.
  - `SimpleShooting`: single shooting with configurable ODE and nonlinear solvers.

## Detailed Solvers Explanation

```@docs
SimpleBoundaryValueDiffEq.SimpleMIRK4
SimpleBoundaryValueDiffEq.SimpleMIRK5
SimpleBoundaryValueDiffEq.SimpleMIRK6
SimpleBoundaryValueDiffEq.SimpleShooting
```
