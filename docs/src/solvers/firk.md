# [BoundaryValueDiffEqFIRK](@id firk)

Fully Implicit Runge–Kutta (FIRK) methods for first-order boundary value problems.
Install and load the solver subpackage with:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqFIRK")
using BoundaryValueDiffEqFIRK
```

## Solver API

```julia
RadauIIa3(;
    nlsolve = nothing, optimize = nothing, jac_alg = BVPJacobianAlgorithm(),
    platform = CPU(), nested_nlsolve = false, nested_nlsolve_kwargs = (;),
    defect_threshold = 0.1, max_num_subintervals = 3000
)
solve(prob::BVProblem, alg; dt, kwargs...)
solve(prob::TwoPointBVProblem, alg; dt, kwargs...)
```

All Radau and Lobatto constructors accept the keywords shown for `RadauIIa3`.

| Keyword | Meaning |
|:--|:--|
| `nlsolve` | Nonlinear solver; `nothing` selects the package default |
| `optimize` | Optional optimization solver; load its package before use |
| `jac_alg` | BVP Jacobian configuration, which takes precedence over the nonlinear solver's autodiff setting |
| `platform` | KernelAbstractions backend; defaults to `CPU()` |
| `nested_nlsolve` | Solve implicit stages locally instead of including them in the global residual; defaults to `false` |
| `nested_nlsolve_kwargs` | Options for the nested nonlinear solve |
| `defect_threshold` | Threshold used by defect control |
| `max_num_subintervals` | Maximum number of mesh subintervals |

Pass mesh and tolerance options to `solve`, for example
`solve(prob, RadauIIa3(); dt = 0.05, abstol = 1.0e-8)`. Mesh adaptivity is enabled
by default for methods that support it; use `adaptive = false` for a fixed mesh.
See [Common Solver Options](@ref solver_options),
[Error Control Adaptivity](@ref error_control), and [Reexported API](@ref reexports).

### Nested nonlinear solving

Set `nested_nlsolve = true` to solve the implicit Runge–Kutta stages in local
nonlinear systems. The default, `nested_nlsolve = false`, includes those stages
in the global collocation residual.

Configure the nested solve with `nested_nlsolve_kwargs`, for example,
`RadauIIa5(; nested_nlsolve = true, nested_nlsolve_kwargs = (; abstol = 1e-6, reltol = 1e-6))`.
The CPU nested solver accepts
[NonlinearSolve options](https://docs.sciml.ai/NonlinearSolve/stable/basics/solve/).
On the resident device path, stages use batched Newton iterations with backtracking
and pivoted local LU; only `abstol`, `reltol` and `maxiters` are supported, and stage
sensitivities are computed by implicit differentiation.

## GPU execution

A device initial guess such as a `CuArray` selects a resident solve and supplies
the backend. Both expanded (`nested_nlsolve = false`) and nested formulations are
supported. A CPU initial guess with `platform = CUDA.CUDABackend()` selects hybrid
collocation with a CPU nonlinear solve. Load CUDA and CUDSS for the resident
sparse direct-solve workflow, and use GPU-compatible RHS and boundary functions.

Resident Jacobians support `AutoForwardDiff` and forward/central `AutoFiniteDiff`,
optionally wrapped in `AutoSparse`. Square and least-squares systems are supported;
optimization solvers and optimization constraints are not. `RadauIIa1`,
`LobattoIIIb2` and `LobattoIIIc2` require `adaptive = false` or `NoErrorControl()`.
Problems with algebraic variables also require a fixed mesh. See
[Solving Boundary Value Problems on GPUs](@ref gpu) for complete examples.

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

### Example

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
