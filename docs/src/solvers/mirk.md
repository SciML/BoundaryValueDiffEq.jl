# [BoundaryValueDiffEqMIRK](@id mirk)

Monotonic Implicit Runge–Kutta (MIRK) methods for first-order boundary value
problems. Install and load the solver subpackage with:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqMIRK")
using BoundaryValueDiffEqMIRK
```

## Solver API

```julia
MIRK4(;
    nlsolve = nothing, optimize = nothing, jac_alg = BVPJacobianAlgorithm(),
    platform = CPU(), defect_threshold = 0.1, max_num_subintervals = 3000
)
solve(prob::BVProblem, alg; dt, kwargs...)
solve(prob::TwoPointBVProblem, alg; dt, kwargs...)
```

All MIRK constructors accept the keywords shown for `MIRK4`.

| Keyword | Meaning |
|:--|:--|
| `nlsolve` | Nonlinear solver; `nothing` selects the package default |
| `optimize` | Optional optimization solver; load its package before use |
| `jac_alg` | BVP Jacobian configuration, which takes precedence over the nonlinear solver's autodiff setting |
| `platform` | KernelAbstractions backend; defaults to `CPU()` |
| `defect_threshold` | Threshold used by defect control |
| `max_num_subintervals` | Maximum number of mesh subintervals |

Pass mesh and tolerance options to `solve`, for example
`solve(prob, MIRK4(); dt = 0.05, abstol = 1.0e-8)`. Mesh adaptivity is enabled by
default; use `adaptive = false` for a fixed mesh. See [Common Solver Options](@ref
solver_options), [Error Control Adaptivity](@ref error_control), and
[Reexported API](@ref reexports).

## GPU execution

A device initial guess such as a `CuArray` selects a resident solve and supplies
the backend. A CPU initial guess with `platform = CUDA.CUDABackend()` offloads
collocation while keeping the nonlinear solve on the CPU. The resident path
requires GPU-compatible RHS and boundary functions and supports `AutoForwardDiff`
and forward/central `AutoFiniteDiff`, optionally wrapped in `AutoSparse`.

CUDA resident solves use sparse CSR Jacobians. Load CUDA and CUDSS for the sparse
direct-solve workflow. Mesh adaptivity and nonlinear least squares are supported;
problems with algebraic variables require `adaptive = false`. Resident
optimization and optimization constraints are unsupported. See
[Solving Boundary Value Problems on GPUs](@ref gpu) for setup and examples.

## Full List of Methods

  - `MIRK2`: second-order MIRK method, with defect control adaptivity.
  - `MIRK3`: third-order MIRK method, with defect control adaptivity.
  - `MIRK4`: fourth-order MIRK method, with defect control adaptivity.
  - `MIRK5`: fifth-order MIRK method, with defect control adaptivity.
  - `MIRK6`: sixth-order MIRK method, with defect control adaptivity.
  - `MIRK6I`: sixth-order MIRK variant, exported by `BoundaryValueDiffEqMIRK`.

`maxsol` and `minsol` locate extrema of a MIRK solution; see their API below.

## Detailed Solvers Explanation

```@docs
MIRK2
MIRK3
MIRK4
MIRK5
MIRK6
MIRK6I
maxsol
minsol
```
