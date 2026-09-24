# [BoundaryValueDiffEqMIRKN](@id mirkn)

Monotonic Implicit Runge–Kutta–Nyström (MIRKN) methods solve second-order boundary
value problems directly. Install and load the solver subpackage with:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqMIRKN")
using BoundaryValueDiffEqMIRKN
```

## Solver API

```julia
MIRKN4(;
    nlsolve = nothing, optimize = nothing, jac_alg = BVPJacobianAlgorithm(),
    platform = CPU(), defect_threshold = 0.1, max_num_subintervals = 3000
)
solve(prob::SecondOrderBVProblem, alg; dt, kwargs...)
solve(prob::TwoPointSecondOrderBVProblem, alg; dt, kwargs...)
```

All MIRKN constructors accept the keywords shown for `MIRKN4`.

| Keyword | Meaning |
|:--|:--|
| `nlsolve` | Nonlinear solver; `nothing` selects the package default |
| `optimize` | Optional optimization solver for the CPU or hybrid workflow; load its package before use |
| `jac_alg` | BVP Jacobian configuration, which takes precedence over the nonlinear solver's autodiff setting |
| `platform` | KernelAbstractions backend; defaults to `CPU()` |
| `defect_threshold` | Defect-control threshold; unused by the fixed-mesh solver |
| `max_num_subintervals` | Mesh-size limit; unused by the fixed-mesh solver |

Pass mesh and tolerance options to `solve`, for example
`solve(prob, MIRKN4(); dt = 0.05, abstol = 1.0e-8)`. MIRKN uses a fixed mesh.
For two-point problems, `jac_alg` uses `diffmode`; otherwise it uses
`bc_diffmode` and `nonbc_diffmode`. See [Common Solver Options](@ref solver_options),
[Error Control Adaptivity](@ref error_control), and [Reexported API](@ref reexports).

The in-place RHS signature is `f!(ddu, du, u, p, t)`. General boundary functions
have signature `bc!(r, du, u, p, t)`; endpoint functions for
`TwoPointSecondOrderBVProblem` have signature `bc!(r, du, u, p)`. Supply
`bcresid_prototype = (left, right)` for device two-point problems.

!!! note "Fixed mesh"

    MIRKN has no mesh adaptivity. Use a positive `dt`, `adaptive = false` and
    `NoErrorControl()` (the defaults for the latter two). The constructor fields
    `defect_threshold` and `max_num_subintervals` do not enable adaptive refinement.
    Refine the mesh manually to check accuracy. Solution interpolation is linear
    between mesh points.

## GPU execution

A device initial guess selects a resident solve and supplies its backend. Load
CUDA and CUDSS for CUDA sparse solves. A CPU initial guess with
`platform = CUDA.CUDABackend()` instead selects hybrid GPU collocation with a CPU
nonlinear solve. Both RHS and boundary functions must be GPU-compatible for the
resident path.

Resident MIRKN accepts `Float32`/`Float64` states, forward-mode AD or
forward/central finite differences, and square or least-squares BVPs. It requires
a fixed mesh and does not support optimization constraints or parameter tuning.
See [Solving Boundary Value Problems on GPUs](@ref gpu) for a complete example and
how to access the position and velocity in the solution.

## Full List of Methods

  - `MIRKN4`: fourth-order MIRKN method, without mesh adaptivity.
  - `MIRKN6`: sixth-order MIRKN method, without mesh adaptivity.

## Detailed Solvers Explanation

```@docs
MIRKN4
MIRKN6
```
