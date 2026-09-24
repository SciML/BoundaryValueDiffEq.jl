# [BoundaryValueDiffEqAscher](@id ascher)

Gauss–Legendre collocation methods with Ascher's error control and mesh refinement
for boundary value ODEs and DAEs. Install and load the solver subpackage with:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqAscher")
using BoundaryValueDiffEqAscher
```

## Solver API

```julia
Ascher3(;
    nlsolve = nothing, optimize = nothing, zeta = Float64[],
    jac_alg = BVPJacobianAlgorithm(), platform = CPU(), device = false,
    max_num_subintervals = 3000
)
solve(prob::BVProblem, alg; dt, kwargs...)
solve(prob::TwoPointBVProblem, alg; dt, kwargs...)
```

All Ascher constructors accept the keywords shown for `Ascher3`.

| Keyword | Meaning |
|:--|:--|
| `nlsolve` | Nonlinear solver; `nothing` selects the package default |
| `optimize` | Optional optimization solver for the CPU workflow |
| `zeta` | Locations of the boundary conditions; the device path can infer them for two-point problems |
| `jac_alg` | BVP Jacobian configuration, which takes precedence over the nonlinear solver's autodiff setting |
| `platform` | KernelAbstractions backend; defaults to `CPU()` |
| `device` | Force the packed sparse formulation, including on the CPU |
| `max_num_subintervals` | Maximum number of mesh subintervals |

Pass mesh and tolerance options to `solve`, for example
`solve(prob, Ascher3(; zeta = [0.0, 1.0]); dt = 0.05, abstol = 1.0e-8)`
for a problem with one boundary condition at each endpoint of `(0.0, 1.0)`.
Mesh adaptivity is enabled by default; use `adaptive = false` for a fixed mesh.

For a standard Ascher `BVProblem`, `bc!(r, u, p, t)` receives the local state `u`:
component `r[i]` is evaluated at `zeta[i]`. Supply one location per differential
variable, including repeated locations when several conditions share a point.
For `TwoPointBVProblem`, provide endpoint callbacks and
`bcresid_prototype = (left, right)`; the device path derives the locations from
their sizes, while the ordinary CPU path also requires `zeta`. See
[Common Solver Options](@ref solver_options),
[Error Control Adaptivity](@ref error_control), and
[Reexported API](@ref reexports).

## GPU execution

A device initial state or a GPU `platform` selects the resident sparse formulation.
`device = true` also makes this formulation available on `CPU()` for comparison.
With CUDA, load `CUDA`; loading `CUDSS` enables sparse direct Newton solves.

The device path requires vector `Float32`/`Float64` states, GPU-compatible RHS and
boundary functions, and a square unconstrained BVP. It supports `AutoForwardDiff`
and forward/central `AutoFiniteDiff`, optionally wrapped in `AutoSparse`.
Optimization, least-squares systems and parameter tuning are unsupported.
Use `GlobalErrorControl()` for adaptive mesh refinement, or `adaptive = false`
with `NoErrorControl()` for a fixed mesh. See
[Solving Boundary Value Problems on GPUs](@ref gpu) for endpoint and multipoint
examples.

## Full List of Methods

  - `Ascher1`: 1 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher2`: 2 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher3`: 3 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher4`: 4 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher5`: 5 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher6`: 6 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.
  - `Ascher7`: 7 stage Gauss Legendre collocation method with Ascher's error control adaptivity and mesh refinement.

## Detailed Solvers Explanation

```@docs
Ascher1
Ascher2
Ascher3
Ascher4
Ascher5
Ascher6
Ascher7
```
