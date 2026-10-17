# [Wrapper Methods](@id wrapper)

These algorithms wrap the CPU Fortran solvers supplied by ODEInterface.jl. Load
both packages to activate the extension:

```julia
using Pkg
Pkg.add(["BoundaryValueDiffEq", "ODEInterface"])
using BoundaryValueDiffEq, ODEInterface
```

## Solver API

```julia
BVPM2(;
    max_num_subintervals = 3000, method_choice = 4, diagnostic_output = -1,
    error_control = 1, singular_term = nothing
)
BVPSOL(; bvpclass = 2, sol_method = 0, odesolver = nothing)
COLNEW(;
    bvpclass = 1, collocationpts = 7, diagnostic_output = 1,
    max_num_subintervals = 3000, bc_func = nothing, dbc_func = nothing, zeta = nothing
)

solve(prob::TwoPointBVProblem, alg::BVPM2; dt, reltol = 1.0e-3, kwargs...)
solve(prob::TwoPointBVProblem, alg::BVPSOL; dt, reltol = 1.0e-3, maxiters = 1000, kwargs...)
solve(prob::BVProblem, alg::COLNEW; dt, reltol = 1.0e-3, maxiters = 1000, kwargs...)
```

The wrapper constructors accept different keywords, as shown above.

| Keyword | Meaning |
|:--|:--|
| `max_num_subintervals` | Maximum number of mesh subintervals for `BVPM2` and `COLNEW` |
| `method_choice` | Runge–Kutta order for `BVPM2`: 2, 4 or 6 |
| `diagnostic_output` | Diagnostic output level for `BVPM2` and `COLNEW` |
| `error_control` | Error-control strategy for `BVPM2` |
| `singular_term` | Optional constant matrix for a singular term in `BVPM2` |
| `bvpclass` | Problem classification for `BVPSOL` or `COLNEW`; meanings depend on the solver |
| `sol_method` | Local condensing or global sparse linear solver for `BVPSOL` |
| `odesolver` | Optional internal ODE solver for `BVPSOL` |
| `collocationpts` | Number of collocation points per subinterval for `COLNEW` |
| `bc_func`, `dbc_func` | Multipoint boundary function and its Jacobian for `COLNEW` |
| `zeta` | Multipoint boundary-condition locations for `COLNEW` |

Pass mesh and tolerance options to `solve`, for example
`solve(prob, BVPM2(); dt = 0.05, reltol = 1.0e-6)` for a two-point problem.

`BVPM2` and `BVPSOL` require a `TwoPointBVProblem`; `BVPSOL` additionally requires
an initial guess over the mesh. `COLNEW` also supports multipoint boundary
conditions through `bc_func`, `dbc_func` and `zeta`. Use a positive `dt` to
specify the mesh spacing; `BVPM2` and `BVPSOL` can infer a mesh from an initial
guess containing mesh values.

The wrapper algorithms do not accept the native solvers' `jac_alg` option. See
the [ODEInterface API](@ref ODEInterface.jl) for solver-specific option values.

## GPU execution

The wrappers run CPU Fortran solvers and do not expose `platform` or
`device_steps`. For GPU execution, use the native solver packages described in
[Solving Boundary Value Problems on GPUs](@ref gpu).

## Full List of Methods

  - `BVPM2`: Runge–Kutta boundary value solver with selectable method order and error control.
  - `BVPSOL`: Multiple-shooting boundary value solver.
  - `COLNEW`: Collocation solver for mixed-order systems with multipoint boundary conditions.

## Detailed Solvers Explanation

The full constructor references are documented in the
[ODEInterface API](@ref ODEInterface.jl):

  - [`BVPM2`](@ref)
  - [`BVPSOL`](@ref)
  - [`COLNEW`](@ref)
