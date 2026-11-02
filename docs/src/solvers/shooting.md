# [BoundaryValueDiffEqShooting](@id shooting)

Single and multiple shooting methods reduce a BVP to initial value problems and a
nonlinear matching problem. Install and load the solver subpackage with:

```julia
using Pkg
Pkg.add("BoundaryValueDiffEqShooting")
using BoundaryValueDiffEqShooting
```

## Solver API

```julia
Shooting(; ode_alg = nothing, nlsolve = nothing, optimize = nothing, jac_alg = nothing)
Shooting(ode_alg; kwargs...)
Shooting(ode_alg, nlsolve; kwargs...)

MultipleShooting(;
    nshoots::Int, ode_alg = nothing, nlsolve = nothing, optimize = nothing,
    jac_alg = nothing, platform = CPU(), device_steps = nothing,
    device_linsolve = nothing, grid_coarsening = (device_steps === nothing)
)
MultipleShooting(nshoots; kwargs...)
MultipleShooting(nshoots, ode_alg; kwargs...)
MultipleShooting(nshoots, ode_alg, nlsolve; kwargs...)

solve(prob::BVProblem, alg; kwargs...)
solve(prob::TwoPointBVProblem, alg; kwargs...)
```

`Shooting` and `MultipleShooting` share the ODE, nonlinear solver, optimization
and Jacobian options. The remaining keywords apply to `MultipleShooting` only.

| Keyword | Meaning |
|:--|:--|
| `ode_alg` | Internal ODE solver; import the algorithm from its owning solver package |
| `nlsolve` | Nonlinear solver; `nothing` selects the package default |
| `optimize` | Optional optimization solver for the ordinary CPU workflow; load its package before use |
| `jac_alg` | BVP Jacobian configuration, which takes precedence over the nonlinear solver's autodiff setting |
| `nshoots` | Number of shooting intervals; must be positive |
| `platform` | KernelAbstractions backend for fixed-step multiple shooting; defaults to `CPU()` |
| `device_steps` | Fixed ODE steps per interval; `nothing` selects the ordinary CPU integration path |
| `device_linsolve` | Optional linear solver for the default fixed-step Newton solve on square problems; cannot be combined with `nlsolve` |
| `grid_coarsening` | Grid-coarsening strategy; defaults to `true` on the ordinary CPU path and must be `false` when `device_steps` is set |

Pass tolerance options to `solve`, for example
`solve(prob, Shooting(Tsit5()); abstol = 1.0e-8)` after importing `Tsit5` from
`OrdinaryDiffEqTsit5`. Single shooting uses `jac_alg.diffmode`; multiple shooting
uses `diffmode` for two-point problems and `bc_diffmode` / `nonbc_diffmode` for
general problems. See [Common Solver Options](@ref solver_options) and
[Reexported API](@ref reexports).

The ordinary CPU multiple-shooting path accepts `EnsembleSerial()` or
`EnsembleThreads()` through the `ensemblealg` **solve** keyword. Its
`grid_coarsening` constructor keyword accepts a Boolean, an integer vector or
tuple, or a function selecting successive grid sizes. Use `odesolve_kwargs` and
`nlsolve_kwargs` to configure the internal CPU ODE and nonlinear solves.

## GPU execution

Set `device_steps` to a positive integer to use fixed-step interval integrations.
For CUDA, load `CUDA` and `DiffEqGPU`, pass a kernel algorithm such as
`DiffEqGPU.GPUTsit5()` or `DiffEqGPU.GPUVern7()`, and select
`platform = CUDA.CUDABackend()`. A CPU initial guess is accepted and transferred
to the selected backend. An OrdinaryDiffEq algorithm such as `Tsit5()` is only
accepted with `CPU()` on this path.

`device_steps` controls integration resolution on each of the `nshoots` intervals.
Keep `grid_coarsening = false`, its default when `device_steps` is set. Configure
nonlinear convergence with the `abstol` and `nlsolve_kwargs` solve keywords.

Both in-place and out-of-place RHS functions are supported. States must be vectors
of `Float32` or `Float64`, and the RHS and boundary functions must compile for the
GPU. Jacobians use `AutoForwardDiff` or forward/central `AutoFiniteDiff`, optionally
wrapped in `AutoSparse`, and CUDA stores them in sparse CSR format. Loading `CUDSS`
enables cached direct solves for square CUDA problems, with segment condensation
for two-point boundaries. Without CUDSS the default uses GMRES; least-squares
problems use LSMR.

There is no final single-shooting solve on this path. The result uses cubic Hermite
interpolation between shooting nodes. `abstol` does not control ODE discretization
error: increase `device_steps` or `nshoots` to check accuracy. `odesolve_kwargs`,
callbacks, nonidentity mass matrices, optimization constraints and parameter
tuning are unsupported. See [Solving Boundary Value Problems on GPUs](@ref gpu)
for a complete CUDA example.

`Shooting` has no `platform` or `device_steps` keyword; the dedicated GPU interval
workflow belongs to `MultipleShooting`.

## Full List of Methods

  - `Shooting`: solves one IVP and adjusts its initial condition to satisfy the boundary conditions.
  - `MultipleShooting`: solves IVPs on multiple intervals and enforces continuity and the boundary conditions; generally more stable than single shooting.

## Detailed Solvers Explanation

```@docs
Shooting
MultipleShooting
```

### Example

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
