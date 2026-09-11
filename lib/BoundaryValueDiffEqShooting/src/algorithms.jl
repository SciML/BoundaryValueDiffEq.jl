# Algorithms
abstract type AbstractShooting <: AbstractBoundaryValueDiffEqAlgorithm end

"""
    Shooting(ode_alg; kwargs...)
    Shooting(ode_alg, nlsolve; kwargs...)
    Shooting(; ode_alg = nothing, nlsolve = nothing, optimize = nothing, jac_alg = nothing) -> Shooting

Configures the single-shooting algorithm for a boundary value problem. Single shooting
integrates one initial value problem and solves for the initial condition that satisfies the
boundary conditions.

## Arguments

  - `ode_alg`: algorithm used to solve the internal `SciMLBase.ODEProblem`. Pass this as the
    first positional argument or keyword argument. `nothing` selects a loaded polyalgorithm;
    otherwise an ODE algorithm must be supplied.
  - `nlsolve`: nonlinear-solver algorithm for the shooting residual. Its autodiff setting is
    superseded by `jac_alg` when a Jacobian algorithm is materialized.

## Keywords

  - `ode_alg = nothing`: ODE algorithm, as described above.
  - `nlsolve = nothing`: nonlinear-solver algorithm, as described above.
  - `optimize = nothing`: optimization-solver algorithm used when the selected BVP solve path
    formulates the residual as an optimization problem.
  - `jac_alg = nothing`: `BVPJacobianAlgorithm` configuration. When omitted, the constructor
    derives it from `nlsolve` and the problem during solve initialization. For single shooting,
    only its `diffmode` setting is used; the default is `AutoForwardDiff` when applicable and
    otherwise `AutoFiniteDiff`.

## Fields

  - `ode_alg`: configured ODE algorithm or `nothing`.
  - `nlsolve`: configured nonlinear-solver algorithm or `nothing`.
  - `optimize`: configured optimization-solver algorithm or `nothing`.
  - `jac_alg::BVPJacobianAlgorithm`: materialized Jacobian-algorithm configuration.

## Returns

  - `Shooting`: an algorithm object accepted by `SciMLBase.solve` for a boundary value problem.

## Examples

```jldoctest
using BoundaryValueDiffEqShooting: Shooting
using OrdinaryDiffEqTsit5: Tsit5

alg = Shooting(Tsit5())
@assert alg isa Shooting
# output
```
"""
@concrete struct Shooting{J <: BVPJacobianAlgorithm} <: AbstractShooting
    ode_alg
    nlsolve
    optimize
    jac_alg::J
end

function Shooting(; ode_alg = nothing, nlsolve = nothing, optimize = nothing, jac_alg = nothing)
    return Shooting(ode_alg, nlsolve, optimize, __materialize_jacobian_algorithm(nlsolve, jac_alg))
end
@inline Shooting(ode_alg; kwargs...) = Shooting(; ode_alg, kwargs...)
@inline Shooting(ode_alg, nlsolve; kwargs...) = Shooting(; ode_alg, nlsolve, kwargs...)

@inline function concretize_jacobian_algorithm(alg::Shooting, prob)
    alg.jac_alg.diffmode === nothing &&
        (return @set alg.jac_alg.diffmode = __default_nonsparse_ad(prob.u0))
    return alg
end

"""
    MultipleShooting(;
            nshoots::Int, ode_alg = nothing, nlsolve = nothing,
            optimize = nothing, grid_coarsening = true, jac_alg = nothing
        ) -> MultipleShooting
    MultipleShooting(nshoots::Int; kwargs...)
    MultipleShooting(nshoots::Int, ode_alg; kwargs...)
    MultipleShooting(nshoots::Int, ode_alg, nlsolve; kwargs...)

Configures the multiple-shooting algorithm for a boundary value problem. Multiple shooting
integrates an IVP on `nshoots` subintervals and solves for their matching initial conditions;
it is generally more stable than [`Shooting`](@ref).

## Arguments

  - `nshoots::Int`: number of shooting subintervals.
  - `ode_alg`: algorithm used to solve each internal `SciMLBase.ODEProblem`. Pass this as the
    second positional argument or keyword argument. `nothing` selects a loaded polyalgorithm;
    otherwise an ODE algorithm must be supplied.
  - `nlsolve`: nonlinear-solver algorithm for the multiple-shooting residual.

## Keywords

  - `ode_alg = nothing`: ODE algorithm, as described above.
  - `nlsolve = nothing`: nonlinear-solver algorithm, as described above.
  - `optimize = nothing`: optimization-solver algorithm used when the selected BVP solve path
    formulates the residual as an optimization problem.
  - `jac_alg = nothing`: `BVPJacobianAlgorithm` configuration. When omitted, the constructor
    derives it from `nlsolve` and the problem during solve initialization.

      + For `TwoPointBVProblem`, only `diffmode` is used (defaults to
        `AutoSparse(AutoForwardDiff())` if possible else `AutoSparse(AutoFiniteDiff())`).
      + For `BVProblem`, `bc_diffmode` and `nonbc_diffmode` are used. For `nonbc_diffmode`
        we default to `AutoSparse(AutoForwardDiff())` if possible else
        `AutoSparse(AutoFiniteDiff())`. For `bc_diffmode`, we default to `AutoForwardDiff`
        if possible else `AutoFiniteDiff`.
  - `grid_coarsening = true`: coarsens the multiple-shooting grid while generating a stable
    IVP solution. Supported values are:

      + `true`: Halve the grid size, till we reach a grid size of 1.
      + `false`: Do not coarsen the grid. Solve a Multiple Shooting Problem and finally
        solve a Single Shooting Problem.
      + `AbstractVector{<:Int}` or `Ntuple{N, <:Integer}`: Use the provided grid coarsening.
        For example, if `nshoots = 10` and `grid_coarsening = [5, 2]`, then the grid will be
        coarsened to `[5, 2]`. Note that `1` should not be present in the grid coarsening.
      + `Function`: Takes the current number of shooting points and returns the next number
        of shooting points. For example, if `nshoots = 10` and
        `grid_coarsening = n -> n ÷ 2`, then the grid will be coarsened to `[5, 2]`.

## Fields

  - `ode_alg`: configured ODE algorithm or `nothing`.
  - `nlsolve`: configured nonlinear-solver algorithm or `nothing`.
  - `optimize`: configured optimization-solver algorithm or `nothing`.
  - `jac_alg::BVPJacobianAlgorithm`: materialized Jacobian-algorithm configuration.
  - `nshoots::Int`: configured number of shooting subintervals.
  - `grid_coarsening`: configured grid-coarsening strategy.

## Returns

  - `MultipleShooting`: an algorithm object accepted by `SciMLBase.solve` for a boundary value
    problem.

## Examples

```jldoctest
using BoundaryValueDiffEqShooting: MultipleShooting
using OrdinaryDiffEqTsit5: Tsit5

alg = MultipleShooting(8, Tsit5(); grid_coarsening = true)
@assert alg isa MultipleShooting
# output
```
"""
@concrete struct MultipleShooting{J <: BVPJacobianAlgorithm} <: AbstractShooting
    ode_alg
    nlsolve
    optimize
    jac_alg::J
    nshoots::Int
    grid_coarsening
end

function concretize_jacobian_algorithm(alg::MultipleShooting, prob)
    jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    return MultipleShooting(
        alg.ode_alg, alg.nlsolve, alg.optimize, jac_alg, alg.nshoots, alg.grid_coarsening
    )
end

function update_nshoots(alg::MultipleShooting, nshoots::Int)
    return MultipleShooting(
        alg.ode_alg, alg.nlsolve, alg.optimize, alg.jac_alg, nshoots, alg.grid_coarsening
    )
end

function MultipleShooting(;
        nshoots::Int,
        ode_alg = nothing,
        nlsolve = nothing,
        optimize = nothing,
        grid_coarsening::Union{
            Bool, Function, <:AbstractVector{<:Integer}, Tuple{Vararg{Integer}},
        } = true,
        jac_alg = nothing
    )
    grid_coarsening isa Tuple && (grid_coarsening = Vector(grid_coarsening...))
    if grid_coarsening isa AbstractVector
        sort!(grid_coarsening; rev = true)
        @assert all(grid_coarsening .> 0) && 1 ∉ grid_coarsening
    end
    return MultipleShooting(
        ode_alg, nlsolve, optimize,
        __materialize_jacobian_algorithm(nlsolve, jac_alg), nshoots, grid_coarsening
    )
end
@inline MultipleShooting(nshoots::Int; kwargs...) = MultipleShooting(; nshoots, kwargs...)
@inline MultipleShooting(nshoots::Int, ode_alg; kwargs...) = MultipleShooting(; nshoots, ode_alg, kwargs...)
@inline MultipleShooting(
    nshoots::Int, ode_alg, nlsolve;
    kwargs...
) = MultipleShooting(; nshoots, ode_alg, nlsolve, kwargs...)
