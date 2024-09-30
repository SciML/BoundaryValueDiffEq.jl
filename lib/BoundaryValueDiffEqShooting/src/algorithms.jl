# Algorithms
abstract type AbstractShooting <: BoundaryValueDiffEqAlgorithm end

"""
    Shooting(ode_alg; kwargs...)
    Shooting(ode_alg, nlsolve; kwargs...)
    Shooting(; ode_alg = nothing, nlsolve = nothing, jac_alg = nothing)

Single shooting method, reduces BVP to an initial value problem and solves the IVP.

## Arguments

  - `ode_alg`: ODE algorithm to use for solving the IVP. Any solver which conforms to the
    SciML `ODEProblem` interface can be used! (Defaults to `nothing` which will use
    poly-algorithm if `DifferentialEquations.jl` is loaded else this must be supplied)
  - `nlsolve`: Internal Nonlinear solver. Any solver which conforms to the SciML
    `NonlinearProblem` interface can be used. Note that any autodiff argument for the solver
    will be ignored and a custom jacobian algorithm will be used.
  - `jac_alg`: Jacobian Algorithm used for the Nonlinear Solver. If this is not set, we
    check if `nlsolve.ad` exists and is not nothing. If it is, we use that to construct
    the jacobian. If not, we try to use the best algorithm based on the input types
    and problem type. If `BVPJacobianAlgorithm` is provided, only `diffmode` is used
    (defaults to `AutoForwardDiff` if possible else `AutoFiniteDiff`).
"""
@concrete struct Shooting{J <: BVPJacobianAlgorithm} <: AbstractShooting
    ode_alg
    nlsolve
    jac_alg::J
end

function Shooting(; ode_alg = nothing, nlsolve = nothing, jac_alg = nothing)
    return Shooting(ode_alg, nlsolve, __materialize_jacobian_algorithm(nlsolve, jac_alg))
end
@inline Shooting(ode_alg; kwargs...) = Shooting(; ode_alg, kwargs...)
@inline Shooting(ode_alg, nlsolve; kwargs...) = Shooting(; ode_alg, nlsolve, kwargs...)

@inline function concretize_jacobian_algorithm(alg::Shooting, prob)
    alg.jac_alg.diffmode === nothing &&
        (return @set alg.jac_alg.diffmode = __default_nonsparse_ad(prob.u0))
    return alg
end

"""
    MultipleShooting(; nshoots::Int, ode_alg = nothing, nlsolve = nothing,
        grid_coarsening = true, jac_alg = nothing)
    MultipleShooting(nshoots::Int; kwargs...)
    MultipleShooting(nshoots::Int, ode_alg; kwargs...)
    MultipleShooting(nshoots::Int, ode_alg, nlsolve; kwargs...)

Multiple Shooting method, reduces BVP to an initial value problem and solves the IVP.
Significantly more stable than Single Shooting.

## Arguments

  - `nshoots`: Number of shooting points.

  - `ode_alg`: ODE algorithm to use for solving the IVP. Any solver which conforms to the
    SciML `ODEProblem` interface can be used! (Defaults to `nothing` which will use
    poly-algorithm if `DifferentialEquations.jl` is loaded else this must be supplied)
  - `nlsolve`: Internal Nonlinear solver. Any solver which conforms to the SciML
    `NonlinearProblem` interface can be used.
  - `jac_alg`: Jacobian Algorithm used for the nonlinear solver. Defaults to
    `BVPJacobianAlgorithm()`, which automatically decides the best algorithm to use based
    on the input types and problem type.

      + For `TwoPointBVProblem`, only `diffmode` is used (defaults to
        `AutoSparse(AutoForwardDiff())` if possible else `AutoSparse(AutoFiniteDiff())`).
      + For `BVProblem`, `bc_diffmode` and `nonbc_diffmode` are used. For `nonbc_diffmode`
        we default to `AutoSparse(AutoForwardDiff())` if possible else
        `AutoSparse(AutoFiniteDiff())`. For `bc_diffmode`, we default to `AutoForwardDiff`
        if possible else `AutoFiniteDiff`.
  - `grid_coarsening`: Coarsening the multiple-shooting grid to generate a stable IVP
    solution. Possible Choices:

      + `true`: Halve the grid size, till we reach a grid size of 1.
      + `false`: Do not coarsen the grid. Solve a Multiple Shooting Problem and finally
        solve a Single Shooting Problem.
      + `AbstractVector{<:Int}` or `Ntuple{N, <:Integer}`: Use the provided grid coarsening.
        For example, if `nshoots = 10` and `grid_coarsening = [5, 2]`, then the grid will be
        coarsened to `[5, 2]`. Note that `1` should not be present in the grid coarsening.
      + `Function`: Takes the current number of shooting points and returns the next number
        of shooting points. For example, if `nshoots = 10` and
        `grid_coarsening = n -> n ÷ 2`, then the grid will be coarsened to `[5, 2]`.
"""
@concrete struct MultipleShooting{J <: BVPJacobianAlgorithm} <: AbstractShooting
    ode_alg
    nlsolve
    jac_alg::J
    nshoots::Int
    grid_coarsening
end

function concretize_jacobian_algorithm(alg::MultipleShooting, prob)
    jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    return MultipleShooting(
        alg.ode_alg, alg.nlsolve, jac_alg, alg.nshoots, alg.grid_coarsening)
end

function update_nshoots(alg::MultipleShooting, nshoots::Int)
    return MultipleShooting(
        alg.ode_alg, alg.nlsolve, alg.jac_alg, nshoots, alg.grid_coarsening)
end

function MultipleShooting(; nshoots::Int,
        ode_alg = nothing,
        nlsolve = nothing,
        grid_coarsening::Union{
            Bool, Function, <:AbstractVector{<:Integer}, Tuple{Vararg{Integer}}} = true,
        jac_alg = nothing)
    grid_coarsening isa Tuple && (grid_coarsening = Vector(grid_coarsening...))
    if grid_coarsening isa AbstractVector
        sort!(grid_coarsening; rev = true)
        @assert all(grid_coarsening .> 0) && 1 ∉ grid_coarsening
    end
    return MultipleShooting(
        ode_alg, nlsolve, __materialize_jacobian_algorithm(nlsolve, jac_alg),
        nshoots, grid_coarsening)
end
@inline MultipleShooting(nshoots::Int; kwargs...) = MultipleShooting(; nshoots, kwargs...)
@inline MultipleShooting(nshoots::Int, ode_alg; kwargs...) = MultipleShooting(;
    nshoots, ode_alg, kwargs...)
@inline MultipleShooting(nshoots::Int, ode_alg, nlsolve; kwargs...) = MultipleShooting(;
    nshoots, ode_alg, nlsolve, kwargs...)
