# Algorithms
abstract type BoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end
abstract type AbstractShooting <: BoundaryValueDiffEqAlgorithm end
abstract type AbstractMIRK <: BoundaryValueDiffEqAlgorithm end

## Disable the ugly verbose printing by default
@inline __modifier_text!(list, fieldname, field) = push!(list, "$fieldname = $(field)")
@inline __modifier_text!(list, fieldname, ::Nothing) = list
@inline __modifier_text!(list, fieldname, ::Missing) = list
@inline function __modifier_text!(list, fieldname, field::SciMLBase.AbstractODEAlgorithm)
    push!(list, "$fieldname = $(__nameof(field))()")
end

function Base.show(io::IO, alg::BoundaryValueDiffEqAlgorithm)
    print(io, "$(__nameof(alg))(")
    modifiers = String[]
    for field in fieldnames(typeof(alg))
        __modifier_text!(modifiers, field, getfield(alg, field))
    end
    print(io, join(modifiers, ", "))
    print(io, ")")
end

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
        `AutoSparseForwardDiff` if possible else `AutoSparseFiniteDiff`).
      + For `BVProblem`, `bc_diffmode` and `nonbc_diffmode` are used. For `nonbc_diffmode`
        we default to `AutoSparseForwardDiff` if possible else `AutoSparseFiniteDiff`. For
        `bc_diffmode`, we default to `AutoForwardDiff` if possible else `AutoFiniteDiff`.
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

# function Base.show(io::IO, alg::MultipleShooting)
#     print(io, "MultipleShooting(")
#     modifiers = String[]
#     alg.nlsolve !== nothing && push!(modifiers, "nlsolve = $(alg.nlsolve)")
#     alg.jac_alg !== nothing && push!(modifiers, "jac_alg = $(alg.jac_alg)")
#     alg.ode_alg !== nothing && push!(modifiers, "ode_alg = $(__nameof(alg.ode_alg))()")
#     print(io, join(modifiers, ", "))
#     print(io, ")")
# end

function concretize_jacobian_algorithm(alg::MultipleShooting, prob)
    jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    return MultipleShooting(alg.ode_alg, alg.nlsolve, jac_alg, alg.nshoots,
        alg.grid_coarsening)
end

function update_nshoots(alg::MultipleShooting, nshoots::Int)
    return MultipleShooting(alg.ode_alg, alg.nlsolve, alg.jac_alg, nshoots,
        alg.grid_coarsening)
end

function MultipleShooting(; nshoots::Int, ode_alg = nothing, nlsolve = nothing,
        grid_coarsening::Union{Bool, Function, <:AbstractVector{<:Integer},
            Tuple{Vararg{Integer}}} = true, jac_alg = nothing)
    grid_coarsening isa Tuple && (grid_coarsening = Vector(grid_coarsening...))
    if grid_coarsening isa AbstractVector
        sort!(grid_coarsening; rev = true)
        @assert all(grid_coarsening .> 0) && 1 ∉ grid_coarsening
    end
    return MultipleShooting(ode_alg, nlsolve,
        __materialize_jacobian_algorithm(nlsolve, jac_alg), nshoots, grid_coarsening)
end
@inline MultipleShooting(nshoots::Int; kwargs...) = MultipleShooting(; nshoots, kwargs...)
@inline MultipleShooting(nshoots::Int, ode_alg; kwargs...) = MultipleShooting(;
    nshoots, ode_alg, kwargs...)
@inline MultipleShooting(nshoots::Int, ode_alg, nlsolve; kwargs...) = MultipleShooting(;
    nshoots, ode_alg, nlsolve, kwargs...)

for order in (2, 3, 4, 5, 6)
    alg = Symbol("MIRK$(order)")

    @eval begin
        """
            $($alg)(; nlsolve = NewtonRaphson(), jac_alg = BVPJacobianAlgorithm())

        $($order)th order Monotonic Implicit Runge Kutta method.

        ## Keyword Arguments

          - `nlsolve`: Internal Nonlinear solver. Any solver which conforms to the SciML
            `NonlinearProblem` interface can be used. Note that any autodiff argument for
            the solver will be ignored and a custom jacobian algorithm will be used.
          - `jac_alg`: Jacobian Algorithm used for the nonlinear solver. Defaults to
            `BVPJacobianAlgorithm()`, which automatically decides the best algorithm to
            use based on the input types and problem type.
            - For `TwoPointBVProblem`, only `diffmode` is used (defaults to
              `AutoSparseForwardDiff` if possible else `AutoSparseFiniteDiff`).
            - For `BVProblem`, `bc_diffmode` and `nonbc_diffmode` are used. For
              `nonbc_diffmode` defaults to `AutoSparseForwardDiff` if possible else
              `AutoSparseFiniteDiff`. For `bc_diffmode`, defaults to `AutoForwardDiff` if
              possible else `AutoFiniteDiff`.

        !!! note
            For type-stability, the chunksizes for ForwardDiff ADTypes in
            `BVPJacobianAlgorithm` must be provided.

        ## References

        @article{Enright1996RungeKuttaSW,
            title={Runge-Kutta Software with Defect Control for Boundary Value ODEs},
            author={Wayne H. Enright and Paul H. Muir},
            journal={SIAM J. Sci. Comput.},
            year={1996},
            volume={17},
            pages={479-497}
        }
        """
        Base.@kwdef struct $(alg){N, J <: BVPJacobianAlgorithm} <: AbstractMIRK
            nlsolve::N = nothing
            jac_alg::J = BVPJacobianAlgorithm()
        end
    end
end

"""
    BVPM2(; max_num_subintervals = 3000, method_choice = 4, diagnostic_output = 1,
        error_control = 1, singular_term = nothing)
    BVPM2(max_num_subintervals::Int, method_choice::Int, diagnostic_output::Int,
        error_control::Int, singular_term)

Fortran code for solving two-point boundary value problems. For detailed documentation, see
[ODEInterface.jl](https://github.com/luchr/ODEInterface.jl/blob/master/doc/SolverOptions.md#bvpm2).

## Keyword Arguments:

    - `max_num_subintervals`: Number of maximal subintervals, default as 3000.
    - `method_choice`: Choice for IVP-solvers, default as Runge-Kutta method of order 4, available choices:
        - `2`: Runge-Kutta method of order 2.
        - `4`: Runge-Kutta method of order 4.
        - `6`: Runge-Kutta method of order 6.
    - `diagnostic_output`: Diagnostic output for BVPM2, default as non printout, available choices:
        - `-1`: Full diagnostic printout.
        - `0`: Selected printout.
        - `1`: No printout.
    - `error_control`: Determines the error-estimation for which RTOL is used, default as defect control, available choices:
        - `1`: Defect control.
        - `2`: Global error control.
        - `3`: Defect and then global error control.
        - `4`: Linear combination of defect and global error control.
    - `singular_term`: either nothing if the ODEs have no singular terms at the left boundary or a constant (d,d) matrix for the
        singular term.

!!! note

    Only available if the `ODEInterface` package is loaded.
"""
struct BVPM2{S} <: BoundaryValueDiffEqAlgorithm
    max_num_subintervals::Int
    method_choice::Int
    diagnostic_output::Int
    error_control::Int
    singular_term::S

    function BVPM2(max_num_subintervals::Int, method_choice::Int, diagnostic_output::Int,
            error_control::Int, singular_term::Union{Nothing, AbstractMatrix})
        if Base.get_extension(@__MODULE__, :BoundaryValueDiffEqODEInterfaceExt) === nothing
            error("BVPM2 requires ODEInterface.jl to be loaded")
        end
        return new{typeof(singular_term)}(max_num_subintervals, method_choice,
            diagnostic_output, error_control, singular_term)
    end
end

function BVPM2(; max_num_subintervals::Int = 3000, method_choice::Int = 4,
        diagnostic_output::Int = -1, error_control::Int = 1, singular_term = nothing)
    return BVPM2(max_num_subintervals, method_choice, diagnostic_output, error_control,
        singular_term)
end

"""
    BVPSOL(; bvpclass = 2, sol_method = 0, odesolver = nothing)
    BVPSOL(bvpclass::Int, sol_methods::Int, odesolver)

A FORTRAN77 code which solves highly nonlinear **two point boundary value problems** using a
local linear solver (condensing algorithm) or a global sparse linear solver for the solution
of the arising linear subproblems, by Peter Deuflhard, Georg Bader, Lutz Weimann.
For detailed documentation, see
[ODEInterface.jl](https://github.com/luchr/ODEInterface.jl/blob/master/doc/SolverOptions.md#bvpsol).

## Keyword Arguments

    - `bvpclass`: Boundary value problem classification, default as highly nonlinear with bad initial data, available choices:
        - `0`: Linear boundary value problem.
        - `1`: Nonlinear with good initial data.
        - `2`: Highly Nonlinear with bad initial data.
        - `3`: Highly nonlinear with bad initial data and initial rank reduction to seperable
            linear boundary conditions.
    - `sol_method`: Switch for solution methods, default as local linear solver with condensing algorithm, available choices:
        - `0`: Use local linear solver with condensing algorithm.
        - `1`: Use global sparse linear solver.
    - `odesolver`: Either `nothing` or ode-solver(dopri5, dop853, seulex, etc.).

!!! note

    Only available if the `ODEInterface` package is loaded.
"""
struct BVPSOL{O} <: BoundaryValueDiffEqAlgorithm
    bvpclass::Int
    sol_method::Int
    odesolver::O

    function BVPSOL(bvpclass::Int, sol_method::Int, odesolver)
        if Base.get_extension(@__MODULE__, :BoundaryValueDiffEqODEInterfaceExt) === nothing
            error("BVPSOL requires ODEInterface.jl to be loaded")
        end
        return new{typeof(odesolver)}(bvpclass, sol_method, odesolver)
    end
end

"""
    COLNEW(; bvpclass = 1, collocationpts = 7, diagnostic_output = 1,
        max_num_subintervals = 3000)
    COLNEW(bvpclass::Int, collocationpts::Int, diagnostic_output::Int,
        max_num_subintervals::Int)

## Keyword Arguments:

    - `bvpclass`: Boundary value problem classification, default as nonlinear and "extra sensitive", available choices:
        - `0`: Linear boundary value problem.
        - `1`: Nonlinear and regular.
        - `2`: Nonlinear and "extra sensitive" (first relax factor is rstart and the
            nonlinear iteration does not rely on past convergence).
        - `3`: fail-early: return immediately upon:
            (a) two successive non-convergences.
            (b) after obtaining an error estimate for the first time.
    - `collocationpts`: Number of collocation points per subinterval. Require orders[i] ≤ k ≤ 7, default as 7
    - `diagnostic_output`: Diagnostic output for COLNEW, default as no printout, available choices:
        - `-1`: Full diagnostic printout.
        - `0`: Selected printout.
        - `1`: No printout.
    - `max_num_subintervals`: Number of maximal subintervals, default as 3000.

A Fortran77 code solves a multi-points boundary value problems for a mixed order system of
ODEs. It incorporates a new basis representation replacing b-splines, and improvements for
the linear and nonlinear algebraic equation solvers.

!!! warning
    Only supports two-point boundary value problems.

!!! note
    Only available if the `ODEInterface` package is loaded.
"""
struct COLNEW <: BoundaryValueDiffEqAlgorithm
    bvpclass::Int
    collocationpts::Int
    diagnostic_output::Int
    max_num_subintervals::Int

    function COLNEW(; bvpclass::Int = 1, collocationpts::Int = 7,
            diagnostic_output::Int = 1, max_num_subintervals::Int = 3000)
        return COLNEW(bvpclass, collocationpts, diagnostic_output, max_num_subintervals)
    end
    function COLNEW(bvpclass::Int, collocationpts::Int, diagnostic_output::Int,
            max_num_subintervals::Int)
        if Base.get_extension(@__MODULE__, :BoundaryValueDiffEqODEInterfaceExt) === nothing
            error("COLNEW requires ODEInterface.jl to be loaded")
        end
        return new(bvpclass, collocationpts, diagnostic_output, max_num_subintervals)
    end
end
