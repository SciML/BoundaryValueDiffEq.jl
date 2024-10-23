# Algorithms from ODEInterface.jl
"""
    BVPM2(; max_num_subintervals = 3000, method_choice = 4, diagnostic_output = 1,
        error_control = 1, singular_term = nothing)
    BVPM2(max_num_subintervals::Int, method_choice::Int, diagnostic_output::Int,
        error_control::Int, singular_term)

Fortran code for solving two-point boundary value problems. For detailed documentation, see
[ODEInterface.jl](https://github.com/luchr/ODEInterface.jl/blob/master/doc/SolverOptions.md#bvpm2).

## Keyword Arguments:

    - `max_num_subintervals`: Number of maximal subintervals, default as 3000.
    - `method_choice`: Choice for IVP-solvers, default as Runge-Kutta method of order 4,
      available choices:
        - `2`: Runge-Kutta method of order 2.
        - `4`: Runge-Kutta method of order 4.
        - `6`: Runge-Kutta method of order 6.
    - `diagnostic_output`: Diagnostic output for BVPM2, default as non printout, available
      choices:
        - `-1`: Full diagnostic printout.
        - `0`: Selected printout.
        - `1`: No printout.
    - `error_control`: Determines the error-estimation for which RTOL is used, default as
      defect control, available choices:
        - `1`: Defect control.
        - `2`: Global error control.
        - `3`: Defect and then global error control.
        - `4`: Linear combination of defect and global error control.
    - `singular_term`: either nothing if the ODEs have no singular terms at the left
      boundary or a constant (d,d) matrix for the
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
            error("`BVPM2` requires `ODEInterface.jl` to be loaded")
        end
        return new{typeof(singular_term)}(max_num_subintervals, method_choice,
            diagnostic_output, error_control, singular_term)
    end
end

function BVPM2(; max_num_subintervals::Int = 3000, method_choice::Int = 4,
        diagnostic_output::Int = -1, error_control::Int = 1, singular_term = nothing)
    return BVPM2(max_num_subintervals, method_choice,
        diagnostic_output, error_control, singular_term)
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

    - `bvpclass`: Boundary value problem classification, default as highly nonlinear with
      bad initial data, available choices:
        - `0`: Linear boundary value problem.
        - `1`: Nonlinear with good initial data.
        - `2`: Highly Nonlinear with bad initial data.
        - `3`: Highly nonlinear with bad initial data and initial rank reduction to
          seperable linear boundary conditions.
    - `sol_method`: Switch for solution methods, default as local linear solver with
      condensing algorithm, available choices:
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
            error("`BVPSOL` requires `ODEInterface.jl` to be loaded")
        end
        return new{typeof(odesolver)}(bvpclass, sol_method, odesolver)
    end
end

function BVPSOL(; bvpclass = 2, sol_method = 0, odesolver = nothing)
    return BVPSOL(bvpclass, sol_method, odesolver)
end

"""
    COLNEW(; bvpclass = 1, collocationpts = 7, diagnostic_output = 1,
        max_num_subintervals = 3000, bc_func = nothing, dbc_func = nothing,
        zeta = nothing)
    COLNEW(bvpclass::Int, collocationpts::Int, diagnostic_output::Int,
        max_num_subintervals::Int, bc_func, dbc_func, zeta::AbstractVector)

## Keyword Arguments:

  - `bvpclass`: Boundary value problem classification, default as nonlinear and
    "extra sensitive", available choices:

      + `0`: Linear boundary value problem.
      + `1`: Nonlinear and regular.
      + `2`: Nonlinear and "extra sensitive" (first relax factor is rstart and the
        nonlinear iteration does not rely on past convergence).
      + `3`: fail-early: return immediately upon: a) two successive non-convergences.
        b) after obtaining an error estimate for the first time.

  - `collocationpts`: Number of collocation points per subinterval. Require
    orders[i] ≤ k ≤ 7, default as 7
  - `diagnostic_output`: Diagnostic output for COLNEW, default as no printout, available
    choices:

      + `-1`: Full diagnostic printout.
      + `0`: Selected printout.
      + `1`: No printout.
  - `max_num_subintervals`: Number of maximal subintervals, default as 3000.
  - `bc_func`: Boundary condition accord with ODEInterface.jl, only used for multi-points BVP.
  - `dbc_func`: Jacobian of boundary condition accord with ODEInterface.jl, only used for multi-points BVP.
  - `zeta`: The points in interval where boundary conditions are specified, only used for multi-points BVP.

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
    bc_func::Union{Function, Nothing}
    dbc_func::Union{Function, Nothing}
    zeta::Union{AbstractVector, Nothing}

    function COLNEW(bvpclass::Int, collocationpts::Int, diagnostic_output::Int,
            max_num_subintervals::Int, bc_func::Union{Function, Nothing},
            dbc_func::Union{Function, Nothing}, zeta::Union{AbstractVector, Nothing})
        if Base.get_extension(@__MODULE__, :BoundaryValueDiffEqODEInterfaceExt) === nothing
            error("`COLNEW` requires `ODEInterface.jl` to be loaded")
        end
        return new(bvpclass, collocationpts, diagnostic_output,
            max_num_subintervals, bc_func, dbc_func, zeta)
    end
end

function COLNEW(; bvpclass::Int = 1, collocationpts::Int = 7, diagnostic_output::Int = 1,
        max_num_subintervals::Int = 3000, bc_func::Union{Function, Nothing} = nothing,
        dbc_func::Union{Function, Nothing} = nothing,
        zeta::Union{AbstractVector, Nothing} = nothing)
    return COLNEW(bvpclass, collocationpts, diagnostic_output,
        max_num_subintervals, bc_func, dbc_func, zeta)
end
