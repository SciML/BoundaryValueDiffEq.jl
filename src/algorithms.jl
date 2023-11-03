# Algorithms
abstract type BoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end
abstract type AbstractMIRK <: BoundaryValueDiffEqAlgorithm end

"""
    Shooting(ode_alg; nlsolve = NewtonRaphson(), jac_alg = BVPJacobianAlgorithm())

Single shooting method, reduces BVP to an initial value problem and solves the IVP.

## Arguments

  - `ode_alg`: ODE algorithm to use for solving the IVP. Any solver which conforms to the
    SciML `ODEProblem` interface can be used!

## Keyword Arguments

  - `nlsolve`: Internal Nonlinear solver. Any solver which conforms to the SciML
    `NonlinearProblem` interface can be used.Note that any autodiff argument for the solver
    will be ignored and a custom jacobian algorithm will be used.
  - `jac_alg`: Jacobian Algorithm used for the nonlinear solver. Defaults to
    `BVPJacobianAlgorithm()`, which automatically decides the best algorithm to use based
    on the input types and problem type. Only `diffmode` is used (defaults to
    `AutoForwardDiff` if possible else `AutoFiniteDiff`).

!!! note
    For type-stability, the chunksizes for ForwardDiff ADTypes in `BVPJacobianAlgorithm`
    must be provided.
"""
struct Shooting{O, N, L <: BVPJacobianAlgorithm} <: BoundaryValueDiffEqAlgorithm
    ode_alg::O
    nlsolve::N
    jac_alg::L
end

function concretize_jacobian_algorithm(alg::Shooting, prob)
    jac_alg = alg.jac_alg
    diffmode = jac_alg.diffmode === nothing ? __default_nonsparse_ad(prob.u0) :
               jac_alg.diffmode
    return Shooting(alg.ode_alg, alg.nlsolve, BVPJacobianAlgorithm(diffmode))
end

function Shooting(ode_alg, nlsolve; jac_alg = BVPJacobianAlgorithm())
    return Shooting(ode_alg, nlsolve, jac_alg)
end

function Shooting(ode_alg; nlsolve = NewtonRaphson(), jac_alg = BVPJacobianAlgorithm())
    return Shooting(ode_alg, nlsolve, jac_alg)
end

"""
    MultipleShooting(nshoots::Int, ode_alg; nlsolve = NewtonRaphson(),
        grid_coarsening = true, jac_alg = BVPJacobianAlgorithm())

Multiple Shooting method, reduces BVP to an initial value problem and solves the IVP.
Significantly more stable than Single Shooting.

## Arguments

  - `nshoots`: Number of shooting points.
  - `ode_alg`: ODE algorithm to use for solving the IVP. Any solver which conforms to the
    SciML `ODEProblem` interface can be used!

## Keyword Arguments

  - `nlsolve`: Internal Nonlinear solver. Any solver which conforms to the SciML
    `NonlinearProblem` interface can be used. Note that any autodiff argument for the solver
    will be ignored and a custom jacobian algorithm will be used.
  - `jac_alg`: Jacobian Algorithm used for the nonlinear solver. Defaults to
    `BVPJacobianAlgorithm()`, which automatically decides the best algorithm to use based
    on the input types and problem type.
    - For `TwoPointBVProblem`, only `diffmode` is used (defaults to
      `AutoSparseForwardDiff` if possible else `AutoSparseFiniteDiff`).
    - For `BVProblem`, `bc_diffmode` and `nonbc_diffmode` are used. For `nonbc_diffmode`
      defaults to `AutoSparseForwardDiff` if possible else `AutoSparseFiniteDiff`. For
      `bc_diffmode`, defaults to `AutoForwardDiff` if possible else `AutoFiniteDiff`.
  - `grid_coarsening`: Coarsening the multiple-shooting grid to generate a stable IVP
    solution. Possible Choices:
    - `true`: Halve the grid size, till we reach a grid size of 1.
    - `false`: Do not coarsen the grid. Solve a Multiple Shooting Problem and finally
      solve a Single Shooting Problem.
    - `AbstractVector{<:Int}` or `Ntuple{N, <:Integer}`: Use the provided grid coarsening.
      For example, if `nshoots = 10` and `grid_coarsening = [5, 2]`, then the grid will be
      coarsened to `[5, 2]`. Note that `1` should not be present in the grid coarsening.
    - `Function`: Takes the current number of shooting points and returns the next number
      of shooting points. For example, if `nshoots = 10` and
      `grid_coarsening = n -> n ÷ 2`, then the grid will be coarsened to `[5, 2]`.

!!! note
    For type-stability, the chunksizes for ForwardDiff ADTypes in `BVPJacobianAlgorithm`
    must be provided.
"""
@concrete struct MultipleShooting{J <: BVPJacobianAlgorithm}
    ode_alg
    nlsolve
    jac_alg::J
    nshoots::Int
    grid_coarsening
end

function concretize_jacobian_algorithm(alg::MultipleShooting, prob)
    jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    return MultipleShooting(alg.ode_alg, alg.nlsolve, jac_alg, alg.nshoots,
        alg.grid_coarsening)
end

function update_nshoots(alg::MultipleShooting, nshoots::Int)
    return MultipleShooting(alg.ode_alg, alg.nlsolve, alg.jac_alg, nshoots,
        alg.grid_coarsening)
end

function MultipleShooting(nshoots::Int, ode_alg; nlsolve = NewtonRaphson(),
        grid_coarsening = true, jac_alg = BVPJacobianAlgorithm())
    @assert grid_coarsening isa Bool || grid_coarsening isa Function ||
            grid_coarsening isa AbstractVector{<:Integer} ||
            grid_coarsening isa NTuple{N, <:Integer} where {N}
    grid_coarsening isa Tuple && (grid_coarsening = Vector(grid_coarsening...))
    if grid_coarsening isa AbstractVector
        sort!(grid_coarsening; rev = true)
        @assert all(grid_coarsening .> 0) && 1 ∉ grid_coarsening
    end
    return MultipleShooting(ode_alg, nlsolve, jac_alg, nshoots, grid_coarsening)
end

for order in (2, 3, 4, 5, 6)
    alg = Symbol("MIRK$(order)")

    @eval begin
        """
            $($alg)(; nlsolve = NewtonRaphson(), jac_alg = BVPJacobianAlgorithm())

        $($order)th order Monotonic Implicit Runge Kutta method, with Newton Raphson nonlinear solver as default.

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
        struct $(alg){N, J <: BVPJacobianAlgorithm} <: AbstractMIRK
            nlsolve::N
            jac_alg::J
        end

        function $(alg)(; nlsolve = NewtonRaphson(), jac_alg = BVPJacobianAlgorithm())
            return $(alg)(nlsolve, jac_alg)
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

!!! warning
    Only supports inplace two-point boundary value problems, with very limited forms of
    input structures!

!!! note
    Only available if the `ODEInterface` package is loaded.
"""
Base.@kwdef struct BVPM2{S} <: BoundaryValueDiffEqAlgorithm
    max_num_subintervals::Int = 3000
    method_choice::Int = 4
    diagnostic_output::Int = -1
    error_control::Int = 1
    singular_term::S = nothing
end

"""
    BVPSOL(; bvpclass = 2, sol_method = 0, odesolver = nothing)
    BVPSOL(bvpclass::Int, sol_methods::Int, odesolver)

A FORTRAN77 code which solves highly nonlinear two point boundary value problems using a
local linear solver (condensing algorithm) or a global sparse linear solver for the solution
of the arising linear subproblems, by Peter Deuflhard, Georg Bader, Lutz Weimann.
For detailed documentation, see
[ODEInterface.jl](https://github.com/luchr/ODEInterface.jl/blob/master/doc/SolverOptions.md#bvpsol).

!!! warning
    Only supports inplace two-point boundary value problems, with very limited forms of
    input structures!

!!! note
    Only available if the `ODEInterface` package is loaded.
"""
Base.@kwdef struct BVPSOL{O} <: BoundaryValueDiffEqAlgorithm
    bvpclass::Int = 2
    sol_method::Int = 0
    odesolver::O = nothing
end
