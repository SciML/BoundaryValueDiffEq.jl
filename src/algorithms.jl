const DEFAULT_NLSOLVE_SHOOTING = NewtonRaphson(; autodiff = AutoForwardDiff())
const DEFAULT_NLSOLVE_MIRK = NewtonRaphson(; autodiff = AutoForwardDiff())
const DEFAULT_JACOBIAN_ALGORITHM_MIRK = MIRKJacobianComputationAlgorithm()

# Algorithms
abstract type BoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end
abstract type AbstractMIRK <: BoundaryValueDiffEqAlgorithm end

"""
    Shooting(ode_alg; nlsolve = BoundaryValueDiffEq.DEFAULT_NLSOLVE_SHOOTING)

Single shooting method, reduces BVP to an initial value problem and solves the IVP.
"""
struct Shooting{O, N} <: BoundaryValueDiffEqAlgorithm
    ode_alg::O
    nlsolve::N
end

Shooting(ode_alg; nlsolve = DEFAULT_NLSOLVE_SHOOTING) = Shooting(ode_alg, nlsolve)

for order in (2, 3, 4, 5, 6)
    alg = Symbol("MIRK$(order)")

    @eval begin
        """
            $($alg)(; nlsolve = BoundaryValueDiffEq.DEFAULT_NLSOLVE_MIRK,
                jac_alg = BoundaryValueDiffEq.DEFAULT_JACOBIAN_ALGORITHM_MIRK)

        $($order)th order Monotonic Implicit Runge Kutta method, with Newton Raphson nonlinear solver as default.

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
        struct $(alg){N, J <: MIRKJacobianComputationAlgorithm} <: AbstractMIRK
            nlsolve::N
            jac_alg::J
        end

        function $(alg)(; nlsolve = DEFAULT_NLSOLVE_MIRK,
            jac_alg = DEFAULT_JACOBIAN_ALGORITHM_MIRK)
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
