# Algorithms
abstract type BoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end
abstract type AbstractRK <: BoundaryValueDiffEqAlgorithm end
abstract type AbstractMIRK <: BoundaryValueDiffEqAlgorithm end
abstract type AbstractFIRK <: BoundaryValueDiffEqAlgorithm end
abstract type AbstractRKCache{iip, T} end

"""
    Shooting(ode_alg; nlsolve = NewtonRaphson())

Single shooting method, reduces BVP to an initial value problem and solves the IVP.
"""
struct Shooting{O, N} <: BoundaryValueDiffEqAlgorithm
    ode_alg::O
    nlsolve::N
end

Shooting(ode_alg; nlsolve = NewtonRaphson()) = Shooting(ode_alg, nlsolve)

"""
    MultipleShooting(nshoots::Int, ode_alg; nlsolve = NewtonRaphson(),
        grid_coarsening = true)

Multiple Shooting method, reduces BVP to an initial value problem and solves the IVP.
Significantly more stable than Single Shooting.
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

for order in (1, 3, 5, 9, 13)
    alg = Symbol("RadauIIa$(order)")

    @eval begin
        """
            $($alg)(; nlsolve = NewtonRaphson(),
                jac_alg = BVPJacobianAlgorithm())

        $($order)th order RadauIIa method, with Newton Raphson nonlinear solver as default.

        ## References
        TODO
        }
        """
        struct $(alg){N, J <: BVPJacobianAlgorithm} <: AbstractFIRK
            nlsolve::N
            jac_alg::J
            nested_nlsolve::Bool
        end

        function $(alg)(; nlsolve = NewtonRaphson(),
            jac_alg = BVPJacobianAlgorithm(),
            nested_nlsolve = true)
            return $(alg)(nlsolve, jac_alg, nested_nlsolve)
        end
    end
end


for order in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIa$(order)")

    @eval begin
        """
            $($alg)(; nlsolve = NewtonRaphson(),
                jac_alg = BVPJacobianAlgorithm())

        $($order)th order LobattoIIIa method, with Newton Raphson nonlinear solver as default.

        ## References
        TODO
        }
        """
        struct $(alg){N, J <: BVPJacobianAlgorithm} <: AbstractFIRK
            nlsolve::N
            jac_alg::J
            nested_nlsolve::Bool
        end

        function $(alg)(; nlsolve = NewtonRaphson(),
            jac_alg = BVPJacobianAlgorithm(),
            nested_nlsolve = true)
            return $(alg)(nlsolve, jac_alg, nested_nlsolve)
        end
    end
end

for order in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIb$(order)")

    @eval begin
        """
            $($alg)(; nlsolve = NewtonRaphson(),
                jac_alg = BVPJacobianAlgorithm())

        $($order)th order LobattoIIIb method, with Newton Raphson nonlinear solver as default.

        ## References
        TODO
        }
        """
        struct $(alg){N, J <: BVPJacobianAlgorithm} <: AbstractFIRK
            nlsolve::N
            jac_alg::J
            nested_nlsolve::Bool
        end

        function $(alg)(; nlsolve = NewtonRaphson(),
            jac_alg = BVPJacobianAlgorithm(),
            nested_nlsolve = true)
            return $(alg)(nlsolve, jac_alg, nested_nlsolve)
        end
    end
end


for order in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIc$(order)")

    @eval begin
        """
            $($alg)(; nlsolve = NewtonRaphson(),
                jac_alg = BVPJacobianAlgorithm())

        $($order)th order LobattoIIIc method, with Newton Raphson nonlinear solver as default.

        ## References
        TODO
        }
        """
        struct $(alg){N, J <: BVPJacobianAlgorithm} <: AbstractFIRK
            nlsolve::N
            jac_alg::J
            nested_nlsolve::Bool
        end

        function $(alg)(; nlsolve = NewtonRaphson(),
            jac_alg = BVPJacobianAlgorithm(),
            nested_nlsolve = true)
            return $(alg)(nlsolve, jac_alg, nested_nlsolve)
        end
    end
end

# FIRK Algorithms that don't use adaptivity
const FIRKNoAdaptivity = Union{LobattoIIIb2, RadauIIa1}

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
