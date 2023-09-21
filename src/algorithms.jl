const DEFAULT_NLSOLVE_SHOOTING = TrustRegion(; autodiff = Val(true))
const DEFAULT_NLSOLVE_MIRK = NewtonRaphson(; autodiff = Val(true))
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

for order in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIb$(order)")

    @eval begin
        """
            $($alg)(; nlsolve = BoundaryValueDiffEq.DEFAULT_NLSOLVE_MIRK,
                jac_alg = BoundaryValueDiffEq.DEFAULT_JACOBIAN_ALGORITHM_MIRK)

        $($order)th order LobattoIIIb method, with Newton Raphson nonlinear solver as default.

        ## References
        TODO
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
