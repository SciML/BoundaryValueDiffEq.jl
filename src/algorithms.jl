const DEFAULT_NLSOLVE_SHOOTING = TrustRegion(; autodiff = Val(true))
const DEFAULT_NLSOLVE_MIRK = NewtonRaphson(; autodiff = Val(true))

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

"""
    MIRK2(; nlsolve = BoundaryValueDiffEq.DEFAULT_NLSOLVE_MIRK)
  
2nd order Monotonic Implicit Runge Kutta method, with Newton Raphson nonlinear solver as default.

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
Base.@kwdef struct MIRK2{N} <: AbstractMIRK
    nlsolve::N = DEFAULT_NLSOLVE_MIRK
end

"""
    MIRK3(; nlsolve = BoundaryValueDiffEq.DEFAULT_NLSOLVE_MIRK)
  
3rd order Monotonic Implicit Runge Kutta method, with Newton Raphson nonlinear solver as default.

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
Base.@kwdef struct MIRK3{N} <: AbstractMIRK
    nlsolve::N = DEFAULT_NLSOLVE_MIRK
end

"""
    MIRK4(; nlsolve = BoundaryValueDiffEq.DEFAULT_NLSOLVE_MIRK)
  
4th order Monotonic Implicit Runge Kutta method, with Newton Raphson nonlinear solver as default.

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
Base.@kwdef struct MIRK4{N} <: AbstractMIRK
    nlsolve::N = DEFAULT_NLSOLVE_MIRK
end

"""
    MIRK5(; nlsolve = BoundaryValueDiffEq.DEFAULT_NLSOLVE_MIRK)
  
5th order Monotonic Implicit Runge Kutta method, with Newton Raphson nonlinear solver as default.

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
Base.@kwdef struct MIRK5{N} <: AbstractMIRK
    nlsolve::N = DEFAULT_NLSOLVE_MIRK
end

"""
    MIRK6(; nlsolve = BoundaryValueDiffEq.DEFAULT_NLSOLVE_MIRK)
  
6th order Monotonic Implicit Runge Kutta method, with Newton Raphson nonlinear solver as default.

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
Base.@kwdef struct MIRK6{N} <: AbstractMIRK
    nlsolve::N = DEFAULT_NLSOLVE_MIRK
end
