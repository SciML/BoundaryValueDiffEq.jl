# Algorithms
abstract type BoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end
abstract type GeneralMIRK <: BoundaryValueDiffEqAlgorithm end
abstract type MIRK <: BoundaryValueDiffEqAlgorithm end

struct Shooting{T, F} <: BoundaryValueDiffEqAlgorithm
    ode_alg::T
    nlsolve::F
end
DEFAULT_NLSOLVE = (loss, u0) -> (res = NLsolve.nlsolve(loss, u0);
(res.zero,
    res.f_converged))
Shooting(ode_alg; nlsolve = DEFAULT_NLSOLVE) = Shooting(ode_alg, nlsolve)

"""
@article{Enright1996RungeKuttaSW,
  title={Runge-Kutta Software with Defect Control for Boundary Value ODEs},
  author={Wayne H. Enright and Paul H. Muir},
  journal={SIAM J. Sci. Comput.},
  year={1996},
  volume={17},
  pages={479-497}
}
"""
struct GeneralMIRK4 <: GeneralMIRK
    nlsolve::Any
end

"""
@article{Enright1996RungeKuttaSW,
  title={Runge-Kutta Software with Defect Control for Boundary Value ODEs},
  author={Wayne H. Enright and Paul H. Muir},
  journal={SIAM J. Sci. Comput.},
  year={1996},
  volume={17},
  pages={479-497}
}
"""
struct GeneralMIRK6 <: GeneralMIRK
    nlsolve::Any
end

"""
@article{Enright1996RungeKuttaSW,
  title={Runge-Kutta Software with Defect Control for Boundary Value ODEs},
  author={Wayne H. Enright and Paul H. Muir},
  journal={SIAM J. Sci. Comput.},
  year={1996},
  volume={17},
  pages={479-497}
}
"""
struct MIRK4 <: MIRK
    nlsolve::Any
end

"""
@article{Enright1996RungeKuttaSW,
  title={Runge-Kutta Software with Defect Control for Boundary Value ODEs},
  author={Wayne H. Enright and Paul H. Muir},
  journal={SIAM J. Sci. Comput.},
  year={1996},
  volume={17},
  pages={479-497}
}
"""
struct MIRK6 <: MIRK
    nlsolve::Any
end
GeneralMIRK4(; nlsolve = DEFAULT_NLSOLVE) = GeneralMIRK4(nlsolve)
GeneralMIRK6(; nlsolve = DEFAULT_NLSOLVE) = GeneralMIRK6(nlsolve)
MIRK4(; nlsolve = DEFAULT_NLSOLVE) = MIRK4(nlsolve)
MIRK6(; nlsolve = DEFAULT_NLSOLVE) = MIRK6(nlsolve)
