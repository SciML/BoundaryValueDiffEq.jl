const DEFAULT_NLSOLVE_SHOOTING = TrustRegion(; autodiff = Val(true))
const DEFAULT_NLSOLVE_MIRK = NewtonRaphson(; autodiff = Val(true))

# Algorithms
abstract type BoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end
abstract type GeneralMIRK <: BoundaryValueDiffEqAlgorithm end
abstract type MIRK <: BoundaryValueDiffEqAlgorithm end

struct Shooting{O, N} <: BoundaryValueDiffEqAlgorithm
    ode_alg::O
    nlsolve::N
end

Shooting(ode_alg; nlsolve = DEFAULT_NLSOLVE_SHOOTING) = Shooting(ode_alg, nlsolve)

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
struct GeneralMIRK3{N} <: GeneralMIRK
    nlsolve::N
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
struct GeneralMIRK4{N} <: GeneralMIRK
    nlsolve::N
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
struct GeneralMIRK5{N} <: GeneralMIRK
    nlsolve::N
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
struct GeneralMIRK6{N} <: GeneralMIRK
    nlsolve::N
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
struct MIRK3{N} <: MIRK
    nlsolve::N
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
struct MIRK4{N} <: MIRK
    nlsolve::N
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
struct MIRK5{N} <: MIRK
    nlsolve::N
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
struct MIRK6{N} <: MIRK
    nlsolve::N
end

GeneralMIRK3(; nlsolve = DEFAULT_NLSOLVE_MIRK) = GeneralMIRK3(nlsolve)
GeneralMIRK4(; nlsolve = DEFAULT_NLSOLVE_MIRK) = GeneralMIRK4(nlsolve)
GeneralMIRK5(; nlsolve = DEFAULT_NLSOLVE_MIRK) = GeneralMIRK5(nlsolve)
GeneralMIRK6(; nlsolve = DEFAULT_NLSOLVE_MIRK) = GeneralMIRK6(nlsolve)
MIRK3(; nlsolve = DEFAULT_NLSOLVE_MIRK) = MIRK3(nlsolve)
MIRK4(; nlsolve = DEFAULT_NLSOLVE_MIRK) = MIRK4(nlsolve)
MIRK5(; nlsolve = DEFAULT_NLSOLVE_MIRK) = MIRK5(nlsolve)
MIRK6(; nlsolve = DEFAULT_NLSOLVE_MIRK) = MIRK6(nlsolve)
