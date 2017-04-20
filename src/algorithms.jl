# Algorithms
abstract AbstractBoundaryValueAlgorithm # This will eventually move to DiffEqBase.jl
abstract BoundaryValueDiffEqAlgorithm <: AbstractBoundaryValueAlgorithm
immutable Shooting{T,F} <: BoundaryValueDiffEqAlgorithm
  ode_alg::T
  nlsolve::F
end
DEFAULT_NLSOLVE = (loss, u0) -> (res=NLsolve.nlsolve(loss, u0);res.zero)
Shooting(ode_alg;nlsolve=DEFAULT_NLSOLVE) = Shooting(ode_alg,nlsolve)
