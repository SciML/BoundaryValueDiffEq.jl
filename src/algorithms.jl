# Algorithms
abstract AbstractBoundaryValueAlgorithm # This will eventually move to DiffEqBase.jl
abstract BoundaryValueDiffEqAlgorithm <: AbstractBoundaryValueAlgorithm
immutable Shooting{T,F} <: BoundaryValueDiffEqAlgorithm
  ode_alg::T
  nlsolve::F
end
Shooting(ode_alg;nlsolve=NLsolve.nlsolve) = Shooting(ode_alg,nlsolve)
