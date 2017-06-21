# Algorithms
abstract type AbstractBoundaryValueAlgorithm end # This will eventually move to DiffEqBase.jl
abstract type BoundaryValueDiffEqAlgorithm <: AbstractBoundaryValueAlgorithm end
immutable Shooting{T,F} <: BoundaryValueDiffEqAlgorithm
  ode_alg::T
  nlsolve::F
end
DEFAULT_NLSOLVE = (loss, u0) -> (res=NLsolve.nlsolve(loss, u0);(res.zero, res.f_converged))
Shooting(ode_alg;nlsolve=DEFAULT_NLSOLVE) = Shooting(ode_alg,nlsolve)

immutable MIRK{T} <: BoundaryValueDiffEqAlgorithm
    order::Int
    nlsolve::T
end

MIRK(order;nlsolve=DEFAULT_NLSOLVE) = MIRK(order,nlsolve)

