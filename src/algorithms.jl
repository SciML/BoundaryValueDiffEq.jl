# Algorithms
abstract AbstractBoundaryValueAlgorithm # This will eventually move to DiffEqBase.jl
abstract BoundaryValueDiffEqAlgorithm <: AbstractBoundaryValueAlgorithm
immutable Shooting{T,F} <: BoundaryValueDiffEqAlgorithm
  ode_alg::T
  nlsolve::F
end
DEFAULT_NLSOLVE = (loss, u0) -> (res=NLsolve.nlsolve(loss, u0);res.zero)
Shooting(ode_alg;nlsolve=DEFAULT_NLSOLVE) = Shooting(ode_alg,nlsolve)

immutable MIRK{T,F} <: BoundaryValueDiffEqAlgorithm
    order::Int
    dt::T
    nlsolve::F
end
function DEFAULT_NLSOLVE_MIRK(loss, u0, M, N)
    res = NLsolve.nlsolve(NLsolve.not_in_place(loss), vec(u0))
    opt = res.zero
    reshape(opt, M, N)
end
MIRK(order,dt;nlsolve=DEFAULT_NLSOLVE_MIRK) = MIRK(order,dt,nlsolve)
