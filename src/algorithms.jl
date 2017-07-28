# Algorithms
abstract type AbstractBoundaryValueAlgorithm end # This will eventually move to DiffEqBase.jl
abstract type BoundaryValueDiffEqAlgorithm <: AbstractBoundaryValueAlgorithm end
abstract type MIRK <: BoundaryValueDiffEqAlgorithm end
immutable Shooting{T,F} <: BoundaryValueDiffEqAlgorithm
    ode_alg::T
    nlsolve::F
end
DEFAULT_NLSOLVE = (loss, u0) -> (res=NLsolve.nlsolve(loss, u0);(res.zero, res.f_converged))
Shooting(ode_alg;nlsolve=DEFAULT_NLSOLVE) = Shooting(ode_alg,nlsolve)

immutable MIRK4 <: MIRK
    nlsolve
end

DEFAULT_KINSOL(loss, u0; kwargs...) = res=Sundials.kinsol(loss, u0; kwargs...)

MIRK4(;nlsolve=DEFAULT_KINSOL) = MIRK4(nlsolve)

