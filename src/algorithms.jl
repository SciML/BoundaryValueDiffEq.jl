# Algorithms
abstract type AbstractBoundaryValueAlgorithm end # This will eventually move to DiffEqBase.jl
abstract type BoundaryValueDiffEqAlgorithm <: AbstractBoundaryValueAlgorithm end
abstract type GeneralMIRK <: BoundaryValueDiffEqAlgorithm end
abstract type MIRK <: BoundaryValueDiffEqAlgorithm end

struct Shooting{T,F} <: BoundaryValueDiffEqAlgorithm
  ode_alg::T
  nlsolve::F
end
DEFAULT_NLSOLVE = (loss, u0) -> (res=NLsolve.nlsolve(loss, u0);(res.zero, res.f_converged))
Shooting(ode_alg;nlsolve=DEFAULT_NLSOLVE) = Shooting(ode_alg,nlsolve)

struct GeneralMIRK4 <: GeneralMIRK
    nlsolve
end
struct MIRK4 <: MIRK
    nlsolve
end
GeneralMIRK4(;nlsolve=DEFAULT_NLSOLVE) = GeneralMIRK4(nlsolve)
MIRK4(;nlsolve=DEFAULT_NLSOLVE) = MIRK4(nlsolve)
