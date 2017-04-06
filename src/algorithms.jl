# Algorithms
abstract AbstractBoundaryValueAlgorithm # This will eventually move to DiffEqBase.jl
abstract BoundaryValueDiffEqAlgorithm <: AbstractBoundaryValueAlgorithm
immutable Shooting{T} <: BoundaryValueDiffEqAlgorithm 
  ode_alg::T
end
