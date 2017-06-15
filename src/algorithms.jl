# Algorithms
abstract type AbstractBoundaryValueAlgorithm end # This will eventually move to DiffEqBase.jl
abstract type BoundaryValueDiffEqAlgorithm <: AbstractBoundaryValueAlgorithm end
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

# Auxiliary functions for working with vector of vectors
function vector_alloc(T, M, N)
    v = Vector{Vector{T}}(N)
    for i in eachindex(v)
        v[i] = Vector{T}(M)
    end
    v
end

flatten_vector{T}(V::Vector{Vector{T}}) = vcat(V...)

function nest_vector{T}(v::Vector{T}, M, N)
    V = vector_alloc(T, M, N)
    for i in eachindex(V)
        copy!(V[i], v[(M*(i-1))+1:(M*i)])
    end
    V
end

function DEFAULT_NLSOLVE_MIRK(loss, u0)
    res = NLsolve.nlsolve(NLsolve.not_in_place(loss), u0)
    opt = res.zero
end
MIRK(order;dt=.2,nlsolve=DEFAULT_NLSOLVE_MIRK) = MIRK(order,dt,nlsolve)

