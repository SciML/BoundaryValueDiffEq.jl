"""
    AbstractBoundaryValueDiffEqCache

Abstract Type for all BoundaryValueDiffEqCore Caches.

### Interface Functions

  - `SciMLBase.isinplace(cache)`: whether or not the solver is inplace.
  - `get_abstol(cache)`: get the `abstol` provided to the cache.
  - `Base.eltype(cache)`: get the element type of the cache.
"""
abstract type AbstractBoundaryValueDiffEqCache end

Base.eltype(cache::AbstractBoundaryValueDiffEqCache) = SciMLBase.eltype(cache.prob.u0)

function SciMLBase.isinplace(cache::AbstractBoundaryValueDiffEqCache)
    SciMLBase.isinplace(cache.prob)
end

function get_abstol(cache::AbstractBoundaryValueDiffEqCache)
    abstol = get(cache.kwargs, :abstol, nothing)
    return get_abstol(abstol, eltype(cache))
end

# default absolute tolerance
function get_abstol(::Nothing, ::Type{T}) where {T}
    return real(oneunit(T) * 1 // 10^6)
end
get_abstol(η, ::Type{T}) where {T} = real(η)
