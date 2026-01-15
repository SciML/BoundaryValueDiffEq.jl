"""
    AbstractBoundaryValueDiffEqCache

Abstract Type for all BoundaryValueDiffEqCore Caches.

### Interface Functions

  - `SciMLBase.isinplace(cache)`: whether or not the solver is inplace.
  - `Base.eltype(cache)`: get the element type of the cache.
"""
abstract type AbstractBoundaryValueDiffEqCache end

function SciMLBase.isinplace(cache::AbstractBoundaryValueDiffEqCache)
    return SciMLBase.isinplace(cache.prob)
end
