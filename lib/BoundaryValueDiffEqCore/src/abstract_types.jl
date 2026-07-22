"""
    AbstractBoundaryValueDiffEqCache

Developer-facing abstract type for BoundaryValueDiffEq solver caches.

A solver package's `SciMLBase.__init` implementation returns a concrete subtype of this type.
The cache must expose a `prob` field so the default `SciMLBase.isinplace(cache)` method can
delegate to the boundary value problem, and the solver package must implement
`SciMLBase.solve!(cache)`. This is a versioned interface for solver implementations, not an
end-user extension point.

# Interface

- `SciMLBase.isinplace(cache)`: determines whether the cache's problem is in-place.
- `Base.eltype(cache)`: returns the cache element type when the solver needs one.

# Examples

```julia
using BoundaryValueDiffEqCore, SciMLBase

struct MyBVPCache{P} <: AbstractBoundaryValueDiffEqCache
    prob::P
end

SciMLBase.solve!(cache::MyBVPCache) = nothing
```
"""
abstract type AbstractBoundaryValueDiffEqCache end

function SciMLBase.isinplace(cache::AbstractBoundaryValueDiffEqCache)
    return SciMLBase.isinplace(cache.prob)
end
