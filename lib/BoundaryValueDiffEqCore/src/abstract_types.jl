"""
    AbstractBoundaryValueDiffEqCache

Developer-facing abstract type for BoundaryValueDiffEq solver caches.

A solver package's `SciMLBase.__init` implementation returns a concrete subtype of this type.
This is a versioned developer interface for solver implementations, not an end-user extension
point.

# Interface

- Every cache must store the exact problem supplied to `SciMLBase.__init` in a field named
  `prob`. The default `SciMLBase.isinplace(cache)` delegates to that field.
- Every cache must implement `SciMLBase.solve!(cache)` and return the solver result expected by
  its algorithm.
- Define `Base.eltype(cache)` when the solver's implementation requires an element type.

The cache and its `solve!` method must be owned by the package that owns the corresponding
algorithm subtype. Do not extend another solver package's cache type.

# Examples

```julia
using BoundaryValueDiffEqCore, SciMLBase

struct MyBVPCache{P} <: AbstractBoundaryValueDiffEqCache
    prob::P
end

SciMLBase.solve!(cache::MyBVPCache) = cache.prob
```
"""
abstract type AbstractBoundaryValueDiffEqCache end

function SciMLBase.isinplace(cache::AbstractBoundaryValueDiffEqCache)
    return SciMLBase.isinplace(cache.prob)
end
