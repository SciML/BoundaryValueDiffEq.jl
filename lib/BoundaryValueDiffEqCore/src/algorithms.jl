# Algorithms
"""
    AbstractBoundaryValueDiffEqAlgorithm

Developer-facing abstract type for boundary value problem algorithms.

Packages that implement a BoundaryValueDiffEq solver subtype this type and implement the
SciML solve interface for that algorithm. This is a versioned developer interface for solver
packages, not an end-user extension point. Solver users should select a concrete algorithm such
as `MIRK4()` or `Shooting()` rather than subtype this interface.

# Interface

For every concrete subtype `Alg`, define:

```julia
SciMLBase.__init(prob::SciMLBase.AbstractBVProblem, alg::Alg, args...; kwargs...)
```

The method must return a concrete [`AbstractBoundaryValueDiffEqCache`](@ref) whose `prob` field
is the supplied problem. It must accept and interpret the positional and keyword arguments that
the solver package supports. The matching cache type must implement `SciMLBase.solve!(cache)`.
`SciMLBase.solve(prob, alg, args...; kwargs...)` dispatches through these two methods in order;
`solve!` returns the solver result. Do not add methods for algorithms owned by another package.

# Examples

```julia
using BoundaryValueDiffEqCore, SciMLBase

struct MyBVPAlgorithm <: AbstractBoundaryValueDiffEqAlgorithm end
struct MyBVPCache{P} <: AbstractBoundaryValueDiffEqCache
    prob::P
end

SciMLBase.__init(prob::SciMLBase.AbstractBVProblem, ::MyBVPAlgorithm; kwargs...) =
    MyBVPCache(prob)
SciMLBase.solve!(cache::MyBVPCache) = cache.prob

SciMLBase.solve(prob, MyBVPAlgorithm()) # calls __init, then solve!
```

See the concrete solver packages in this repository for complete implementations.
"""
abstract type AbstractBoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end

@inline __nameof(::T) where {T} = nameof(T)
@inline __nameof(::Type{T}) where {T} = nameof(T)

## Disable the ugly verbose printing by default
@inline __modifier_text!(list, fieldname, field) = push!(list, "$fieldname = $(field)")
@inline __modifier_text!(list, fieldname, ::Nothing) = list
@inline __modifier_text!(list, fieldname, ::Missing) = list
@inline function __modifier_text!(list, fieldname, field::SciMLBase.AbstractODEAlgorithm)
    return push!(list, "$fieldname = $(__nameof(field))()")
end

function Base.show(io::IO, alg::AbstractBoundaryValueDiffEqAlgorithm)
    print(io, "$(__nameof(alg))(")
    modifiers = String[]
    for field in fieldnames(typeof(alg))
        __modifier_text!(modifiers, field, getfield(alg, field))
    end
    print(io, join(modifiers, ", "))
    return print(io, ")")
end

# Check what's the internal solver, nonlinear or optimization?
function __internal_solver(alg::AbstractBoundaryValueDiffEqAlgorithm)
    # We don't allow both `nlsolve` and `optimize` to be specified at the same time
    (isnothing(alg.nlsolve) && isnothing(alg.optimize)) &&
        error("Either `nlsolve` or `optimize` must be specified in the algorithm, but not both.")
    isnothing(alg.nlsolve) && return alg.optimize
    return isnothing(alg.optimize) && return alg.nlsolve
end
