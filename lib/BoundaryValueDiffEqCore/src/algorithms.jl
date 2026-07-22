# Algorithms
"""
    AbstractBoundaryValueDiffEqAlgorithm

Developer-facing abstract type for boundary value problem algorithms.

Packages that implement a BoundaryValueDiffEq solver subtype this type and implement the
SciML solve interface for that algorithm. In particular, provide an
`SciMLBase.__init(prob, alg; kwargs...)` method that constructs a solver cache and a
corresponding `SciMLBase.solve!` method for that cache. Solver users should select a concrete
algorithm such as `MIRK4()` or `Shooting()` rather than subtype this interface.

# Examples

```julia
using BoundaryValueDiffEqCore, SciMLBase

struct MyBVPAlgorithm <: AbstractBoundaryValueDiffEqAlgorithm end

function SciMLBase.__init(prob, ::MyBVPAlgorithm; kwargs...)
    # Construct and return the cache consumed by SciMLBase.solve!.
end
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
