# Algorithms
"""
    BoundaryValueDiffEqAlgorithm

Abstract type for all boundary value problem algorithms.
"""
abstract type BoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end

@inline __nameof(::T) where {T} = nameof(T)
@inline __nameof(::Type{T}) where {T} = nameof(T)

## Disable the ugly verbose printing by default
@inline __modifier_text!(list, fieldname, field) = push!(list, "$fieldname = $(field)")
@inline __modifier_text!(list, fieldname, ::Nothing) = list
@inline __modifier_text!(list, fieldname, ::Missing) = list
@inline function __modifier_text!(list, fieldname, field::SciMLBase.AbstractODEAlgorithm)
    push!(list, "$fieldname = $(__nameof(field))()")
end

function Base.show(io::IO, alg::BoundaryValueDiffEqAlgorithm)
    print(io, "$(__nameof(alg))(")
    modifiers = String[]
    for field in fieldnames(typeof(alg))
        __modifier_text!(modifiers, field, getfield(alg, field))
    end
    print(io, join(modifiers, ", "))
    print(io, ")")
end
