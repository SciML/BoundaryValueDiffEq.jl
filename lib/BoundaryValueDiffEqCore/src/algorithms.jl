# Algorithms
"""
    AbstractBoundaryValueDiffEqAlgorithm

Abstract type for all boundary value problem algorithms.
"""
abstract type AbstractBoundaryValueDiffEqAlgorithm <: SciMLBase.AbstractBVPAlgorithm end

@inline __nameof(::T) where {T} = nameof(T)
@inline __nameof(::Type{T}) where {T} = nameof(T)

## Disable the ugly verbose printing by default
@inline __modifier_text!(list, fieldname, field) = push!(list, "$fieldname = $(field)")
@inline __modifier_text!(list, fieldname, ::Nothing) = list
@inline __modifier_text!(list, fieldname, ::Missing) = list
@inline function __modifier_text!(list, fieldname, field::SciMLBase.AbstractODEAlgorithm)
    push!(list, "$fieldname = $(__nameof(field))()")
end

function Base.show(io::IO, alg::AbstractBoundaryValueDiffEqAlgorithm)
    print(io, "$(__nameof(alg))(")
    modifiers = String[]
    for field in fieldnames(typeof(alg))
        __modifier_text!(modifiers, field, getfield(alg, field))
    end
    print(io, join(modifiers, ", "))
    print(io, ")")
end

# Check what's the internal solver, nonlinear or optimization?
function __internal_solver(alg::AbstractBoundaryValueDiffEqAlgorithm)
    # We don't allow both `nlsolve` and `optimize` to be specified at the same time
    (isnothing(alg.nlsolve) && isnothing(alg.optimize)) &&
        error("Either `nlsolve` or `optimize` must be specified in the algorithm, but not both.")
    isnothing(alg.nlsolve) && return alg.optimize
    isnothing(alg.optimize) && return alg.nlsolve
end
