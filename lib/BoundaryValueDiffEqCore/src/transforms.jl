# Time Domain Transformations for Infinite Time BVPs
# These transformations map semi-infinite or doubly-infinite domains to finite domains

"""
    TimeDomainTransform

Abstract type for time domain transformations used in infinite time BVPs.
Transformations map between the original (possibly infinite) time domain and a
finite transformed domain suitable for collocation methods.
"""
abstract type TimeDomainTransform end

"""
    IdentityTransform <: TimeDomainTransform

Identity transformation for finite time domains. No transformation is applied.
"""
struct IdentityTransform <: TimeDomainTransform end

"""
    SemiInfiniteTransform{T} <: TimeDomainTransform

Transformation for semi-infinite intervals [a, ∞) to [0, 1].

Uses the transformation τ = (t - a)/(1 + t - a), which maps:
- t = a → τ = 0
- t → ∞ → τ → 1

The inverse transformation is t = a + τ/(1 - τ).

# Fields
- `a::T`: The finite endpoint of the original domain
"""
struct SemiInfiniteTransform{T} <: TimeDomainTransform
    a::T
end

"""
    NegativeSemiInfiniteTransform{T} <: TimeDomainTransform

Transformation for semi-infinite intervals (-∞, b] to [0, 1].

Uses the transformation τ = 1/(1 - t + b), which maps:
- t → -∞ → τ → 0
- t = b → τ = 1

The inverse transformation is t = b + 1 - 1/τ.

# Fields
- `b::T`: The finite endpoint of the original domain
"""
struct NegativeSemiInfiniteTransform{T} <: TimeDomainTransform
    b::T
end

# ============================================================================
# Transformation Functions for SemiInfiniteTransform
# ============================================================================

"""
    τ_to_t(transform, τ)

Convert from transformed time τ to original time t.
"""
@inline function τ_to_t(::IdentityTransform, τ)
    return τ
end

@inline function τ_to_t(trans::SemiInfiniteTransform, τ)
    # t = a + τ/(1 - τ)
    # Handle τ = 1 case (t → ∞)
    if τ >= one(τ)
        return oftype(τ, Inf)
    end
    return trans.a + τ / (one(τ) - τ)
end

@inline function τ_to_t(trans::NegativeSemiInfiniteTransform, τ)
    # t = b + 1 - 1/τ
    # Handle τ = 0 case (t → -∞)
    if τ <= zero(τ)
        return oftype(τ, -Inf)
    end
    return trans.b + one(τ) - one(τ) / τ
end

"""
    t_to_τ(transform, t)

Convert from original time t to transformed time τ.
"""
@inline function t_to_τ(::IdentityTransform, t)
    return t
end

@inline function t_to_τ(trans::SemiInfiniteTransform, t)
    # τ = (t - a)/(1 + t - a)
    if isinf(t) && t > 0
        return one(typeof(t))
    end
    Δt = t - trans.a
    return Δt / (one(Δt) + Δt)
end

@inline function t_to_τ(trans::NegativeSemiInfiniteTransform, t)
    # τ = 1/(1 - t + b)
    if isinf(t) && t < 0
        return zero(typeof(t))
    end
    return one(typeof(t)) / (one(typeof(t)) - t + trans.b)
end

"""
    dtdτ(transform, τ)

Compute the derivative dt/dτ for the transformation.
This is needed to transform the ODE: du/dτ = f(t(τ), u) * dt/dτ
"""
@inline function dtdτ(::IdentityTransform, τ)
    return one(τ)
end

@inline function dtdτ(::SemiInfiniteTransform, τ)
    # t = a + τ/(1 - τ)
    # dt/dτ = 1/(1 - τ)²
    denom = one(τ) - τ
    return one(τ) / (denom * denom)
end

@inline function dtdτ(::NegativeSemiInfiniteTransform, τ)
    # t = b + 1 - 1/τ
    # dt/dτ = 1/τ²
    return one(τ) / (τ * τ)
end

"""
    is_identity_transform(transform)

Check if the transformation is an identity transformation.
"""
@inline is_identity_transform(::IdentityTransform) = true
@inline is_identity_transform(::TimeDomainTransform) = false

"""
    select_transform(t₀, t₁)

Select the appropriate transformation based on the time span.
Returns a tuple (transform, τ₀, τ₁) where τ₀ and τ₁ are the transformed endpoints.

For semi-infinite intervals, we use τ₁ < 1 to avoid the singularity at infinity.
The value τ_max = 0.99 corresponds to t ≈ 99 in original coordinates for a = 0.
"""
function select_transform(t₀::T, t₁::T) where {T}
    if isinf(t₁) && !isinf(t₀)
        # Semi-infinite interval [t₀, ∞)
        # Use τ_max < 1 to avoid singularity at τ = 1
        # τ = 0.99 corresponds to t = a + 0.99/(1 - 0.99) = a + 99
        τ_max = T(0.99)
        return SemiInfiniteTransform(t₀), zero(T), τ_max
    elseif isinf(t₀) && !isinf(t₁)
        # Semi-infinite interval (-∞, t₁]
        # Use τ_min > 0 to avoid singularity at τ = 0
        τ_min = T(0.01)
        return NegativeSemiInfiniteTransform(t₁), τ_min, one(T)
    elseif isinf(t₀) && isinf(t₁)
        throw(ArgumentError("Doubly-infinite intervals (-∞, ∞) are not yet supported"))
    else
        # Finite interval
        return IdentityTransform(), t₀, t₁
    end
end

"""
    original_tspan(transform, τ₀, τ₁)

Get the original time span from the transformed endpoints.
"""
@inline function original_tspan(::IdentityTransform, τ₀, τ₁)
    return (τ₀, τ₁)
end

@inline function original_tspan(trans::SemiInfiniteTransform, τ₀, τ₁)
    return (trans.a, oftype(trans.a, Inf))
end

@inline function original_tspan(trans::NegativeSemiInfiniteTransform, τ₀, τ₁)
    return (oftype(trans.b, -Inf), trans.b)
end

"""
    transform_mesh_point(transform, τ)

Transform a mesh point from τ (transformed domain) to t (original domain).
Alias for τ_to_t for clarity when working with mesh operations.
"""
@inline transform_mesh_point(trans::TimeDomainTransform, τ) = τ_to_t(trans, τ)
