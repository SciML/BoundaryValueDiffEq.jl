@concrete struct MIRKCache{T}
    order::Int                 # The order of MIRK method
    stage::Int                 # The state of MIRK method
    M::Int
    in_size
    f!                         # FIXME: After supporting OOP functions
    bc!                        # FIXME: After supporting OOP functions
    prob
    problem_type
    p
    alg
    TU
    ITU
    # Everything below gets resized in adaptive methods
    mesh
    mesh_dt
    k_discrete
    k_interp
    y
    y₀
    residual
    # The following 2 caches are never resized
    fᵢ_cache
    fᵢ₂_cache
    defect
    new_stages
    kwargs
end

Base.eltype(::MIRKCache{T}) where {T} = T

"""
    expand_cache!(cache::MIRKCache)

After redistributing or halving the mesh, this function expands the required vectors to
match the length of the new mesh.
"""
function expand_cache!(cache::MIRKCache)
    Nₙ = length(cache.mesh)
    __append_similar!(cache.k_discrete, Nₙ - 1, cache.M)
    __append_similar!(cache.k_interp, Nₙ - 1, cache.M)
    __append_similar!(cache.y, Nₙ, cache.M)
    __append_similar!(cache.y₀, Nₙ, cache.M)
    __append_similar!(cache.residual, Nₙ, cache.M)
    __append_similar!(cache.defect, Nₙ - 1, cache.M)
    __append_similar!(cache.new_stages, Nₙ - 1, cache.M)
    return cache
end

function __append_similar!(x::AbstractVector{<:AbstractArray}, n, _)
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [similar(first(x)) for _ in 1:N])
    return x
end

function __append_similar!(x::AbstractVector{<:MaybeDiffCache}, n, M)
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    chunksize = pickchunksize(M * (N + length(x)))
    append!(x, [maybe_allocate_diffcache(first(x), chunksize) for _ in 1:N])
    return x
end
