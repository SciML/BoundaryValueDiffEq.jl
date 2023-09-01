const AA3 = AbstractArray{T, 3} where {T}  # TODO: Remove

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
