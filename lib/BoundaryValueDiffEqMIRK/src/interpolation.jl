# MIRK Interpolation
@concrete struct MIRKInterpolation <: AbstractDiffEqInterpolation
    t
    u
    cache
end

function SciMLBase.interp_summary(interp::MIRKInterpolation)
    return "MIRK Order $(interp.cache.order) Interpolation"
end

function (id::MIRKInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::MIRKInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
    return
end

@inline function interpolation(
        tvals, id::MIRKInterpolation, idxs, deriv::D,
        p, continuity::Symbol = :left
    ) where {D}
    (; t, u, cache) = id
    (; mesh, mesh_dt) = cache
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    if idxs isa Number
        vals = Vector{eltype(first(u))}(undef, length(tvals))
    elseif idxs isa AbstractVector
        vals = Vector{Vector{eltype(first(u))}}(undef, length(tvals))
    else
        vals = Vector{eltype(u)}(undef, length(tvals))
    end

    for j in idx
        z = similar(cache.fᵢ₂_cache)
        interpolant!(z, id, cache, tvals[j], mesh, mesh_dt, deriv)
        vals[j] = idxs !== nothing ? z[idxs] : z
    end
    return DiffEqArray(vals, tvals)
end

@inline function interpolation!(
        vals, tvals, id::MIRKInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left
    ) where {D}
    (; t, cache) = id
    (; mesh, mesh_dt) = cache
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    for j in idx
        z = similar(id.u[1])
        interpolant!(z, id, cache, tvals[j], mesh, mesh_dt, deriv)
        vals[j] = z
    end
    return
end

@inline function interpolation(
        tval::Number, id::MIRKInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left
    ) where {D}
    z = similar(id.u[1])
    interpolant!(z, id, id.cache, tval, id.cache.mesh, id.cache.mesh_dt, deriv)
    return idxs !== nothing ? z[idxs] : z
end

@inline function interpolant!(
        z::AbstractArray, id::MIRKInterpolation, cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{0}}
    )
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    w, _ = interp_weights(τ, cache.alg)
    return sum_stages!(z, id, cache, w, i, τ, T)
end

@inline function interpolant!(
        dz::AbstractArray, id::MIRKInterpolation,
        cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{1}}
    )
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    _, w′ = interp_weights(τ, cache.alg)
    return sum_stages!(dz, id, cache, w′, i, τ, T)
end

@views function sum_stages!(
        z::AbstractArray, id::MIRKInterpolation,
        cache::MIRKCache{iip, T, use_both, DiffCacheNeeded},
        w, i::Int, τ, ::Type{Val{0}}
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp, M) = cache
    (; s_star) = cache.ITU
    dt = cache.mesh_dt[i]

    has_control = !isnothing(cache.prob.f.f_prototype)

    # state variables have their interpolation polynomials
    length_z = has_control ? length(cache.prob.f.f_prototype) : length(z)
    z .= zero(z)
    __maybe_matmul!(z[1:length_z], k_discrete[i].du[1:length_z, 1:stage], w[1:stage])
    __maybe_matmul!(
        z[1:length_z], k_interp.u[i][1:length_z, 1:(s_star - stage)],
        w[(stage + 1):s_star], true, true
    )

    # control variable just use linear interpolation
    if has_control
        inc = τ / dt .* (id.u[i + 1] .- id.u[i])
        copyto!(z, (length_z + 1):M, inc, (length_z + 1):M)
    end
    z .= z .* dt .+ id.u[i]

    return nothing
end
@views function sum_stages!(
        z::AbstractArray, id::MIRKInterpolation,
        cache::MIRKCache{iip, T, use_both, NoDiffCacheNeeded},
        w, i::Int, τ, ::Type{Val{0}}
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp, M) = cache
    (; s_star) = cache.ITU
    dt = cache.mesh_dt[i]

    has_control = !isnothing(cache.prob.f.f_prototype)
    length_z = has_control ? length(cache.prob.f.f_prototype) : length(z)

    z .= zero(z)
    __maybe_matmul!(z[1:length_z], k_discrete[i][1:length_z, 1:stage], w[1:stage])
    __maybe_matmul!(
        z[1:length_z], k_interp.u[i][1:length_z, 1:(s_star - stage)],
        w[(stage + 1):s_star], true, true
    )

    # control variable just use linear interpolation
    if has_control
        inc = τ / dt .* (id.u[i + 1] .- id.u[i])
        copyto!(z, (length_z + 1):M, inc, (length_z + 1):M)
    end

    z .= z .* dt .+ id.u[i]

    return nothing
end

@views function sum_stages!(
        z′, id::MIRKInterpolation, cache::MIRKCache{iip, T, use_both, DiffCacheNeeded},
        w′, i::Int, τ, ::Type{Val{1}}
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp, M) = cache
    (; s_star) = cache.ITU
    has_control = !isnothing(cache.prob.f.f_prototype)
    length_z = has_control ? length(cache.prob.f.f_prototype) : length(z′)

    z′ .= zero(z′)
    __maybe_matmul!(z′[1:length_z], k_discrete[i].du[1:length_z, 1:stage], w′[1:stage])
    __maybe_matmul!(
        z′[1:length_z], k_interp.u[i][1:length_z, 1:(s_star - stage)],
        w′[(stage + 1):s_star], true, true
    )

    # control variable just use linear interpolation
    if has_control
        inc = τ .* id.u[i + 1] .+ (1 - τ) .* id.u[i]
        copyto!(z′, (length_z + 1):M, inc, (length_z + 1):M)
    end

    return nothing
end
@views function sum_stages!(
        z′, id::MIRKInterpolation, cache::MIRKCache{iip, T, use_both, NoDiffCacheNeeded},
        w′, i::Int, τ, ::Type{Val{1}}
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp, M) = cache
    (; s_star) = cache.ITU
    has_control = !isnothing(cache.prob.f.f_prototype)
    length_z = has_control ? length(cache.prob.f.f_prototype) : length(z′)

    z′ .= zero(z′)
    __maybe_matmul!(z′[1:length_z], k_discrete[i][1:length_z, 1:stage], w′[1:stage])
    __maybe_matmul!(
        z′[1:length_z], k_interp.u[i][1:length_z, 1:(s_star - stage)],
        w′[(stage + 1):s_star], true, true
    )

    # control variable just use linear interpolation
    if has_control
        inc = τ .* id.u[i + 1] .+ (1 - τ) .* id.u[i]
        copyto!(z′, (length_z + 1):M, inc, (length_z + 1):M)
    end

    return nothing
end

@inline __build_interpolation(cache::MIRKCache, u::AbstractVector) = MIRKInterpolation(cache.mesh, u, cache)
