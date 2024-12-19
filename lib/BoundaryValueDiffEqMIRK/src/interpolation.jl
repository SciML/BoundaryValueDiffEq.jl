# MIRK Interpolation
@concrete struct MIRKInterpolation <: AbstractDiffEqInterpolation
    t
    u
    cache
end

function DiffEqBase.interp_summary(interp::MIRKInterpolation)
    return "MIRK Order $(interp.cache.order) Interpolation"
end

function (id::MIRKInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::MIRKInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
    return
end

@inline function interpolation(tvals, id::MIRKInterpolation, idxs, deriv::D,
        p, continuity::Symbol = :left) where {D}
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

@inline function interpolation!(vals, tvals, id::MIRKInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
    (; t, cache) = id
    (; mesh, mesh_dt) = cache
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    for j in idx
        z = similar(id.u[1])
        interpolant!(z, id, cache, tvals[j], mesh, mesh_dt, deriv)
        vals[j] = z
    end
end

@inline function interpolation(tval::Number, id::MIRKInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
    z = similar(id.u[1])
    interpolant!(z, id, id.cache, tval, id.cache.mesh, id.cache.mesh_dt, deriv)
    return idxs !== nothing ? z[idxs] : z
end

@inline function interpolant!(
        z::AbstractArray, id, cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{0}})
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    w, w′ = interp_weights(τ, cache.alg)
    sum_stages!(z, id, cache, w, i)
end

@inline function interpolant!(dz::AbstractArray, id::MIRKInterpolation,
        cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{1}})
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    w, w′ = interp_weights(τ, cache.alg)
    z = similar(dz)
    sum_stages!(z, dz, id, cache, w, w′, i)
end

function sum_stages!(z::AbstractArray, id::MIRKInterpolation,
        cache::MIRKCache, w, i::Int, dt = cache.mesh_dt[i])
    (; stage, k_discrete, k_interp) = cache
    (; s_star) = cache.ITU
    z .= zero(z)
    __maybe_matmul!(z, k_discrete[i].du[:, 1:stage], w[1:stage])
    __maybe_matmul!(
        z, k_interp.u[i][:, 1:(s_star - stage)], w[(stage + 1):s_star], true, true)
    z .= z .* dt .+ id.u[i]

    return z
end

@views function sum_stages!(z, z′, id::MIRKInterpolation, cache::MIRKCache,
        w, w′, i::Int, dt = cache.mesh_dt[i])
    (; stage, k_discrete, k_interp) = cache
    (; s_star) = cache.ITU

    z .= zero(z)
    __maybe_matmul!(z, k_discrete[i].du[:, 1:stage], w[1:stage])
    __maybe_matmul!(
        z, k_interp.u[i][:, 1:(s_star - stage)], w[(stage + 1):s_star], true, true)
    z′ .= zero(z′)
    __maybe_matmul!(z′, k_discrete[i].du[:, 1:stage], w′[1:stage])
    __maybe_matmul!(
        z′, k_interp.u[i][:, 1:(s_star - stage)], w′[(stage + 1):s_star], true, true)
    z .= z .* dt[1] .+ id.u[i]

    return z, z′
end

@inline __build_interpolation(cache::MIRKCache, u::AbstractVector) = MIRKInterpolation(
    cache.mesh, u, cache)

# Intermidiate solution for evaluating boundry conditions
# basically simplified version of the interpolation for MIRK
function (s::EvalSol{C})(tval::Number) where {C <: MIRKCache}
    (; t, u, cache) = s
    (; alg, k_discrete) = cache
    stage = alg_stage(alg)
    # Quick handle for the case where tval is at the boundary
    (tval == t[1]) && return first(u)
    (tval == t[end]) && return last(u)
    z = zero(last(u))
    ii = interval(t, tval)
    dt = t[ii + 1] - t[ii]
    τ = (tval - t[ii]) / dt
    w, _ = interp_weights(τ, alg)
    __maybe_matmul!(z, k_discrete[ii].du[:, 1:stage], w[1:stage])
    z .= z .* dt .+ u[ii]
    return z
end
