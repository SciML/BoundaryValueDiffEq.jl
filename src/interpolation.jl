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

# FIXME: Fix the interpolation outside the tspan

@inline function interpolation(
        tvals, id::I, idxs, deriv::D, p, continuity::Symbol = :left) where {I, D}
    (; t, u, cache) = id
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
        interpolant!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt, deriv)
        vals[j] = idxs !== nothing ? z[idxs] : z
    end
    return DiffEqArray(vals, tvals)
end

@inline function interpolation!(
        vals, tvals, id::I, idxs, deriv::D, p, continuity::Symbol = :left) where {I, D}
    (; t, cache) = id
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    for j in idx
        z = similar(cache.fᵢ₂_cache)
        interpolant!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt, deriv)
        vals[j] = z
    end
end

@inline function interpolation(
        tval::Number, id::I, idxs, deriv::D, p, continuity::Symbol = :left) where {I, D}
    z = similar(id.cache.fᵢ₂_cache)
    interpolant!(z, id.cache, tval, id.cache.mesh, id.cache.mesh_dt, deriv)
    return idxs !== nothing ? z[idxs] : z
end

@inline function interpolant!(
        z::AbstractArray, cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{0}})
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    w, w′ = interp_weights(τ, cache.alg)
    sum_stages!(z, cache, w, i)
end

@inline function interpolant!(
        dz::AbstractArray, cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{1}})
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    w, w′ = interp_weights(τ, cache.alg)
    z = similar(dz)
    sum_stages!(z, dz, cache, w, w′, i)
end
