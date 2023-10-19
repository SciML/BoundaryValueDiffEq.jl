struct MIRKInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
    cache
end

function DiffEqBase.interp_summary(interp::MIRKInterpolation)
    return "MIRK Order $(interp.cache.order) Interpolation"
end

function (id::MIRKInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::MIRKInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end

# FIXME: Fix the interpolation outside the tspan

@inline function interpolation(tvals, id::I, idxs, deriv::D, p,
    continuity::Symbol = :left) where {I, D}
    @unpack t, u, cache = id
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    if typeof(idxs) <: Number
        vals = Vector{eltype(first(u))}(undef, length(tvals))
    elseif typeof(idxs) <: AbstractVector
        vals = Vector{Vector{eltype(first(u))}}(undef, length(tvals))
    else
        vals = Vector{eltype(u)}(undef, length(tvals))
    end

    for j in idx
        z = similar(cache.fᵢ₂_cache)
        interp_eval!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt)
        vals[j] = z
    end
    return DiffEqArray(vals, tvals)
end

@inline function interpolation!(vals, tvals, id::I, idxs, deriv::D, p,
    continuity::Symbol = :left) where {I, D}
    @unpack t, cache = id
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    for j in idx
        z = similar(cache.fᵢ₂_cache)
        interp_eval!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt)
        vals[j] = z
    end
end

@inline function interpolation(tval::Number, id::I, idxs, deriv::D, p,
    continuity::Symbol = :left) where {I, D}
    z = similar(id.cache.fᵢ₂_cache, typeof(id.u[1][1]))
    interp_eval!(z, id.cache, id.u, tval, id.cache.mesh, id.cache.mesh_dt)
    return z
end
