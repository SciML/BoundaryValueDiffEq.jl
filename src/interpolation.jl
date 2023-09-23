struct MIRKInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
    cache
end

function (id::MIRKInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::MIRKInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end

@inline function interpolation(tvals,
    id::I,
    idxs,
    deriv::D,
    p,
    continuity::Symbol = :left) where {I, D}
    t = id.t
    u = id.u
    cache = id.cache
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
        tval = tvals[j]
        i = interval(t, tval)
        dt = t[i + 1] - t[i]
        θ = (tval - t[i]) / dt
        weights, _ = interp_weights(θ, cache.alg)
        z = zeros(cache.M)
        sum_stages!(z, cache, weights, i)
        vals[j] = copy(z)
    end
    DiffEqArray(vals, tvals)
end

@inline function interpolation!(vals,
    tvals,
    id::I,
    idxs,
    deriv::D,
    p,
    continuity::Symbol = :left) where {I, D}
    t = id.t
    cache = id.cache
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    for j in idx
        tval = tvals[j]
        i = interval(t, tval)
        dt = t[i] - t[i - 1]
        θ = (tval - t[i]) / dt
        weights, _ = interp_weights(θ, cache.alg)
        z = zeros(cache.M)
        sum_stages!(z, cache, weights, i)
        vals[j] = copy(z)
    end
end

@inline function interpolation(tval::Number,
    id::I,
    idxs,
    deriv::D,
    p,
    continuity::Symbol = :left) where {I, D}
    t = id.t
    cache = id.cache
    i = interval(t, tval)
    dt = t[i] - t[i - 1]
    θ = (tval - t[i]) / dt
    weights, _ = interp_weights(θ, cache.alg)
    z = zeros(cache.M)
    sum_stages!(z, cache, weights, i)
    val = copy(z)
    val
end
