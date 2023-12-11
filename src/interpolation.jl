struct RKInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
    cache
end

function DiffEqBase.interp_summary(interp::RKInterpolation)
    return "Runge-Kutta Order $(interp.cache.order) Interpolation"
end

function (id::RKInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::RKInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end

# FIXME: Fix the interpolation outside the tspan

@inline function interpolation(tvals, id::I, idxs, deriv::D, p,
        continuity::Symbol = :left) where {I, D}
    @unpack t, u, cache = id
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
        interp_eval!(z, id.cache,id.cache.ITU, tvals[j], id.cache.mesh, id.cache.mesh_dt)
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
        interp_eval!(z, id.cache,id.cache.ITU, tvals[j], id.cache.mesh, id.cache.mesh_dt)
        vals[j] = z
    end
end

@inline function interpolation(tval::Number, id::I, idxs, deriv::D, p,
        continuity::Symbol = :left) where {I, D}
    z = similar(id.cache.fᵢ₂_cache)
    interp_eval!(z, id.cache, id.cache.ITU, tval, id.cache.mesh, id.cache.mesh_dt)
    return z
end

"""
    get_ymid(yᵢ, coeffs, K, h)

Gets the interpolated middle value for a RK method, see bvp5c paper.
"""
function get_ymid(yᵢ, coeffs, K, h)
    res = copy(yᵢ)
    for i in axes(K, 2)
        res .+= h .* K[:, i] .* coeffs[i]
    end
    return res
end

"""
    s_constraints(M, h)

Form the quartic interpolation constraint matrix, see bvp5c paper.
"""
function s_constraints(M, h)
    t = vec(repeat([0.0, 1.0*h, 0.5*h, 0.0, 1.0*h, 0.5*h], 1, M))
    A = zeros(6 * M, 6 * M)
    for i in 1:6
        row_start = (i - 1) * M + 1
        if i <= 3
            for k = 0:M-1
                for j in 1:6
                    A[row_start+k, j+k*6] = t[i+k*6]^(j - 1)
                end
            end
        else
            for k = 0:M-1
                for j in 1:6
                    A[row_start+k, j+k*6] = j == 1.0 ? 0.0 : (j - 1) * t[i+k*6]^(j - 2)
                end
            end
        end
    end
    return A
end

"""
    get_s_coeffs(yᵢ, yᵢ₊₁, dyᵢ, dyᵢ₊₁, ymid)

Gets the coefficients for the (local) s(x) polynomial, see bvp5c paper.
"""
function get_s_coeffs(yᵢ, yᵢ₊₁, dyᵢ, dyᵢ₊₁, ymid)
    vals = vcat(yᵢ, yᵢ₊₁, dyᵢ, dyᵢ₊₁, ymid)
    M = length(yᵢ)
    A = s_constraints(M)
    coeffs = reshape(A \ vals, 5, M)'
    return coeffs
end

"""
    s_interpolate(t, coeffs)

Evaluate the s(x) interpolation, see bvp5c paper.
"""
function s_interpolate(t, coeffs)
    ts = [t^(i - 1) for i in axes(coeffs, 2)]
    return coeffs * ts
end
