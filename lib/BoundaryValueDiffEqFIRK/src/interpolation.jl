# FIRK Expand Interpolation
struct FIRKExpandInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
    cache
end

function DiffEqBase.interp_summary(interp::FIRKExpandInterpolation)
    return "FIRK Order $(interp.cache.order) Interpolation"
end

function (id::FIRKExpandInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::FIRKExpandInterpolation)(
        val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end

# FIRK Nested Interpolation
struct FIRKNestedInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
    cache
end

function DiffEqBase.interp_summary(interp::FIRKNestedInterpolation)
    return "FIRK Order $(interp.cache.order) Interpolation"
end

function (id::FIRKNestedInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::FIRKNestedInterpolation)(
        val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end

@inline function interpolation(tvals, id::FIRKNestedInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
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
        interp_eval!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt)
        vals[j] = idxs !== nothing ? z[idxs] : z
    end
    return DiffEqArray(vals, tvals)
end

@inline function interpolation!(vals, tvals, id::FIRKNestedInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
    (; t, cache) = id
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    for j in idx
        z = similar(cache.fᵢ₂_cache)
        interp_eval!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt)
        vals[j] = z
    end
end

@inline function interpolation(tval::Number, id::FIRKNestedInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
    z = similar(id.cache.fᵢ₂_cache)
    interp_eval!(z, id.cache, tval, id.cache.mesh, id.cache.mesh_dt)
    return idxs !== nothing ? z[idxs] : z
end

## Expanded
@inline function interpolation(tvals, id::FIRKExpandInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
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
        interp_eval!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt)
        vals[j] = idxs !== nothing ? z[idxs] : z
    end
    return DiffEqArray(vals, tvals)
end

@inline function interpolation!(vals, tvals, id::FIRKExpandInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
    (; t, cache) = id
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    for j in idx
        z = similar(cache.fᵢ₂_cache)
        interp_eval!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt)
        vals[j] = z
    end
end

@inline function interpolation(tval::Number, id::FIRKExpandInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
    z = similar(id.cache.fᵢ₂_cache)
    interp_eval!(z, id.cache, tval, id.cache.mesh, id.cache.mesh_dt)
    return idxs !== nothing ? z[idxs] : z
end

@inline __build_interpolation(cache::FIRKCacheExpand, u::AbstractVector) = FIRKExpandInterpolation(
    cache.mesh, u, cache)
@inline __build_interpolation(cache::FIRKCacheNested, u::AbstractVector) = FIRKNestedInterpolation(
    cache.mesh, u, cache)

# Intermidiate solution for evaluating boundry conditions
# basically simplified version of the interpolation for FIRK
# Expanded FIRK
function (s::EvalSol{C})(tval::Number) where {C <: FIRKCacheExpand}
    (; t, u, cache) = s
    (; f, alg, ITU, p) = cache
    (; q_coeff) = ITU
    stage = alg_stage(alg)
    # Quick handle for the case where tval is at the boundary
    (tval == t[1]) && return first(u)
    (tval == t[end]) && return last(u)
    K = __similar(first(u), length(first(u)), stage)
    j = interval(t, tval)
    ctr_y = (j - 1) * (stage + 1) + 1

    yᵢ = u[ctr_y]
    yᵢ₊₁ = u[ctr_y + stage + 1]

    if SciMLBase.isinplace(cache.prob)
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, p, t[j])
        f(dyᵢ₊₁, yᵢ₊₁, p, t[j + 1])
    else
        dyᵢ = f(yᵢ, p, t[j])
        dyᵢ₊₁ = f(yᵢ₊₁, p, t[j + 1])
    end

    # Load interpolation residual
    for jj in 1:stage
        K[:, jj] = u[ctr_y + jj]
    end
    h = t[j + 1] - t[j]
    τ = tval - t[j]

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, K) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    z = similar(yᵢ)

    S_interpolate!(z, τ, S_coeffs)
    return z
end

nodual_value(x) = x
nodual_value(x::Dual) = ForwardDiff.value(x)
nodual_value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

# Nested FIRK
function (s::EvalSol{C})(tval::Number) where {C <: FIRKCacheNested}
    (; t, u, cache) = s
    (; f, nest_prob, nest_tol, alg, mesh_dt, p, ITU) = cache
    (; q_coeff) = ITU
    stage = alg_stage(alg)
    # Quick handle for the case where tval is at the boundary
    (tval == t[1]) && return first(u)
    (tval == t[end]) && return last(u)
    j = interval(t, tval)
    h = t[j + 1] - t[j]
    τ = tval - t[j]

    nest_nlsolve_alg = __concrete_nonlinearsolve_algorithm(nest_prob, alg.nlsolve)
    nestprob_p = zeros(cache.M + 2)

    yᵢ = u[j]
    yᵢ₊₁ = u[j + 1]

    if SciMLBase.isinplace(cache.prob)
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, p, t[j])
        f(dyᵢ₊₁, yᵢ₊₁, p, t[j + 1])
    else
        dyᵢ = f(yᵢ, p, t[j])
        dyᵢ₊₁ = f(yᵢ₊₁, p, t[j + 1])
    end

    nestprob_p[1] = t[j]
    nestprob_p[2] = mesh_dt[j]
    nestprob_p[3:end] .= nodual_value(yᵢ)

    _nestprob = remake(nest_prob, p = nestprob_p)
    nestsol = __solve(_nestprob, nest_nlsolve_alg; abstol = nest_tol)
    K = nestsol.u

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, K) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)
    z = similar(yᵢ)
    S_interpolate!(z, τ, S_coeffs)
    return z
end
