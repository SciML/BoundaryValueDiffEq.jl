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
        interpolant!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt, deriv)
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
        interpolant!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt, deriv)
        vals[j] = z
    end
end

@inline function interpolation(tval::Number, id::FIRKNestedInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
    z = similar(id.cache.fᵢ₂_cache)
    interpolant!(z, id.cache, tval, id.cache.mesh, id.cache.mesh_dt, deriv)
    return idxs !== nothing ? z[idxs] : z
end

@inline function interpolant!(
        z::AbstractArray, cache::FIRKCacheNested{iip, T, diffcache, fit_parameters},
        t, mesh, mesh_dt, ::Type{Val{0}}) where {iip, T, diffcache, fit_parameters}
    (; f, ITU, nest_prob, alg) = cache
    (; q_coeff) = ITU

    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀) - 1) / (length(cache.y) - 1)
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])
    length_z = length(z)

    nest_nlsolve_alg = __concrete_nonlinearsolve_algorithm(nest_prob, alg.nlsolve)
    nestprob_p = zeros(T, cache.M + 2)

    yᵢ = copy(cache.y[j].du)
    yᵢ₊₁ = copy(cache.y[j + 1].du)

    if iip
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, cache.p, mesh[j])
        f(dyᵢ₊₁, yᵢ₊₁, cache.p, mesh[j + 1])
    else
        dyᵢ = f(yᵢ, cache.p, mesh[j])
        dyᵢ₊₁ = f(yᵢ₊₁, cache.p, mesh[j + 1])
    end

    nestprob_p[1] = mesh[j]
    nestprob_p[2] = mesh_dt[j]
    nestprob_p[3:end] .= ifelse.(fit_parameters, vcat(yᵢ, cache.p), yᵢ)

    _nestprob = remake(nest_prob, p = nestprob_p)
    nestsol = __solve(_nestprob, nest_nlsolve_alg; alg.nested_nlsolve_kwargs...)
    K = nestsol.u

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, K[1:length_z, :]) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    S_interpolate!(z, τ, S_coeffs)
end

@inline function interpolant!(
        dz::AbstractArray, cache::FIRKCacheNested{iip, T, diffcache, fit_parameters},
        t, mesh, mesh_dt, ::Type{Val{1}}) where {iip, T, diffcache, fit_parameters}
    (; f, ITU, nest_prob, alg) = cache
    (; q_coeff) = ITU

    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀) - 1) / (length(cache.y) - 1)
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])
    length_dz = length(dz)

    nest_nlsolve_alg = __concrete_nonlinearsolve_algorithm(nest_prob, alg.nlsolve)
    nestprob_p = zeros(T, cache.M + 2)

    yᵢ = copy(cache.y[j].du)
    yᵢ₊₁ = copy(cache.y[j + 1].du)

    if iip
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, cache.p, mesh[j])
        f(dyᵢ₊₁, yᵢ₊₁, cache.p, mesh[j + 1])
    else
        dyᵢ = f(yᵢ, cache.p, mesh[j])
        dyᵢ₊₁ = f(yᵢ₊₁, cache.p, mesh[j + 1])
    end

    nestprob_p[1] = mesh[j]
    nestprob_p[2] = mesh_dt[j]
    nestprob_p[3:end] .= ifelse.(fit_parameters, vcat(yᵢ, cache.p), yᵢ)

    _nestprob = remake(nest_prob, p = nestprob_p)
    nestsol = __solve(_nestprob, nest_nlsolve_alg; alg.nested_nlsolve_kwargs...)
    K = nestsol.u

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, @view(K[1:length_dz, :]))
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    dS_interpolate!(dz, τ, S_coeffs)
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
        interpolant!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt, deriv)
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
        interpolant!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt, deriv)
        vals[j] = z
    end
end

@inline function interpolation(tval::Number, id::FIRKExpandInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
    z = similar(id.cache.fᵢ₂_cache)
    interpolant!(z, id.cache, tval, id.cache.mesh, id.cache.mesh_dt, deriv)
    return idxs !== nothing ? z[idxs] : z
end

@inline function interpolant!(z::AbstractArray, cache::FIRKCacheExpand{iip},
        t, mesh, mesh_dt, ::Type{Val{0}}) where {iip}
    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀) - 1) / (length(cache.y) - 1)
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])
    length_z = length(z)

    (; f, M, stage, p, ITU) = cache
    (; q_coeff) = ITU

    K = safe_similar(cache.y[1].du, M, stage)

    ctr_y = (j - 1) * (stage + 1) + 1

    yᵢ = cache.y[ctr_y].du
    yᵢ₊₁ = cache.y[ctr_y + stage + 1].du

    if iip
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, p, mesh[j])
        f(dyᵢ₊₁, yᵢ₊₁, p, mesh[j + 1])
    else
        dyᵢ = f(yᵢ, p, mesh[j])
        dyᵢ₊₁ = f(yᵢ₊₁, p, mesh[j + 1])
    end

    # Load interpolation residual
    for jj in 1:stage
        K[1:length_z, jj] = cache.y[ctr_y + jj].du
    end

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, @view(K[1:length_z, :])) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    S_interpolate!(z, τ, S_coeffs)
end

@inline function interpolant!(dz::AbstractArray, cache::FIRKCacheExpand{iip},
        t, mesh, mesh_dt, ::Type{Val{1}}) where {iip}
    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀) - 1) / (length(cache.y) - 1)
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])
    length_dz = length(dz)

    (; f, M, stage, p, ITU) = cache
    (; q_coeff) = ITU

    K = safe_similar(cache.y[1].du, M, stage)

    ctr_y = (j - 1) * (stage + 1) + 1

    yᵢ = cache.y[ctr_y].du
    yᵢ₊₁ = cache.y[ctr_y + stage + 1].du

    if iip
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, p, mesh[j])
        f(dyᵢ₊₁, yᵢ₊₁, p, mesh[j + 1])
    else
        dyᵢ = f(yᵢ, p, mesh[j])
        dyᵢ₊₁ = f(yᵢ₊₁, p, mesh[j + 1])
    end

    # Load interpolation residual
    for jj in 1:stage
        K[1:length_dz, jj] = cache.y[ctr_y + jj].du
    end

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, @view(K[1:length_dz, :])) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    dS_interpolate!(dz, τ, S_coeffs)
end

@inline __build_interpolation(cache::FIRKCacheExpand,
    u::AbstractVector) = FIRKExpandInterpolation(cache.mesh, u, cache)
@inline __build_interpolation(cache::FIRKCacheNested,
    u::AbstractVector) = FIRKNestedInterpolation(cache.mesh, u, cache)

# Intermidiate solution for evaluating boundry conditions
# basically simplified version of the interpolation for FIRK
# Expanded FIRK
function (s::EvalSol{C})(tval::Number) where {C <: FIRKCacheExpand}
    (; t, u, cache) = s
    (; f, alg, ITU, mesh_dt, p) = cache
    (; q_coeff) = ITU
    stage = alg_stage(alg)
    # Quick handle for the case where tval is at the boundary
    (tval == t[1]) && return first(u)
    (tval == t[end]) && return last(u)
    K = safe_similar(first(u), length(first(u)), stage)
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
    h = mesh_dt[j]
    τ = tval - t[j]

    M = size(K, 1)
    z₁, z₁′ = similar(yᵢ), similar(yᵢ₊₁)
    for i in 1:M
        ki = @view K[i, :]
        coeffs = get_q_coeffs_interp(q_coeff, ki, h)
        z₁[i] = yᵢ[i] + sum(coeffs[ii] * (τ * h)^(ii) for ii in axes(coeffs, 1))
        z₁′[i] = sum(ii * coeffs[ii] * (τ * h)^(ii - 1) for ii in axes(coeffs, 1))
    end

    S_coeffs = get_S_coeffs_interp(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    z = similar(yᵢ)

    S_interpolate!(z, τ, S_coeffs)
    return z
end

function get_S_coeffs_interp(h, yᵢ, yᵢ₊₁, dyᵢ, dyᵢ₊₁, ymid, dymid)
    vals = vcat(yᵢ, yᵢ₊₁, dyᵢ, dyᵢ₊₁, ymid, dymid)
    M = length(yᵢ)
    A = s_constraints_interp(M, h)
    coeffs = reshape(A \ vals, 6, M)'
    return coeffs
end

function get_q_coeffs_interp(A, ki, h)
    coeffs = A * ki
    for i in axes(coeffs, 1)
        coeffs[i] = coeffs[i] / (h^(i - 1))
    end
    return coeffs
end

function s_constraints_interp(M, h)
    t = repeat([0.0, 1.0 * h, 0.5 * h, 0.0, 1.0 * h, 0.5 * h], M)
    A = zeros(6 * M, 6 * M)

    for i in 1:6
        row_start = (i - 1) * M + 1
        for k in 0:(M - 1)
            for j in 1:6
                A[row_start + k, j + k * 6] = t[i + k * 6]^(j - 1)
            end
        end
    end
    for i in 4:6
        row_start = (i - 1) * M + 1
        for k in 0:(M - 1)
            for j in 1:6
                A[row_start + k,
                    j + k * 6] = j == 1.0 ? 0.0 : (j - 1) * t[i + k * 6]^(j - 2)
            end
        end
    end

    return A
end

# Nested FIRK
function (s::EvalSol{C})(tval::Number) where {C <: FIRKCacheNested}
    (; t, u, cache) = s
    (; f, nest_prob, alg, mesh_dt, p, ITU) = cache
    (; q_coeff) = ITU
    stage = alg_stage(alg)
    # Quick handle for the case where tval is at the boundary
    (tval == t[1]) && return first(u)
    (tval == t[end]) && return last(u)
    j = interval(t, tval)
    h = mesh_dt[j]
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

    # TODO: Better initial guess or nestprob
    _nestprob = remake(nest_prob, p = nestprob_p, u0 = zeros(length(first(u)), stage))
    nestsol = __solve(_nestprob, nest_nlsolve_alg; alg.nested_nlsolve_kwargs...)
    K = nestsol.u

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, K) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)
    z = similar(yᵢ)
    S_interpolate!(z, τ, S_coeffs)
    return z
end
