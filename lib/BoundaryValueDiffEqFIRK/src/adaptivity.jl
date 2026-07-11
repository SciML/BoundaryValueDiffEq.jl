"""
    interp_eval!(y::AbstractArray, cache::FIRKCacheExpand, t, mesh, mesh_dt)
    interp_eval!(y::AbstractArray, cache::FIRKCacheNested, t, mesh, mesh_dt)

After we construct an interpolant, we use interp_eval to evaluate it.
"""
@views function interp_eval!(
        y::AbstractArray, cache::FIRKCacheExpand{iip, T, DiffCacheNeeded},
        t, mesh, mesh_dt
    ) where {iip, T}
    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀.u) - 1) / (length(cache.y) - 1) # Cache length factor. We use a h corresponding to cache.y. Note that this assumes equidistributed mesh
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])

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
        K[:, jj] = cache.y[ctr_y + jj].du
    end

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, K) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    S_interpolate!(y, τ, S_coeffs)
    return y
end

@views function interp_eval!(
        y::AbstractArray, cache::FIRKCacheExpand{iip, T, NoDiffCacheNeeded},
        t, mesh, mesh_dt
    ) where {iip, T}
    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀.u) - 1) / (length(cache.y) - 1) # Cache length factor. We use a h corresponding to cache.y. Note that this assumes equidistributed mesh
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])

    (; f, M, stage, p, ITU) = cache
    (; q_coeff) = ITU

    K = safe_similar(cache.y[1], M, stage)

    ctr_y = (j - 1) * (stage + 1) + 1

    yᵢ = cache.y[ctr_y]
    yᵢ₊₁ = cache.y[ctr_y + stage + 1]

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
        K[:, jj] = cache.y[ctr_y + jj]
    end

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, K) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    S_interpolate!(y, τ, S_coeffs)
    return y
end

@views function interp_eval!(
        y::AbstractArray, cache::FIRKCacheNested{iip, T, DiffCacheNeeded},
        t, mesh, mesh_dt
    ) where {iip, T}
    (; f, ITU, nest_prob, alg) = cache
    (; q_coeff) = ITU

    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀.u) - 1) / (length(cache.y) - 1) # Cache length factor. We use a h corresponding to cache.y. Note that this assumes equidistributed mesh
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])

    nest_nlsolve_alg = __concrete_solve_algorithm(nest_prob, alg.nlsolve)
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
    nestprob_p[3:end] .= yᵢ

    _nestprob = remake(nest_prob, p = nestprob_p)
    nestsol = __solve(_nestprob, nest_nlsolve_alg; alg.nested_nlsolve_kwargs...)
    K = nestsol.u

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, K) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    S_interpolate!(y, τ, S_coeffs)
    return y
end

@views function interp_eval!(
        y::AbstractArray, cache::FIRKCacheNested{iip, T, NoDiffCacheNeeded},
        t, mesh, mesh_dt
    ) where {iip, T}
    (; f, ITU, nest_prob, alg) = cache
    (; q_coeff) = ITU

    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀.u) - 1) / (length(cache.y) - 1) # Cache length factor. We use a h corresponding to cache.y. Note that this assumes equidistributed mesh
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])

    nest_nlsolve_alg = __concrete_solve_algorithm(nest_prob, alg.nlsolve)
    nestprob_p = zeros(T, cache.M + 2)

    yᵢ = copy(cache.y[j])
    yᵢ₊₁ = copy(cache.y[j + 1])

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
    nestprob_p[3:end] .= yᵢ

    _nestprob = remake(nest_prob, p = nestprob_p)
    nestsol = __solve(_nestprob, nest_nlsolve_alg; alg.nested_nlsolve_kwargs...)
    K = nestsol.u

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, K) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    S_interpolate!(y, τ, S_coeffs)
    return y
end

function get_S_coeffs(h, yᵢ, yᵢ₊₁, dyᵢ, dyᵢ₊₁, ymid, dymid)
    M = length(yᵢ)
    A = s_constraints_block(h)
    coeffs = similar(yᵢ, M, 6)
    for k in 1:M
        bk = SVector(yᵢ[k], yᵢ₊₁[k], dyᵢ[k], dyᵢ₊₁[k], ymid[k], dymid[k])
        xk = A \ bk
        @view(coeffs[k, :]) .= xk
    end
    return coeffs
end

# S forward Interpolation
function __firk_matvec!(y, A, b)
    length(y) == size(A, 1) ||
        throw(DimensionMismatch("y has length $(length(y)), but A has $(size(A, 1)) rows"))
    length(b) == size(A, 2) ||
        throw(DimensionMismatch("b has length $(length(b)), but A has $(size(A, 2)) columns"))

    T = typeof(zero(eltype(A)) * zero(eltype(b)))
    for (y_index, row) in zip(eachindex(y), axes(A, 1))
        acc = zero(T)
        for (b_index, col) in zip(eachindex(b), axes(A, 2))
            @inbounds acc += A[row, col] * b[b_index]
        end
        @inbounds y[y_index] = acc
    end
    return y
end

function __firk_matvec(A, b)
    y = similar(b, typeof(zero(eltype(A)) * zero(eltype(b))), size(A, 1))
    return __firk_matvec!(y, A, b)
end

function S_interpolate!(y::AbstractArray, t, coeffs)
    ts = [t^(i - 1) for i in axes(coeffs, 2)]
    return __firk_matvec!(y, coeffs, ts)
end

function dS_interpolate!(dy::AbstractArray, t, S_coeffs)
    ts = zeros(promote_type(eltype(S_coeffs), typeof(t)), size(S_coeffs, 2))
    for i in 2:size(S_coeffs, 2)
        ts[i] = (i - 1) * t^(i - 2)
    end
    return __firk_matvec!(dy, S_coeffs, ts)
end

"""
    s_constraints_block(h)

Form the 6×6 quintic interpolation constraint block as an SMatrix.
The full `6M × 6M` constraint matrix is block-diagonal with `M` identical copies
of this block. See bvp5c paper.

Rows correspond to: value at t=0, value at t=h, value at t=h/2,
                     derivative at t=0, derivative at t=h, derivative at t=h/2.
Columns correspond to polynomial coefficients c₁..c₆ of p(t) = Σ cⱼ tʲ⁻¹.
"""
function s_constraints_block(h)
    t1, t2, t3 = zero(h), h, h / 2
    return SMatrix{6, 6}(
        one(h), one(h), one(h), zero(h), zero(h), zero(h),
        t1, t2, t3, one(h), one(h), one(h),
        t1^2, t2^2, t3^2, 2 * t1, 2 * t2, 2 * t3,
        t1^3, t2^3, t3^3, 3 * t1^2, 3 * t2^2, 3 * t3^2,
        t1^4, t2^4, t3^4, 4 * t1^3, 4 * t2^3, 4 * t3^3,
        t1^5, t2^5, t3^5, 5 * t1^4, 5 * t2^4, 5 * t3^4,
    )
end

"""
    mesh_selector!(cache::FIRKCacheExpand)
    mesh_selector!(cache::FIRKCacheNested)

Generate new mesh based on the defect.
"""
@views function mesh_selector!(
        cache::Union{
            FIRKCacheExpand{iip, T}, FIRKCacheNested{iip, T},
        }
    ) where {iip, T}
    (; order, defect, mesh, mesh_dt) = cache
    (abstol, _, _, _), _ = __split_kwargs(; cache.kwargs...)
    N = length(mesh)

    safety_factor = T(1.3)
    ρ = T(1.0) # Set rho=1 means mesh distribution will take place everytime.
    Nsub_star = 0
    Nsub_star_ub = 4 * (N - 1)
    Nsub_star_lb = N ÷ 2

    info = ReturnCode.Success

    ŝ = [maximum(abs, d) for d in defect.u]  # Broadcasting breaks GPU Compilation
    ŝ .= (ŝ ./ abstol) .^ (T(1) / (order + 1))
    r₁ = maximum(ŝ)
    r₂ = sum(ŝ)
    r₃ = r₂ / (N - 1)

    n_predict = round(Int, (safety_factor * r₂) + 1)
    n = N - 1
    n_ = T(0.1) * n
    n_predict = ifelse(abs((n_predict - n)) < n_, round(Int, n + n_), n_predict)

    if r₁ ≤ ρ * r₃
        Nsub_star = 2 * (N - 1)
        # Need to determine the too large threshold
        if Nsub_star > cache.alg.max_num_subintervals
            info = ReturnCode.Failure
            meshₒ = mesh
            mesh_dt₀ = mesh_dt
        else
            meshₒ = copy(mesh)
            mesh_dt₀ = copy(mesh_dt)
            half_mesh!(cache)
        end
    else
        Nsub_star = clamp(n_predict, Nsub_star_lb, Nsub_star_ub)
        if Nsub_star > cache.alg.max_num_subintervals
            # Mesh redistribution fails
            info = ReturnCode.Failure
            meshₒ = mesh
            mesh_dt₀ = mesh_dt
        else
            ŝ ./= mesh_dt
            meshₒ = copy(mesh)
            mesh_dt₀ = copy(mesh_dt)
            redistribute!(cache, Nsub_star, ŝ, meshₒ, mesh_dt₀)
        end
    end
    return meshₒ, mesh_dt₀, Nsub_star, info
end

"""
    redistribute!(cache::FIRKCacheExpand, Nsub_star, ŝ, mesh, mesh_dt)
    redistribute!(cache::FIRKCacheNested, Nsub_star, ŝ, mesh, mesh_dt)

Generate a new mesh based on the `ŝ`.
"""
function redistribute!(
        cache::Union{FIRKCacheExpand{iip, T}, FIRKCacheNested{iip, T}},
        Nsub_star, ŝ, mesh, mesh_dt
    ) where {iip, T}
    N = length(mesh) - 1
    ζ = sum(ŝ .* mesh_dt) / Nsub_star
    k, i = 1, 0
    resize!(cache.mesh, Nsub_star + 1)
    cache.mesh[1] = mesh[1]
    t = mesh[1]
    integral = T(0)
    while k ≤ N
        next_piece = ŝ[k] * (mesh[k + 1] - t)
        _int_next = integral + next_piece
        if _int_next > ζ
            cache.mesh[i + 2] = (ζ - integral) / ŝ[k] + t
            t = cache.mesh[i + 2]
            i += 1
            integral = T(0)
        else
            integral = _int_next
            t = mesh[k + 1]
            k += 1
        end
    end
    cache.mesh[end] = mesh[end]
    resize!(cache.mesh_dt, Nsub_star)
    diff!(cache.mesh_dt, cache.mesh)
    return cache
end

"""
    half_mesh!(mesh, mesh_dt)
    half_mesh!(cache::FIRKCacheExpand)
    half_mesh!(cache::FIRKCacheNested)

The input mesh has length of `n + 1`. Divide the original subinterval into two equal length
subinterval. The `mesh` and `mesh_dt` are modified in place.
"""
function half_mesh!(mesh::Vector{T}, mesh_dt::Vector{T}) where {T}
    n = length(mesh) - 1
    resize!(mesh, 2n + 1)
    resize!(mesh_dt, 2n)
    mesh[2n + 1] = mesh[n + 1]
    for i in (2n - 1):-2:1
        mesh[i] = mesh[(i + 1) ÷ 2]
        mesh_dt[i + 1] = mesh_dt[(i + 1) ÷ 2] / T(2)
    end
    @simd for i in (2n):-2:2
        mesh[i] = (mesh[i + 1] + mesh[i - 1]) / T(2)
        mesh_dt[i - 1] = mesh_dt[i]
    end
    return mesh, mesh_dt
end
function half_mesh!(cache::Union{FIRKCacheNested, FIRKCacheExpand})
    return half_mesh!(cache.mesh, cache.mesh_dt)
end

"""
    defect_estimate!(cache::FIRKCacheExpand)
    defect_estimate!(cache::FIRKCacheNested)

defect_estimate use the discrete solution approximation Y, plus stages of
the RK method in 'k_discrete', plus some new stages in 'k_interp' to construct
an interpolant
"""
@views function defect_estimate!(
        cache::FIRKCacheExpand{
            iip, T, DiffCacheNeeded,
        }
    ) where {iip, T}
    (; f, M, stage, mesh, mesh_dt, defect, ITU) = cache
    (; q_coeff, τ_star) = ITU

    ctr = 1
    K = zeros(eltype(cache.y[1].du), M, stage)
    for i in 1:(length(mesh) - 1)
        h = mesh_dt[i]

        # Load interpolation residual
        for j in 1:stage
            K[:, j] = cache.y[ctr + j].du
        end

        # Defect estimate from q(x) at y_i + τ* * h
        yᵢ₁ = copy(cache.y[ctr].du)
        yᵢ₂ = copy(yᵢ₁)
        z₁, z₁′ = eval_q(yᵢ₁, τ_star, h, q_coeff, K)
        if iip
            f(yᵢ₁, z₁, cache.p, mesh[i] + τ_star * h)
        else
            yᵢ₁ = f(z₁, cache.p, mesh[i] + τ_star * h)
        end
        yᵢ₁ .= (z₁′ .- yᵢ₁) ./ (abs.(yᵢ₁) .+ T(1))
        est₁ = maximum(abs, yᵢ₁)

        z₂, z₂′ = eval_q(yᵢ₂, (T(1) - τ_star), h, q_coeff, K)
        # Defect estimate from q(x) at y_i + (1-τ*) * h
        if iip
            f(yᵢ₂, z₂, cache.p, mesh[i] + (T(1) - τ_star) * h)
        else
            yᵢ₂ = f(z₂, cache.p, mesh[i] + (T(1) - τ_star) * h)
        end
        yᵢ₂ .= (z₂′ .- yᵢ₂) ./ (abs.(yᵢ₂) .+ T(1))
        est₂ = maximum(abs, yᵢ₂)

        defect.u[i] .= est₁ > est₂ ? yᵢ₁ : yᵢ₂
        ctr += stage + 1 # Advance one step
    end

    return maximum(Base.Fix1(maximum, abs), defect)
end

@views function defect_estimate!(
        cache::FIRKCacheExpand{
            iip, T, NoDiffCacheNeeded,
        }
    ) where {iip, T}
    (; f, M, stage, mesh, mesh_dt, defect, ITU) = cache
    (; q_coeff, τ_star) = ITU

    ctr = 1
    K = zeros(eltype(cache.y[1]), M, stage)
    for i in 1:(length(mesh) - 1)
        h = mesh_dt[i]

        # Load interpolation residual
        for j in 1:stage
            K[:, j] = cache.y[ctr + j]
        end

        # Defect estimate from q(x) at y_i + τ* * h
        yᵢ₁ = copy(cache.y[ctr])
        yᵢ₂ = copy(yᵢ₁)
        z₁, z₁′ = eval_q(yᵢ₁, τ_star, h, q_coeff, K)
        if iip
            f(yᵢ₁, z₁, cache.p, mesh[i] + τ_star * h)
        else
            yᵢ₁ = f(z₁, cache.p, mesh[i] + τ_star * h)
        end
        yᵢ₁ .= (z₁′ .- yᵢ₁) ./ (abs.(yᵢ₁) .+ T(1))
        est₁ = maximum(abs, yᵢ₁)

        z₂, z₂′ = eval_q(yᵢ₂, (T(1) - τ_star), h, q_coeff, K)
        # Defect estimate from q(x) at y_i + (1-τ*) * h
        if iip
            f(yᵢ₂, z₂, cache.p, mesh[i] + (T(1) - τ_star) * h)
        else
            yᵢ₂ = f(z₂, cache.p, mesh[i] + (T(1) - τ_star) * h)
        end
        yᵢ₂ .= (z₂′ .- yᵢ₂) ./ (abs.(yᵢ₂) .+ T(1))
        est₂ = maximum(abs, yᵢ₂)

        defect.u[i] .= est₁ > est₂ ? yᵢ₁ : yᵢ₂
        ctr += stage + 1 # Advance one step
    end

    return maximum(Base.Fix1(maximum, abs), defect)
end

@views function defect_estimate!(
        cache::FIRKCacheNested{
            iip, T, DiffCacheNeeded,
        }
    ) where {iip, T}
    (; f, mesh, mesh_dt, defect, ITU, nest_prob, alg) = cache
    (; q_coeff, τ_star) = ITU

    nlsolve_alg = __concrete_solve_algorithm(nest_prob, cache.alg.nlsolve)
    nestprob_p = zeros(T, cache.M + 2)

    for i in 1:(length(mesh) - 1)
        h = mesh_dt[i]
        yᵢ₁ = copy(cache.y[i].du)
        yᵢ₂ = copy(yᵢ₁)

        K = copy(cache.k_discrete[i].du)

        if minimum(abs.(K)) < 1.0e-2
            K = fill(one(eltype(K)), size(K))
        end

        nestprob_p[1] = mesh[i]
        nestprob_p[2] = mesh_dt[i]
        nestprob_p[3:end] .= yᵢ₁

        _nestprob = remake(nest_prob, p = nestprob_p)
        nest_sol = __solve(_nestprob, nlsolve_alg; alg.nested_nlsolve_kwargs...)

        # Defect estimate from q(x) at y_i + τ* * h
        z₁, z₁′ = eval_q(yᵢ₁, τ_star, h, q_coeff, nest_sol.u)
        if iip
            f(yᵢ₁, z₁, cache.p, mesh[i] + τ_star * h)
        else
            yᵢ₁ = f(z₁, cache.p, mesh[i] + τ_star * h)
        end
        yᵢ₁ .= (z₁′ .- yᵢ₁) ./ (abs.(yᵢ₁) .+ T(1))
        est₁ = maximum(abs, yᵢ₁)

        # Defect estimate from q(x) at y_i + (1-τ*) * h
        z₂, z₂′ = eval_q(yᵢ₂, (T(1) - τ_star), h, q_coeff, nest_sol.u)
        if iip
            f(yᵢ₂, z₂, cache.p, mesh[i] + (T(1) - τ_star) * h)
        else
            yᵢ₂ = f(z₂, cache.p, mesh[i] + (T(1) - τ_star) * h)
        end
        yᵢ₂ .= (z₂′ .- yᵢ₂) ./ (abs.(yᵢ₂) .+ T(1))
        est₂ = maximum(abs, yᵢ₂)

        defect.u[i] .= est₁ > est₂ ? yᵢ₁ : yᵢ₂
    end

    return maximum(Base.Fix1(maximum, abs), defect)
end

@views function defect_estimate!(
        cache::FIRKCacheNested{
            iip, T, NoDiffCacheNeeded,
        }
    ) where {iip, T}
    (; f, mesh, mesh_dt, defect, ITU, nest_prob, alg) = cache
    (; q_coeff, τ_star) = ITU

    nlsolve_alg = __concrete_solve_algorithm(nest_prob, cache.alg.nlsolve)
    nestprob_p = zeros(T, cache.M + 2)

    for i in 1:(length(mesh) - 1)
        h = mesh_dt[i]
        yᵢ₁ = copy(cache.y[i])
        yᵢ₂ = copy(yᵢ₁)

        K = copy(cache.k_discrete[i])

        if minimum(abs.(K)) < 1.0e-2
            K = fill(one(eltype(K)), size(K))
        end

        nestprob_p[1] = mesh[i]
        nestprob_p[2] = mesh_dt[i]
        nestprob_p[3:end] .= yᵢ₁

        _nestprob = remake(nest_prob, p = nestprob_p)
        nest_sol = __solve(_nestprob, nlsolve_alg; alg.nested_nlsolve_kwargs...)

        # Defect estimate from q(x) at y_i + τ* * h
        z₁, z₁′ = eval_q(yᵢ₁, τ_star, h, q_coeff, nest_sol.u)
        if iip
            f(yᵢ₁, z₁, cache.p, mesh[i] + τ_star * h)
        else
            yᵢ₁ = f(z₁, cache.p, mesh[i] + τ_star * h)
        end
        yᵢ₁ .= (z₁′ .- yᵢ₁) ./ (abs.(yᵢ₁) .+ T(1))
        est₁ = maximum(abs, yᵢ₁)

        # Defect estimate from q(x) at y_i + (1-τ*) * h
        z₂, z₂′ = eval_q(yᵢ₂, (T(1) - τ_star), h, q_coeff, nest_sol.u)
        if iip
            f(yᵢ₂, z₂, cache.p, mesh[i] + (T(1) - τ_star) * h)
        else
            yᵢ₂ = f(z₂, cache.p, mesh[i] + (T(1) - τ_star) * h)
        end
        yᵢ₂ .= (z₂′ .- yᵢ₂) ./ (abs.(yᵢ₂) .+ T(1))
        est₂ = maximum(abs, yᵢ₂)

        defect.u[i] .= est₁ > est₂ ? yᵢ₁ : yᵢ₂
    end

    return maximum(Base.Fix1(maximum, abs), defect)
end

function get_q_coeffs(A, ki, h)
    coeffs = __firk_matvec(A, ki)
    for i in axes(coeffs, 1)
        coeffs[i] = coeffs[i] / (h^(i - 1))
    end
    return coeffs
end

function apply_q(y_i, τ, h, coeffs)
    return y_i + sum(coeffs[i] * (τ * h)^(i) for i in axes(coeffs, 1))
end

function apply_q_prime(τ, h, coeffs)
    return sum(i * coeffs[i] * (τ * h)^(i - 1) for i in axes(coeffs, 1))
end

function eval_q(y_i::AbstractArray{T}, τ, h, A, K) where {T}
    M = size(K, 1)
    q = zeros(T, M)
    q′ = zeros(T, M)
    for i in 1:M
        ki = @view K[i, :]
        coeffs = get_q_coeffs(A, ki, h)
        q[i] = apply_q(y_i[i], τ, h, coeffs)
        q′[i] = apply_q_prime(τ, h, coeffs)
    end
    return q, q′
end
