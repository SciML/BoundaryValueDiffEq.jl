"""
    interp_eval!(y::AbstractArray, cache::RKCache, t)

After we construct an interpolant, we use interp_eval to evaluate it.
"""
@views function interp_eval!(y::AbstractArray, cache::RKCache, ITU::MIRKInterpTableau, t,
                             mesh, mesh_dt)
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    w, w′ = interp_weights(τ, cache.alg)
    sum_stages!(y, cache, w, i)
    return y
end

function interp_eval!(y::AbstractArray, i::Int, cache::RKCache, ITU::MIRKInterpTableau, t,
                      mesh, mesh_dt)
    interp_eval!(y[i], cache, ITU, t, mesh, mesh_dt)
end

function get_S_coeffs(h, yᵢ, yᵢ₊₁, dyᵢ, dyᵢ₊₁, ymid, dymid)
    vals = vcat(yᵢ, yᵢ₊₁, dyᵢ, dyᵢ₊₁, ymid, dymid)
    M = length(yᵢ)
    A = s_constraints(M, h)
    coeffs = reshape(A \ vals, 6, M)'
    return coeffs
end

# S forward Interpolation
function S_interpolate(t, coeffs)
    ts = [t^(i - 1) for i in axes(coeffs, 2)]
    return coeffs * ts
end

function dS_interpolate(t, S_coeffs)
    ts = zeros(size(S_coeffs, 2))
    for i in 2:size(S_coeffs, 2)
        ts[i] = (i - 1) * t^(i - 2)
    end
    return S_coeffs * ts
end

@views function interp_eval!(y::AbstractArray, i::Int, cache::RKCache,
                             ITU::RKInterpTableau{false},
                             t,
                             mesh, mesh_dt)

    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀)-1) / (length(cache.y)-1) # Cache length factor. We use a h corresponding to cache.y. Note that this assumes equidistributed mesh
    if lf > 1 
        h *=lf
    end
    τ = (t - mesh[j]) / h

    @unpack f, M, p = cache
    @unpack c, a, b = cache.TU
    @unpack q_coeff, stage = ITU

    K = zeros(eltype(cache.y[1].du), M, stage)

    ctr_y0 = (i - 1) * (ITU.stage + 1) + 1
    ctr_y = (j - 1) * (ITU.stage + 1) + 1

    yᵢ = cache.y[ctr_y].du
    yᵢ₊₁ = cache.y[ctr_y + ITU.stage + 1].du

    dyᵢ = copy(yᵢ)
    dyᵢ₊₁ = copy(yᵢ₊₁)

    f(dyᵢ, yᵢ, cache.p, mesh[j])
    f(dyᵢ₊₁, yᵢ₊₁, cache.p, mesh[j + 1])
    # Load interpolation residual
    for jj in 1:stage
        K[:, jj] = cache.y[ctr_y + jj].du
    end

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, K) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    y[ctr_y0] = S_interpolate(τ * h, S_coeffs)
    if ctr_y0 < length(y)
        for (k, ci) in enumerate(c)
            y[ctr_y0 + k] = dS_interpolate(τ + (1 - τ) * ci, S_coeffs)
        end
    end

    return y[ctr_y0]
end

@views function interp_eval!(y::AbstractArray, i::Int, cache::RKCache,
                             ITU::RKInterpTableau{true},
                             t,
                             mesh, mesh_dt)
    j = interval(mesh, t)
    h = mesh_dt[j]
    τ = (t - mesh[j]) / h

    @unpack f, M, p = cache
    @unpack c, a, b = cache.TU
    @unpack q_coeff, stage = ITU

    K = zeros(eltype(cache.y[1].du), M, stage)

    yᵢ = cache.y[j].du
    yᵢ₊₁ = cache.y[j + 1].du

    dyᵢ = copy(yᵢ)
    dyᵢ₊₁ = copy(yᵢ₊₁)

    f(dyᵢ, yᵢ, cache.p, mesh[j])
    f(dyᵢ₊₁, yᵢ₊₁, cache.p, mesh[j + 1])

    # Load interpolation residual
    prob = NonlinearProblem((K, p) -> FIRK_nlsolve(K, f, a, c, yᵢ, h, mesh[j], stage,
                                                   p), fill(1.0, size(K)), p)
    sol = solve(prob, NewtonRaphson(), reltol = 1e-4, maxiters = 10)
    K .= sol.u

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, K) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    y[i] = S_interpolate(τ, S_coeffs)

    return y[i]
end

"""
    interval(mesh, t)

Find the interval that `t` belongs to in `mesh`. Assumes that `mesh` is sorted.
"""
function interval(mesh, t)
    if t in mesh
        id = findfirst(isequal(t), mesh)

        return clamp(id, 1, length(mesh) - 1)
    else
        return clamp(searchsortedfirst(mesh, t) - 1, 1, length(mesh) - 1)
    end
end

"""
    mesh_selector!(cache::RKCache{T})

Generate new mesh based on the defect.
"""
@views function mesh_selector!(cache::RKCache{iip, T}) where {iip, T}
    @unpack M, order, defect, mesh, mesh_dt = cache
    (_, MxNsub, abstol, _, _), kwargs = __split_mirk_kwargs(; cache.kwargs...)
    N = length(cache.mesh)

    safety_factor = T(1.3)
    ρ = T(1.0) # Set rho=1 means mesh distribution will take place everytime.
    Nsub_star = 0
    Nsub_star_ub = 4 * (N - 1)
    Nsub_star_lb = N ÷ 2

    info = ReturnCode.Success

    ŝ = [maximum(abs, d) for d in defect]  # Broadcasting breaks GPU Compilation
    ŝ .= (ŝ ./ abstol) .^ (T(1) / (order + 1))
    r₁ = maximum(ŝ)
    r₂ = sum(ŝ)
    r₃ = r₂ / (N - 1)

    n_predict = round(Int, (safety_factor * r₂) + 1)
    n = N - 1
    n_ = T(0.1) * n
    n_predict = ifelse(abs((n_predict - n)) < n_, round(Int, n + n_), n_predict)

    if r₁ ≤ ρ * r₂
        Nsub_star = 2 * (N - 1)
        if Nsub_star > MxNsub # Need to determine the too large threshold
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
        if Nsub_star > MxNsub
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
    redistribute!(cache::RKCache{T}, Nsub_star, ŝ, mesh, mesh_dt) where {T}

Generate a new mesh based on the `ŝ`.
"""
function redistribute!(cache::RKCache{T}, Nsub_star, ŝ, mesh, mesh_dt) where {T}
    N = length(mesh)
    ζ = sum(ŝ .* mesh_dt) / Nsub_star
    k, i = 1, 0
    append!(cache.mesh, Nsub_star + 1 - N)
    cache.mesh[1] = mesh[1]
    t = mesh[1]
    integral = T(0)
    while k ≤ N - 1
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
    append!(cache.mesh_dt, Nsub_star - N)
    diff!(cache.mesh_dt, cache.mesh)
    return cache
end

"""
    half_mesh!(mesh, mesh_dt)
    half_mesh!(cache::RKCache)

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
half_mesh!(cache::RKCache) = half_mesh!(cache.mesh, cache.mesh_dt)

"""
    defect_estimate!(cache::RKCache{T})

defect_estimate use the discrete solution approximation Y, plus stages of
the RK method in 'k_discrete', plus some new stages in 'k_interp' to construct
an interpolant
"""
@views function defect_estimate!(cache::RKCache{iip, T}, TU::MIRKTableau) where {iip, T}
    @unpack M, stage, f, alg, mesh, mesh_dt, defect = cache
    @unpack s_star, τ_star = cache.ITU

    # Evaluate at the first sample point
    w₁, w₁′ = interp_weights(τ_star, alg)
    # Evaluate at the second sample point
    w₂, w₂′ = interp_weights(T(1) - τ_star, alg)

    interp_setup!(cache)

    for i in 1:(length(mesh) - 1)
        dt = mesh_dt[i]

        z, z′ = sum_stages!(cache, w₁, w₁′, i)
        if iip
            yᵢ₁ = cache.y[i].du
            f(yᵢ₁, z, cache.p, mesh[i] + τ_star * dt)
        else
            yᵢ₁ = f(z, cache.p, mesh[i] + τ_star * dt)
        end
        yᵢ₁ .= (z′ .- yᵢ₁) ./ (abs.(yᵢ₁) .+ T(1))
        est₁ = maximum(abs, yᵢ₁)

        z, z′ = sum_stages!(cache, w₂, w₂′, i)
        if iip
            yᵢ₂ = cache.y[i + 1].du
            f(yᵢ₂, z, cache.p, mesh[i] + (T(1) - τ_star) * dt)
        else
            yᵢ₂ = f(z, cache.p, mesh[i] + (T(1) - τ_star) * dt)
        end
        yᵢ₂ .= (z′ .- yᵢ₂) ./ (abs.(yᵢ₂) .+ T(1))
        est₂ = maximum(abs, yᵢ₂)

        defect[i] .= est₁ > est₂ ? yᵢ₁ : yᵢ₂
    end
    return maximum(Base.Fix1(maximum, abs), defect)
end

function get_q_coeffs(A, ki, h)
    coeffs = A * ki
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

function eval_q(y_i, τ, h, A, K)
    M = size(K, 1)
    q = zeros(M)
    q′ = zeros(M)
    for i in 1:M
        ki = @view K[i, :]
        coeffs = get_q_coeffs(A, ki, h)
        q[i] = apply_q(y_i[i], τ, h, coeffs)
        q′[i] = apply_q_prime(τ, h, coeffs)
    end
    return q, q′
end

@views function defect_estimate!(cache::RKCache{iip, T},
                                 TU::RKTableau{false}) where {iip, T}
    @unpack f, M, stage, mesh, mesh_dt, defect = cache
    @unpack q_coeff, τ_star = cache.ITU

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

        defect[i] .= est₁ > est₂ ? yᵢ₁ : yᵢ₂
        ctr += stage + 1 # Advance one step
    end

    return maximum(Base.Fix1(maximum, abs), defect)
end

@views function defect_estimate!(cache::RKCache{iip, T}, TU::RKTableau{true}) where {iip, T}
    @unpack f, M, stage, mesh, mesh_dt, defect = cache
    @unpack a, c = cache.TU
    @unpack q_coeff, τ_star = cache.ITU

    K = zeros(eltype(cache.y[1].du), M, stage)
    for i in 1:(length(mesh) - 1) # TODO: add backward differences for last point, easy if equidistributed
        h = mesh_dt[i]
        yᵢ₁ = cache.y[i].du
        yᵢ₂ = copy(yᵢ₁)

        prob = NonlinearProblem((K, p) -> FIRK_nlsolve(K, f, a, c, yᵢ₁, h, mesh[i], stage,
                                                       p), fill(1.0, size(K)), cache.p)
        sol = solve(prob, NewtonRaphson(), reltol = 1e-4, maxiters = 10)
        K .= sol.u

        # Defect estimate from q(x) at y_i + τ* * h
        z₁, z₁′ = eval_q(yᵢ₁, τ_star, h, q_coeff, K)
        if iip
            f(yᵢ₁, z₁, cache.p, mesh[i] + τ_star * h)
        else
            yᵢ₁ = f(z₁, cache.p, mesh[i] + τ_star * h)
        end
        yᵢ₁ .= (z₁′ .- yᵢ₁) ./ (abs.(yᵢ₁) .+ T(1))
        est₁ = maximum(abs, yᵢ₁)

        # Defect estimate from q(x) at y_i + (1-τ*) * h
        z₂, z₂′ = eval_q(yᵢ₂, (T(1) - τ_star), h, q_coeff, K)
        if iip
            f(yᵢ₂, z₂, cache.p, mesh[i] + (T(1) - τ_star) * h)
        else
            yᵢ₂ = f(z₂, cache.p, mesh[i] + (T(1) - τ_star) * h)
        end
        yᵢ₂ .= (z₂′ .- yᵢ₂) ./ (abs.(yᵢ₂) .+ T(1))
        est₂ = maximum(abs, yᵢ₂)

        defect[i] .= est₁ > est₂ ? yᵢ₁ : yᵢ₂
    end

    return maximum(Base.Fix1(maximum, abs), defect)
end

"""
    interp_setup!(cache::RKCache)

`interp_setup!` prepare the extra stages in ki_interp for interpolant construction.
Here, the ki_interp is the stages in one subinterval.
"""
@views function interp_setup!(cache::RKCache{iip, T}) where {iip, T}
    @unpack x_star, s_star, c_star, v_star = cache.ITU
    @unpack k_interp, k_discrete, f, stage, new_stages, y, p, mesh, mesh_dt = cache

    for r in 1:(s_star - stage)
        idx₁ = ((1:stage) .- 1) .* (s_star - stage) .+ r
        idx₂ = ((1:(r - 1)) .+ stage .- 1) .* (s_star - stage) .+ r
        for j in eachindex(k_discrete)
            __maybe_matmul!(new_stages[j], k_discrete[j].du[:, 1:stage], x_star[idx₁])
        end
        if r > 1
            for j in eachindex(k_interp)
                __maybe_matmul!(new_stages[j], k_interp[j][:, 1:(r - 1)], x_star[idx₂],
                                T(1), T(1))
            end
        end
        for i in eachindex(new_stages)
            new_stages[i] .= new_stages[i] .* mesh_dt[i] .+
                             (1 - v_star[r]) .* vec(y[i].du) .+
                             v_star[r] .* vec(y[i + 1].du)
            if iip
                f(k_interp[i][:, r], new_stages[i], p, mesh[i] + c_star[r] * mesh_dt[i])
            else
                k_interp[i][:, r] .= f(new_stages[i], p, mesh[i] + c_star[r] * mesh_dt[i])
            end
        end
    end

    return k_interp
end

"""
    sum_stages!(cache::RKCache, w, w′, i::Int)

sum_stages add the discrete solution, RK method stages and extra stages to construct interpolant.
"""
function sum_stages!(cache::RKCache, w, w′, i::Int, dt = cache.mesh_dt[i])
    sum_stages!(cache.fᵢ_cache.du, cache.fᵢ₂_cache, cache, w, w′, i, dt)
end

function sum_stages!(z, cache::RKCache, w, i::Int, dt = cache.mesh_dt[i])
    @unpack M, stage, mesh, k_discrete, k_interp, mesh_dt = cache
    @unpack s_star = cache.ITU

    z .= zero(z)
    __maybe_matmul!(z, k_discrete[i].du[:, 1:stage], w[1:stage])
    __maybe_matmul!(z, k_interp[i][:, 1:(s_star - stage)],
                    w[(stage + 1):s_star], true, true)
    z .= z .* dt .+ cache.y₀[i]

    return z
end

@views function sum_stages!(z, z′, cache::RKCache, w, w′, i::Int, dt = cache.mesh_dt[i])
    @unpack M, stage, mesh, k_discrete, k_interp, mesh_dt = cache
    @unpack s_star = cache.ITU

    z .= zero(z)
    __maybe_matmul!(z, k_discrete[i].du[:, 1:stage], w[1:stage])
    __maybe_matmul!(z, k_interp[i][:, 1:(s_star - stage)],
                    w[(stage + 1):s_star], true, true)
    z′ .= zero(z′)
    __maybe_matmul!(z′, k_discrete[i].du[:, 1:stage], w′[1:stage])
    __maybe_matmul!(z′, k_interp[i][:, 1:(s_star - stage)],
                    w′[(stage + 1):s_star], true, true)
    z .= z .* dt[1] .+ cache.y₀[i]

    return z, z′
end

for order in (2, 3, 4, 5, 6)
    alg = Symbol("MIRK$(order)")
    @eval begin """
                    interp_weights(τ, alg)

                interp_weights: solver-specified interpolation weights and its first derivative
                """
    function interp_weights(τ::T, ::$(alg)) where {T}
        if $(order == 2)
            w = [0, τ * (1 - τ / 2), τ^2 / 2]

            #     Derivative polynomials.

            wp = [0, 1 - τ, τ]
        elseif $(order == 3)
            w = [τ / 4.0 * (2.0 * τ^2 - 5.0 * τ + 4.0),
                -3.0 / 4.0 * τ^2 * (2.0 * τ - 3.0),
                τ^2 * (τ - 1.0)]

            #     Derivative polynomials.

            wp = [3.0 / 2.0 * (τ - 2.0 / 3.0) * (τ - 1.0),
                -9.0 / 2.0 * τ * (τ - 1.0),
                3.0 * τ * (τ - 2.0 / 3.0)]
        elseif $(order == 4)
            t2 = τ * τ
            tm1 = τ - 1.0
            t4m3 = τ * 4.0 - 3.0
            t2m1 = τ * 2.0 - 1.0

            w = [-τ * (2.0 * τ - 3.0) * (2.0 * t2 - 3.0 * τ + 2.0) / 6.0,
                t2 * (12.0 * t2 - 20.0 * τ + 9.0) / 6.0,
                2.0 * t2 * (6.0 * t2 - 14.0 * τ + 9.0) / 3.0,
                -16.0 * t2 * tm1 * tm1 / 3.0]

            #   Derivative polynomials

            wp = [-tm1 * t4m3 * t2m1 / 3.0,
                τ * t2m1 * t4m3,
                4.0 * τ * t4m3 * tm1,
                -32.0 * τ * t2m1 * tm1 / 3.0]
        elseif $(order == 5)
            w = [
                τ * (22464.0 - 83910.0 * τ + 143041.0 * τ^2 - 113808.0 * τ^3 +
                 33256.0 * τ^4) / 22464.0,
                τ^2 * (-2418.0 + 12303.0 * τ - 19512.0 * τ^2 + 10904.0 * τ^3) /
                3360.0,
                -8 / 81 * τ^2 * (-78.0 + 209.0 * τ - 204.0 * τ^2 + 8.0 * τ^3),
                -25 / 1134 * τ^2 *
                (-390.0 + 1045.0 * τ - 1020.0 * τ^2 + 328.0 * τ^3),
                -25 / 5184 * τ^2 *
                (390.0 + 255.0 * τ - 1680.0 * τ^2 + 2072.0 * τ^3),
                279841 / 168480 * τ^2 *
                (-6.0 + 21.0 * τ - 24.0 * τ^2 + 8.0 * τ^3)]

            #   Derivative polynomials

            wp = [
                1.0 - 13985 // 1872 * τ + 143041 // 7488 * τ^2 -
                2371 // 117 * τ^3 +
                20785 // 2808 * τ^4,
                -403 // 280 * τ + 12303 // 1120 * τ^2 - 813 // 35 * τ^3 +
                1363 // 84 * τ^4,
                416 // 27 * τ - 1672 // 27 * τ^2 + 2176 // 27 * τ^3 -
                320 // 81 * τ^4,
                3250 // 189 * τ - 26125 // 378 * τ^2 + 17000 // 189 * τ^3 -
                20500 // 567 * τ^4,
                -1625 // 432 * τ - 2125 // 576 * τ^2 + 875 // 27 * τ^3 -
                32375 // 648 * τ^4,
                -279841 // 14040 * τ + 1958887 // 18720 * τ^2 -
                279841 // 1755 * τ^3 +
                279841 // 4212 * τ^4]
        elseif $(order == 6)
            w = [
                τ - 28607 // 7434 * τ^2 - 166210 // 33453 * τ^3 +
                334780 // 11151 * τ^4 -
                1911296 // 55755 * τ^5 + 406528 // 33453 * τ^6,
                777 // 590 * τ^2 - 2534158 // 234171 * τ^3 +
                2088580 // 78057 * τ^4 -
                10479104 // 390285 * τ^5 + 11328512 // 1170855 * τ^6,
                -1008 // 59 * τ^2 + 222176 // 1593 * τ^3 - 180032 // 531 * τ^4 +
                876544 // 2655 * τ^5 - 180224 // 1593 * τ^6,
                -1008 // 59 * τ^2 + 222176 // 1593 * τ^3 - 180032 // 531 * τ^4 +
                876544 // 2655 * τ^5 - 180224 // 1593 * τ^6,
                -378 // 59 * τ^2 + 27772 // 531 * τ^3 - 22504 // 177 * τ^4 +
                109568 // 885 * τ^5 - 22528 // 531 * τ^6,
                -95232 // 413 * τ^2 + 62384128 // 33453 * τ^3 -
                49429504 // 11151 * τ^4 +
                46759936 // 11151 * τ^5 - 46661632 // 33453 * τ^6,
                896 // 5 * τ^2 - 4352 // 3 * τ^3 + 3456 * τ^4 -
                16384 // 5 * τ^5 +
                16384 // 15 * τ^6,
                50176 // 531 * τ^2 - 179554304 // 234171 * τ^3 +
                143363072 // 78057 * τ^4 -
                136675328 // 78057 * τ^5 + 137363456 // 234171 * τ^6,
                16384 // 441 * τ^3 - 16384 // 147 * τ^4 + 16384 // 147 * τ^5 -
                16384 // 441 * τ^6]

            #     Derivative polynomials.

            wp = [
                1 - 28607 // 3717 * τ - 166210 // 11151 * τ^2 +
                1339120 // 11151 * τ^3 -
                1911296 // 11151 * τ^4 + 813056 // 11151 * τ^5,
                777 // 295 * τ - 2534158 // 78057 * τ^2 + 8354320 // 78057 * τ^3 -
                10479104 // 78057 * τ^4 + 22657024 // 390285 * τ^5,
                -2016 // 59 * τ + 222176 // 531 * τ^2 - 720128 // 531 * τ^3 +
                876544 // 531 * τ^4 - 360448 // 531 * τ^5,
                -2016 // 59 * τ + 222176 // 531 * τ^2 - 720128 // 531 * τ^3 +
                876544 // 531 * τ^4 - 360448 // 531 * τ^5,
                -756 // 59 * τ + 27772 // 177 * τ^2 - 90016 // 177 * τ^3 +
                109568 // 177 * τ^4 - 45056 // 177 * τ^5,
                -190464 // 413 * τ + 62384128 // 11151 * τ^2 -
                197718016 // 11151 * τ^3 +
                233799680 // 11151 * τ^4 - 93323264 // 11151 * τ^5,
                1792 // 5 * τ - 4352 * τ^2 + 13824 * τ^3 - 16384 * τ^4 +
                32768 // 5 * τ^5,
                100352 // 531 * τ - 179554304 // 78057 * τ^2 +
                573452288 // 78057 * τ^3 -
                683376640 // 78057 * τ^4 + 274726912 // 78057 * τ^5,
                16384 // 147 * τ^2 - 65536 // 147 * τ^3 + 81920 // 147 * τ^4 -
                32768 // 147 * τ^5]
        end
        return T.(w), T.(wp)
    end end
end
