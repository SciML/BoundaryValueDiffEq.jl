"""
    interp_eval!(y::AbstractArray, cache::MIRKCache, t)

After we construct an interpolant, we use interp_eval to evaluate it.
"""
@views function interp_eval!(y::AbstractArray, cache::MIRKCache, t, mesh, mesh_dt)
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    w, w′ = interp_weights(τ, cache.alg)
    sum_stages!(y, cache, w, i)
    return y
end

"""
    interval(mesh, t)

Find the interval that `t` belongs to in `mesh`. Assumes that `mesh` is sorted.
"""
function interval(mesh, t)
    return clamp(searchsortedfirst(mesh, t) - 1, 1, length(mesh) - 1)
end

"""
    mesh_selector!(cache::MIRKCache)

Generate new mesh based on the defect.
"""
@views function mesh_selector!(
        cache::MIRKCache{iip, T}, error_control::DefectControl) where {iip, T}
    (; order, errors, mesh, mesh_dt) = cache
    (abstol, _, _), kwargs = __split_mirk_kwargs(; cache.kwargs...)
    N = length(mesh)

    safety_factor = T(1.3)
    ρ = T(1.0) # Set rho=1 means mesh distribution will take place everytime.
    Nsub_star = 0
    Nsub_star_ub = 4 * (N - 1)
    Nsub_star_lb = N ÷ 2

    info = ReturnCode.Success

    ŝ = [maximum(abs, d) for d in errors]  # Broadcasting breaks GPU Compilation
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
        if Nsub_star > cache.alg.max_num_subintervals # Need to determine the too large threshold
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

@views function mesh_selector!(
        cache::MIRKCache{iip, T}, error_control::GlobalErrorControl) where {iip, T}
    (; order, errors, TU, mesh, mesh_dt) = cache
    (abstol, _, _), kwargs = __split_mirk_kwargs(; cache.kwargs...)
    (; p_star) = TU
    N = length(mesh)

    safety_factor = T(1.3)
    ρ = T(2.0) # Set rho=2 means mesh distribution will take place everytime.
    Nsub_star = 0
    Nsub_star_ub = 4 * (N - 1)
    Nsub_star_lb = N ÷ 2

    info = ReturnCode.Success

    ŝ = [maximum(abs, d) for d in errors]  # Broadcasting breaks GPU Compilation
    ŝ .= (ŝ ./ abstol) .^ (T(1) / order)
    r₁ = maximum(ŝ)
    r₂ = sum(ŝ)
    r₃ = r₂ / (N - 1)

    n_predict = round(Int, (safety_factor * r₂) + 1)
    n = N - 1
    n_ = T(0.1) * n
    n_predict = ifelse(abs((n_predict - n)) < n_, round(Int, n + n_), n_predict)

    if r₁ ≤ ρ * r₃
        Nsub_star = 2 * (N - 1)
        if Nsub_star > cache.alg.max_num_subintervals # Need to determine the too large threshold
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

@views function mesh_selector!(cache::MIRKCache{iip, T},
        error_control::Union{SequentialErrorControl, HybridErrorControl}) where {iip, T}
    (; order, errors, TU, mesh, mesh_dt) = cache
    (abstol, _, _), kwargs = __split_mirk_kwargs(; cache.kwargs...)
    (; p_star) = TU
    N = length(mesh)

    safety_factor = T(1.3)
    ρ = T(2.0) # Set rho=2 means mesh distribution will take place everytime.
    Nsub_star = 0
    Nsub_star_ub = 4 * (N - 1)
    Nsub_star_lb = N ÷ 2

    info = ReturnCode.Success

    #TODO: need mesh selection for both defect and global errors
    ŝ = [maximum(abs, d) for d in errors]  # Broadcasting breaks GPU Compilation
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
        if Nsub_star > cache.alg.max_num_subintervals # Need to determine the too large threshold
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
    redistribute!(cache::MIRKCache, Nsub_star, ŝ, mesh, mesh_dt)

Generate a new mesh based on the `ŝ`.
"""
function redistribute!(
        cache::MIRKCache{iip, T}, Nsub_star, ŝ, mesh, mesh_dt) where {iip, T}
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
    half_mesh!(cache::MIRKCache)

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
function half_mesh!(cache::MIRKCache)
    half_mesh!(cache.mesh, cache.mesh_dt)
end

"""
    half_sol(sol)

The input sol has length of `n + 1`. Divide the original solution into two equal length
solution.
"""
function half_sol(sol::AbstractVectorOfArray{T}) where {T}
    new_sol = copy(sol)
    n = length(sol) - 1
    resize!(new_sol, 2 * n + 1)
    new_sol[2n + 1] = sol[n + 1]
    for i in (2n - 1):-2:1
        new_sol[i] = new_sol[(i + 1) ÷ 2]
    end
    @simd for i in (2n):-2:2
        new_sol[i] = (new_sol[i + 1] + new_sol[i - 1]) ./ T(2)
    end
    return new_sol
end

"""
    error_estimate!(cache::MIRKCache)

defect_estimate use the discrete solution approximation Y, plus stages of
the RK method in 'k_discrete', plus some new stages in 'k_interp' to construct
an interpolant
"""
# Defect control
@views function error_estimate!(
        cache::MIRKCache{iip, T}, error_control::GlobalErrorControl, sol,
        nlsolve_alg, abstol, dt, kwargs, nlsolve_kwargs) where {iip, T}
    return error_estimate!(cache::MIRKCache{iip, T}, error_control, error_control.method,
        sol, nlsolve_alg, abstol, dt, kwargs, nlsolve_kwargs)
end

# Global error control
@views function error_estimate!(cache::MIRKCache{iip, T}, error_control::DefectControl, sol,
        nlsolve_alg, abstol, dt, kwargs, nlsolve_kwargs) where {iip, T}
    (; f, alg, mesh, mesh_dt, errors) = cache
    (; τ_star) = cache.ITU

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

        errors.u[i] .= est₁ > est₂ ? yᵢ₁ : yᵢ₂
    end

    return maximum(Base.Fix1(maximum, abs), errors.u)
end

# Sequential error control
@views function error_estimate!(
        cache::MIRKCache{iip, T}, error_control::SequentialErrorControl,
        sol, nlsolve_alg, abstol, dt, kwargs, nlsolve_kwargs) where {iip, T}
    defect_norm = error_estimate!(cache::MIRKCache{iip, T}, DefectControl(), sol,
        nlsolve_alg, abstol, dt, kwargs, nlsolve_kwargs)
    error_norm = defect_norm
    if defect_norm <= abstol
        global_error_norm = error_estimate!(
            cache::MIRKCache{iip, T}, GlobalErrorControl(), sol,
            nlsolve_alg, abstol, dt, kwargs, nlsolve_kwargs)
        error_norm = global_error_norm
    end
    return error_norm
end

# Hybrid error control
@views function error_estimate!(
        cache::MIRKCache{iip, T}, error_control::HybridErrorControl, sol,
        nlsolve_alg, abstol, dt, kwargs, nlsolve_kwargs) where {iip, T}
    defect_norm = error_estimate!(cache::MIRKCache{iip, T}, DefectControl(), sol,
        nlsolve_alg, abstol, dt, kwargs, nlsolve_kwargs)
    defect_norm
    global_error_norm = error_estimate!(cache::MIRKCache{iip, T}, GlobalErrorControl(), sol,
        nlsolve_alg, abstol, dt, kwargs, nlsolve_kwargs)

    error_norm = defect_norm + global_error_norm
    return error_norm
end

@views function error_estimate!(cache::MIRKCache{iip, T}, error_control::GlobalErrorControl,
        global_error_control::REErrorControl, sol, nlsolve_alg,
        abstol, dt, kwargs, nlsolve_kwargs) where {iip, T}
    (; prob, alg, errors) = cache

    # Use the previous solution as the initial guess
    new_prob = remake(prob, u0 = half_sol(cache.y₀))
    high_cache = SciMLBase.__init(new_prob, alg, dt = dt / 2, adaptive = false)

    high_nlprob = __construct_nlproblem(high_cache, sol, copy(sol))
    high_sol_original = __solve(
        high_nlprob, nlsolve_alg; abstol, kwargs..., nlsolve_kwargs..., alias_u0 = true)
    high_sol = high_sol_original.u[1:2:end]
    error_norm = global_error(VectorOfArray(high_sol), copy(cache.y₀), errors)
    return error_norm
end

@views function error_estimate!(cache::MIRKCache{iip, T}, error_control::GlobalErrorControl,
        global_error_control::HOErrorControl, sol, nlsolve_alg,
        abstol, dt, kwargs, nlsolve_kwargs) where {iip, T}
    (; prob, alg, errors) = cache

    # Use the previous solution as the initial guess
    new_prob = remake(prob, u0 = copy(cache.y₀))
    high_cache = SciMLBase.__init(
        new_prob, __high_order_method(alg), dt = dt, adaptive = false)

    high_nlprob = __construct_nlproblem(high_cache, sol, copy(sol))
    high_sol = __solve(
        high_nlprob, nlsolve_alg; abstol, kwargs..., nlsolve_kwargs..., alias_u0 = true)
    error_norm = global_error(VectorOfArray(high_sol.u), copy(cache.y₀), errors)
    return error_norm
end

@inline function __high_order_mirk(alg::AbstractMIRK)
    new_alg = Symbol("MIRK$(alg_order(alg) + 2)")
    return @eval $(new_alg)()
end

@views function global_error(high_sol, low_sol, errors)
    errors .= (high_sol .- low_sol) ./ (1 .+ low_sol)
    return maximum(Base.Fix1(maximum, abs), errors.u)
end

"""
    interp_setup!(cache::MIRKCache)

`interp_setup!` prepare the extra stages in ki_interp for interpolant construction.
Here, the ki_interp is the stages in one subinterval.
"""
@views function interp_setup!(cache::MIRKCache{iip, T}) where {iip, T}
    (; x_star, s_star, c_star, v_star) = cache.ITU
    (; k_interp, k_discrete, f, stage, new_stages, y, p, mesh, mesh_dt) = cache

    for r in 1:(s_star - stage)
        idx₁ = ((1:stage) .- 1) .* (s_star - stage) .+ r
        idx₂ = ((1:(r - 1)) .+ stage .- 1) .* (s_star - stage) .+ r
        for j in eachindex(k_discrete)
            __maybe_matmul!(new_stages.u[j], k_discrete[j].du[:, 1:stage], x_star[idx₁])
        end
        if r > 1
            for j in eachindex(k_interp)
                __maybe_matmul!(
                    new_stages.u[j], k_interp.u[j][:, 1:(r - 1)], x_star[idx₂], T(1), T(1))
            end
        end
        for i in eachindex(new_stages)
            new_stages.u[i] .= new_stages.u[i] .* mesh_dt[i] .+
                               (1 - v_star[r]) .* vec(y[i].du) .+
                               v_star[r] .* vec(y[i + 1].du)
            if iip
                f(k_interp.u[i][:, r], new_stages.u[i], p, mesh[i] + c_star[r] * mesh_dt[i])
            else
                k_interp.u[i][:, r] .= f(
                    new_stages.u[i], p, mesh[i] + c_star[r] * mesh_dt[i])
            end
        end
    end

    return k_interp
end

"""
    sum_stages!(cache::MIRKCache, w, w′, i::Int)

sum_stages add the discrete solution, RK method stages and extra stages to construct interpolant.
"""
function sum_stages!(cache::MIRKCache, w, w′, i::Int, dt = cache.mesh_dt[i])
    sum_stages!(cache.fᵢ_cache.du, cache.fᵢ₂_cache, cache, w, w′, i, dt)
end

function sum_stages!(z::AbstractArray, cache::MIRKCache, w, i::Int, dt = cache.mesh_dt[i])
    (; stage, k_discrete, k_interp) = cache
    (; s_star) = cache.ITU

    z .= zero(z)
    __maybe_matmul!(z, k_discrete[i].du[:, 1:stage], w[1:stage])
    __maybe_matmul!(
        z, k_interp.u[i][:, 1:(s_star - stage)], w[(stage + 1):s_star], true, true)
    z .= z .* dt .+ cache.y₀.u[i]

    return z
end

@views function sum_stages!(z, z′, cache::MIRKCache, w, w′, i::Int, dt = cache.mesh_dt[i])
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
    z .= z .* dt[1] .+ cache.y₀.u[i]

    return z, z′
end

"""
    interp_weights(τ, alg)

interp_weights: solver-specified interpolation weights and its first derivative
"""
function interp_weights end

for order in (2, 3, 4, 5, 6)
    alg = Symbol("MIRK$(order)")
    @eval begin
        function interp_weights(τ::T, ::$(alg)) where {T}
            if $(order == 2)
                w = [0, τ * (1 - τ / 2), τ^2 / 2]

                #     Derivative polynomials.

                wp = [0, 1 - τ, τ]
            elseif $(order == 3)
                w = [τ / 4.0 * (2.0 * τ^2 - 5.0 * τ + 4.0),
                    -3.0 / 4.0 * τ^2 * (2.0 * τ - 3.0), τ^2 * (τ - 1.0)]

                #     Derivative polynomials.

                wp = [3.0 / 2.0 * (τ - 2.0 / 3.0) * (τ - 1.0),
                    -9.0 / 2.0 * τ * (τ - 1.0), 3.0 * τ * (τ - 2.0 / 3.0)]
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

                wp = [-tm1 * t4m3 * t2m1 / 3.0, τ * t2m1 * t4m3,
                    4.0 * τ * t4m3 * tm1, -32.0 * τ * t2m1 * tm1 / 3.0]
            elseif $(order == 5)
                w = [
                    τ * (22464.0 - 83910.0 * τ + 143041.0 * τ^2 - 113808.0 * τ^3 +
                     33256.0 * τ^4) / 22464.0,
                    τ^2 * (-2418.0 + 12303.0 * τ - 19512.0 * τ^2 + 10904.0 * τ^3) / 3360.0,
                    -8 / 81 * τ^2 * (-78.0 + 209.0 * τ - 204.0 * τ^2 + 8.0 * τ^3),
                    -25 / 1134 * τ^2 * (-390.0 + 1045.0 * τ - 1020.0 * τ^2 + 328.0 * τ^3),
                    -25 / 5184 * τ^2 * (390.0 + 255.0 * τ - 1680.0 * τ^2 + 2072.0 * τ^3),
                    279841 / 168480 * τ^2 * (-6.0 + 21.0 * τ - 24.0 * τ^2 + 8.0 * τ^3)]

                #   Derivative polynomials

                wp = [
                    1.0 - 13985 // 1872 * τ + 143041 // 7488 * τ^2 - 2371 // 117 * τ^3 +
                    20785 // 2808 * τ^4,
                    -403 // 280 * τ + 12303 // 1120 * τ^2 - 813 // 35 * τ^3 +
                    1363 // 84 * τ^4,
                    416 // 27 * τ - 1672 // 27 * τ^2 + 2176 // 27 * τ^3 - 320 // 81 * τ^4,
                    3250 // 189 * τ - 26125 // 378 * τ^2 + 17000 // 189 * τ^3 -
                    20500 // 567 * τ^4,
                    -1625 // 432 * τ - 2125 // 576 * τ^2 + 875 // 27 * τ^3 -
                    32375 // 648 * τ^4,
                    -279841 // 14040 * τ + 1958887 // 18720 * τ^2 - 279841 // 1755 * τ^3 +
                    279841 // 4212 * τ^4]
            elseif $(order == 6)
                w = [
                    τ - 28607 // 7434 * τ^2 - 166210 // 33453 * τ^3 +
                    334780 // 11151 * τ^4 - 1911296 // 55755 * τ^5 + 406528 // 33453 * τ^6,
                    777 // 590 * τ^2 - 2534158 // 234171 * τ^3 + 2088580 // 78057 * τ^4 -
                    10479104 // 390285 * τ^5 + 11328512 // 1170855 * τ^6,
                    -1008 // 59 * τ^2 + 222176 // 1593 * τ^3 - 180032 // 531 * τ^4 +
                    876544 // 2655 * τ^5 - 180224 // 1593 * τ^6,
                    -1008 // 59 * τ^2 + 222176 // 1593 * τ^3 - 180032 // 531 * τ^4 +
                    876544 // 2655 * τ^5 - 180224 // 1593 * τ^6,
                    -378 // 59 * τ^2 + 27772 // 531 * τ^3 - 22504 // 177 * τ^4 +
                    109568 // 885 * τ^5 - 22528 // 531 * τ^6,
                    -95232 // 413 * τ^2 + 62384128 // 33453 * τ^3 -
                    49429504 // 11151 * τ^4 + 46759936 // 11151 * τ^5 -
                    46661632 // 33453 * τ^6,
                    896 // 5 * τ^2 - 4352 // 3 * τ^3 + 3456 * τ^4 - 16384 // 5 * τ^5 +
                    16384 // 15 * τ^6,
                    50176 // 531 * τ^2 - 179554304 // 234171 * τ^3 +
                    143363072 // 78057 * τ^4 - 136675328 // 78057 * τ^5 +
                    137363456 // 234171 * τ^6,
                    16384 // 441 * τ^3 - 16384 // 147 * τ^4 + 16384 // 147 * τ^5 -
                    16384 // 441 * τ^6]

                #     Derivative polynomials.

                wp = [
                    1 - 28607 // 3717 * τ - 166210 // 11151 * τ^2 + 1339120 // 11151 * τ^3 -
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
                    197718016 // 11151 * τ^3 + 233799680 // 11151 * τ^4 -
                    93323264 // 11151 * τ^5,
                    1792 // 5 * τ - 4352 * τ^2 + 13824 * τ^3 - 16384 * τ^4 +
                    32768 // 5 * τ^5,
                    100352 // 531 * τ - 179554304 // 78057 * τ^2 +
                    573452288 // 78057 * τ^3 - 683376640 // 78057 * τ^4 +
                    274726912 // 78057 * τ^5,
                    16384 // 147 * τ^2 - 65536 // 147 * τ^3 + 81920 // 147 * τ^4 -
                    32768 // 147 * τ^5]
            end
            return T.(w), T.(wp)
        end
    end
end

for order in (6,)
    alg = Symbol("MIRK$(order)I")
    @eval begin
        function interp_weights(τ::T, ::$(alg)) where {T}
            if $(order == 6)
                w = [
                    -(12233 + 1450 * sqrt(7)) *
                    (800086000 * τ^5 + 63579600 * sqrt(7) * τ^4 - 2936650584 * τ^4 +
                     4235152620 * τ^3 - 201404565 * sqrt(7) * τ^3 +
                     232506630 * sqrt(7) * τ^2 - 3033109390 * τ^2 + 1116511695 * τ -
                     116253315 * sqrt(7) * τ + 22707000 * sqrt(7) - 191568780) *
                    τ / 2112984835740,
                    -(-10799 + 650 * sqrt(7)) *
                    (24962000 * τ^4 + 473200 * sqrt(7) * τ^3 - 67024328 * τ^3 -
                     751855 * sqrt(7) * τ^2 + 66629600 * τ^2 - 29507250 * τ +
                     236210 * sqrt(7) * τ +
                     5080365 +
                     50895 * sqrt(7)) *
                    τ^2 / 29551834260,
                    7 / 1274940 *
                    (259 + 50 * sqrt(7)) *
                    (14000 * τ^4 - 48216 * τ^3 + 1200 * sqrt(7) * τ^3 -
                     3555 * sqrt(7) * τ^2 +
                     62790 * τ^2 +
                     3610 * sqrt(7) * τ - 37450 * τ + 9135 - 1305 * sqrt(7)) *
                    τ^2,
                    7 / 1274940 *
                    (259 + 50 * sqrt(7)) *
                    (14000 * τ^4 - 48216 * τ^3 + 1200 * sqrt(7) * τ^3 -
                     3555 * sqrt(7) * τ^2 +
                     62790 * τ^2 +
                     3610 * sqrt(7) * τ - 37450 * τ + 9135 - 1305 * sqrt(7)) *
                    τ^2,
                    16 / 2231145 *
                    (259 + 50 * sqrt(7)) *
                    (14000 * τ^4 - 48216 * τ^3 + 1200 * sqrt(7) * τ^3 -
                     3555 * sqrt(7) * τ^2 +
                     62790 * τ^2 +
                     3610 * sqrt(7) * τ - 37450 * τ + 9135 - 1305 * sqrt(7)) *
                    τ^2,
                    4 / 1227278493 *
                    (740 * sqrt(7) - 6083) *
                    (1561000 * τ^2 - 2461284 * τ - 109520 * sqrt(7) * τ +
                     979272 +
                     86913 * sqrt(7)) *
                    (τ - 1)^2 *
                    τ^2,
                    -49 / 63747 *
                    sqrt(7) *
                    (20000 * τ^2 - 20000 * τ + 3393) *
                    (τ - 1)^2 *
                    τ^2,
                    -1250000000 / 889206903 * (28 * τ^2 - 28 * τ + 9) * (τ - 1)^2 * τ^2]

                #     Derivative polynomials.

                wp = [
                    (1450 * sqrt(7) + 12233) *
                    (14 * τ - 7 + sqrt(7)) *
                    (τ - 1) *
                    (-400043 * τ + 75481 + 2083 * sqrt(7)) *
                    (100 * τ - 87) *
                    (2 * τ - 1) / 493029795006,
                    -(650 * sqrt(7) - 10799) *
                    (14 * τ - 7 + sqrt(7)) *
                    (37443 * τ - 13762 - 2083 * sqrt(7)) *
                    (100 * τ - 87) *
                    (2 * τ - 1) *
                    τ / 20686283982,
                    7 / 42498 *
                    (259 + 50 * sqrt(7)) *
                    (14 * τ - 7 + sqrt(7)) *
                    (τ - 1) *
                    (100 * τ - 87) *
                    (2 * τ - 1) *
                    τ,
                    7 / 42498 *
                    (259 + 50 * sqrt(7)) *
                    (14 * τ - 7 + sqrt(7)) *
                    (τ - 1) *
                    (100 * τ - 87) *
                    (2 * τ - 1) *
                    τ,
                    32 / 148743 *
                    (259 + 50 * sqrt(7)) *
                    (14 * τ - 7 + sqrt(7)) *
                    (τ - 1) *
                    (100 * τ - 87) *
                    (2 * τ - 1) *
                    τ,
                    4 / 1227278493 *
                    (740 * sqrt(7) - 6083) *
                    (14 * τ - 7 + sqrt(7)) *
                    (τ - 1) *
                    (100 * τ - 87) *
                    (6690 * τ - 4085 - 869 * sqrt(7)) *
                    τ,
                    -98 / 21249 *
                    sqrt(7) *
                    (τ - 1) *
                    (100 * τ - 13) *
                    (100 * τ - 87) *
                    (2 * τ - 1) *
                    τ,
                    -1250000000 / 2074816107 *
                    (14 * τ - 7 + sqrt(7)) *
                    (τ - 1) *
                    (14 * τ - 7 - sqrt(7)) *
                    (2 * τ - 1) *
                    τ]
            end
            return T.(w), T.(wp)
        end
    end
end
