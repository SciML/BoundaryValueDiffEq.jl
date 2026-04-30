"""
    interp_eval!(y::AbstractArray, cache::MIRKCache, t)

After we construct an interpolant, we use interp_eval to evaluate it.
"""
@views function interp_eval!(y::AbstractArray, cache::MIRKCache, t, mesh, mesh_dt)
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    w, _ = interp_weights(τ, cache.alg)
    sum_stages!(y, cache, w, i, dt)
    return y
end

"""
    mesh_selector!(cache::MIRKCache, controller::DefectControl)
    mesh_selector!(cache::MIRKCache, controller::GlobalErrorControl)
    mesh_selector!(cache::MIRKCache, controller::SequentialErrorControl)
    mesh_selector!(cache::MIRKCache, controller::HybridErrorControl)

Generate new mesh based on the defect or the global error.
"""
@views function mesh_selector!(cache::MIRKCache{iip, T}, controller::DefectControl) where {
        iip, T,
    }
    (; order, errors, mesh, mesh_dt) = cache
    (abstol, _, _, _), _ = __split_kwargs(; cache.kwargs...)
    N = length(mesh)
    n = N - 1

    safety_factor = T(1.3)
    ρ = T(1.0)
    Nsub_star = 0
    Nsub_star_ub = 4 * (N - 1)
    Nsub_star_lb = N ÷ 2

    info = ReturnCode.Success

    ŝ = [maximum(abs, d) for d in errors.u]  # Broadcasting breaks GPU Compilation
    ŝ .= (ŝ ./ abstol) .^ (T(1) / (order + 1))
    r₁ = maximum(ŝ)
    r₂ = sum(ŝ)
    r₃ = r₂ / (N - 1)

    n_predict = round(Int, (safety_factor * r₂) + 1)
    n_ = T(0.1) * n
    n_predict = ifelse(abs((n_predict - n)) < n_, round(Int, n + n_), n_predict)

    if r₁ ≤ ρ * r₃
        Nsub_star = 2 * n
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

@views function mesh_selector!(cache::MIRKCache{iip, T}, controller::GlobalErrorControl) where {
        iip, T,
    }
    (; order, errors, mesh, mesh_dt) = cache
    (abstol, _, _, _), _ = __split_kwargs(; cache.kwargs...)
    N = length(mesh)
    n = N - 1

    safety_factor = T(1.3)
    ρ = T(2.0)
    Nsub_star = 0
    Nsub_star_ub = 4 * (N - 1)
    Nsub_star_lb = N ÷ 2

    info = ReturnCode.Success

    ŝ = [maximum(abs, d) for d in errors.u]
    ŝ .= (ŝ ./ abstol) .^ (T(1) / order)
    r₁ = maximum(ŝ)
    r₂ = sum(ŝ)
    r₃ = r₂ / n

    n_predict = round(Int, (safety_factor * r₂) + 1)
    n_ = T(0.1) * n
    n_predict = ifelse(abs((n_predict - n)) < n_, round(Int, n + n_), n_predict)

    if r₁ ≤ ρ * r₃
        Nsub_star = 2 * n
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

@views function mesh_selector!(cache::MIRKCache{iip, T}, controller::SequentialErrorControl) where {
        iip, T,
    }
    (; order, errors, mesh, mesh_dt) = cache
    (abstol, _, _, _), _ = __split_kwargs(; cache.kwargs...)
    N = length(mesh)
    n = N - 1

    safety_factor = T(1.3)
    ρ = T(2.0)
    Nsub_star = 0
    Nsub_star_ub = 4 * (N - 1)
    Nsub_star_lb = N ÷ 2

    info = ReturnCode.Success

    ŝ = [maximum(abs, d) for d in errors]
    ŝ .= (ŝ ./ abstol) .^ (T(1) / (order + 1))
    r₁ = maximum(ŝ)
    r₂ = sum(ŝ)
    r₃ = r₂ / n

    n_predict = round(Int, (safety_factor * r₂) + 1)
    n = N - 1
    n_ = T(0.1) * n
    n_predict = ifelse(abs((n_predict - n)) < n_, round(Int, n + n_), n_predict)

    if r₁ ≤ ρ * r₃
        Nsub_star = 2 * n
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

@views function mesh_selector!(cache::MIRKCache{iip, T}, controller::HybridErrorControl) where {
        iip, T,
    }
    (; order, errors, mesh, mesh_dt) = cache
    (abstol, _, _, _), _ = __split_kwargs(; cache.kwargs...)
    N = length(mesh)
    n = N - 1

    safety_factor = T(1.3)
    ρ = T(2.0)
    Nsub_star = 0
    Nsub_star_ub = 4 * n
    Nsub_star_lb = N ÷ 2

    info = ReturnCode.Success

    ŝ₁ = [maximum(abs, d) for d in errors.u[1:n]]
    ŝ₂ = [maximum(abs, d) for d in errors.u[N:end]]
    ŝ = similar(ŝ₁)
    ŝ .= (ŝ₁ ./ abstol) .^ (T(1) / (order + 1)) + (ŝ₂ ./ abstol) .^ (T(1) / (order + 1))
    r₁ = maximum(ŝ)
    r₂ = sum(ŝ)
    r₃ = r₂ / n

    n_predict = round(Int, (safety_factor * r₂) + 1)
    n_ = T(0.1) * n
    n_predict = ifelse(abs((n_predict - n)) < n_, round(Int, n + n_), n_predict)

    if r₁ ≤ ρ * r₃
        Nsub_star = 2 * n
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
    redistribute!(cache::MIRKCache, Nsub_star, ŝ, mesh, mesh_dt)

Generate a new mesh based on the `ŝ`.
"""
function redistribute!(
        cache::MIRKCache{iip, T}, Nsub_star, ŝ, mesh, mesh_dt
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
    return half_mesh!(cache.mesh, cache.mesh_dt)
end

"""
    halve_sol(sol)

The input sol has length of `n + 1`. Divide the original mesh and u from original solution into `2n + 1` one.
"""
function halve_sol(sol::AbstractVectorOfArray{T}, mesh) where {T}
    new_sol = copy(sol)
    n = length(sol.u) - 1
    resize!(new_sol, 2 * n + 1)
    new_sol.u[2n + 1] = sol.u[n + 1]
    for i in (2n - 1):-2:1
        new_sol.u[i] = new_sol.u[(i + 1) ÷ 2]
    end
    @simd for i in (2n):-2:2
        new_sol.u[i] = (new_sol.u[i + 1] + new_sol.u[i - 1]) ./ T(2)
    end
    new_mesh = deepcopy(mesh)
    resize!(new_mesh, 2 * n + 1)
    new_mesh[1] = mesh[1]
    new_mesh[end] = mesh[end]
    for i in (2n - 1):-2:1
        new_mesh[i] = new_mesh[(i + 1) ÷ 2]
    end
    for i in (2n):-2:2
        new_mesh[i] = (new_mesh[i + 1] + new_mesh[i - 1]) / 2
    end
    return DiffEqArray(new_sol.u, new_mesh)
end

"""
    error_estimate!(cache::MIRKCache, controller::DefectControl)
    error_estimate!(cache::MIRKCache, controller::GlobalErrorControl)
    error_estimate!(cache::MIRKCache, controller::SequentialErrorControl)
    error_estimate!(cache::MIRKCache, controller::HybridErrorControl)

## Defect Control

error_estimate for the defect uses the discrete solution approximation Y, plus stages of
the RK method in 'k_discrete', plus some new stages in 'k_interp' to construct
an interpolant.

## Global Error Control
error_estimate for the global error use the higher order or doubled mesh to estimate the
global error according to err = max(abs(Y_high - Y_low)) / (1 + abs(Y_low))

## Sequential Error Control
error_estimate for the sequential error first uses the defect controller, if the defect is
satisfying, then use the global error controller.

## Hybrid Error Control
error_estimate for the hybrid error control uses the linear combination of defect and global
error to estimate the error norm.
"""
# Defect control
@views function error_estimate!(
        cache::MIRKCache{iip, T}, controller::GlobalErrorControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T}
    return error_estimate!(
        cache, controller, controller.method, errors, sol, nlsolve_alg, abstol
    )
end

# Global error control
@views function error_estimate!(
        cache::MIRKCache{iip, T, use_both, DiffCacheNeeded}, controller::DefectControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T, use_both}
    (; f, alg, mesh, mesh_dt) = cache
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

    defect_norm = maximum(Base.Fix1(maximum, abs), errors.u)

    # The defect is greater than 10%, the solution is not acceptable
    info = ifelse(defect_norm > controller.defect_threshold, ReturnCode.Failure, ReturnCode.Success)
    return defect_norm, info
end
@views function error_estimate!(
        cache::MIRKCache{iip, T, use_both, NoDiffCacheNeeded}, controller::DefectControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T, use_both}
    (; f, alg, mesh, mesh_dt) = cache
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
            yᵢ₁ = cache.y[i]
            f(yᵢ₁, z, cache.p, mesh[i] + τ_star * dt)
        else
            yᵢ₁ = f(z, cache.p, mesh[i] + τ_star * dt)
        end
        yᵢ₁ .= (z′ .- yᵢ₁) ./ (abs.(yᵢ₁) .+ T(1))
        est₁ = maximum(abs, yᵢ₁)

        z, z′ = sum_stages!(cache, w₂, w₂′, i)
        if iip
            yᵢ₂ = cache.y[i + 1]
            f(yᵢ₂, z, cache.p, mesh[i] + (T(1) - τ_star) * dt)
        else
            yᵢ₂ = f(z, cache.p, mesh[i] + (T(1) - τ_star) * dt)
        end
        yᵢ₂ .= (z′ .- yᵢ₂) ./ (abs.(yᵢ₂) .+ T(1))
        est₂ = maximum(abs, yᵢ₂)

        errors.u[i] .= est₁ > est₂ ? yᵢ₁ : yᵢ₂
    end

    defect_norm = maximum(Base.Fix1(maximum, abs), errors.u)

    # The defect is greater than 10%, the solution is not acceptable
    info = ifelse(defect_norm > controller.defect_threshold, ReturnCode.Failure, ReturnCode.Success)
    return defect_norm, info
end

# Sequential error control
@views function error_estimate!(
        cache::MIRKCache{iip, T}, controller::SequentialErrorControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T}
    defect_norm,
        info = error_estimate!(
        cache::MIRKCache{iip, T}, controller.defect, errors, sol, nlsolve_alg, abstol
    )
    error_norm = defect_norm
    if defect_norm <= abstol
        global_error_norm,
            info = error_estimate!(
            cache::MIRKCache{iip, T}, controller.global_error,
            controller.global_error.method, errors, sol, nlsolve_alg, abstol
        )
        error_norm = global_error_norm
        return error_norm, info
    end
    return error_norm, info
end

# Hybrid error control
function error_estimate!(
        cache::MIRKCache{iip, T}, controller::HybridErrorControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T}
    L = length(cache.mesh) - 1
    defect = errors[:, 1:L]
    global_error = errors[:, (L + 1):end]
    defect_norm,
        _ = error_estimate!(
        cache::MIRKCache{iip, T}, controller.defect, defect, sol, nlsolve_alg, abstol
    )
    global_error_norm,
        _ = error_estimate!(
        cache, controller.global_error, controller.global_error.method,
        global_error, sol, nlsolve_alg, abstol
    )

    error_norm = controller.DE * defect_norm + controller.GE * global_error_norm
    copyto!(errors, VectorOfArray(vcat(defect.u, global_error.u)))
    return error_norm, ReturnCode.Success
end

@views function error_estimate!(
        cache::MIRKCache{iip, T}, controller::GlobalErrorControl,
        global_error_control::REErrorControl, errors,
        sol, nlsolve_alg, abstol
    ) where {iip, T}
    (; prob, alg) = cache

    # Use the previous solution as the initial guess
    high_sol = halve_sol(cache.y₀, cache.mesh)
    new_prob = remake(prob, u0 = high_sol)
    high_cache = SciMLBase.__init(new_prob, alg, adaptive = false)

    high_nlprob = __construct_problem(high_cache, vec(high_sol), VectorOfArray(high_sol.u))
    high_sol_original = __solve(high_nlprob, nlsolve_alg; cache.nlsolve_kwargs..., alias_u0 = true)
    recursive_unflatten!(high_sol, high_sol_original.u)
    error_norm = global_error(VectorOfArray(copy(high_sol.u[1:2:end])), copy(cache.y₀), errors)
    return error_norm * 2^cache.order / (2^cache.order - 1), ReturnCode.Success
end

@views function error_estimate!(
        cache::MIRKCache{iip, T}, controller::GlobalErrorControl,
        global_error_control::HOErrorControl, errors,
        sol, nlsolve_alg, abstol
    ) where {iip, T}
    (; prob, alg) = cache

    # Use the previous solution as the initial guess
    high_sol = DiffEqArray(cache.y₀.u, cache.mesh)
    new_prob = remake(prob, u0 = high_sol)
    high_cache = SciMLBase.__init(new_prob, __high_order_method(alg), adaptive = false)

    high_nlprob = __construct_problem(high_cache, sol.u, high_sol)
    high_sol_nlprob = __solve(high_nlprob, nlsolve_alg; cache.nlsolve_kwargs..., alias_u0 = true)
    recursive_unflatten!(high_sol, high_sol_nlprob)
    error_norm = global_error(VectorOfArray(high_sol.u), cache.y₀, errors)
    return error_norm, ReturnCode.Success
end

@inline function __high_order_method(alg::AbstractMIRK)
    new_alg = Symbol("MIRK$(alg_order(alg) + 2)")
    return @eval $(new_alg)()
end

@views function global_error(high_sol, low_sol, errors)
    err = (high_sol .- low_sol) ./ (1 .+ abs.(low_sol))
    GE_subinterval!(errors, err)
    return maximum(Base.Fix1(maximum, abs), errors.u)
end

# Assigns the global error estimate for each subinterval
# Basically shrink Nig+1 error estimates to Nig error estimates
@views function GE_subinterval!(errors, err)
    copyto!(
        errors.u,
        [
            ifelse(maximum(abs.(err.u[i])) >= maximum(abs.(err.u[i + 1])), err.u[i], err.u[i + 1])
                for i in 1:(length(err) - 1)
        ]
    )
end

"""
    sum_stages!(cache::MIRKCache, w, w′, i::Int)

sum_stages add the discrete solution, RK method stages and extra stages to construct interpolant.
"""
function sum_stages!(
        cache::MIRKCache{iip, T, use_both, DiffCacheNeeded}, w, w′,
        i::Int, dt = cache.mesh_dt[i]
    ) where {iip, T, use_both}
    return sum_stages!(cache.fᵢ_cache.du, cache.fᵢ₂_cache, cache, w, w′, i, dt)
end
function sum_stages!(
        cache::MIRKCache{iip, T, use_both, NoDiffCacheNeeded}, w,
        w′, i::Int, dt = cache.mesh_dt[i]
    ) where {iip, T, use_both}
    return sum_stages!(cache.fᵢ_cache, cache.fᵢ₂_cache, cache, w, w′, i, dt)
end

# Here we should not directly in-place change z in several steps
# because in final step we actually need to use the original z(which is cache.y₀.u[i])
# we use fᵢ₂_cache to avoid additional allocations.
@views function sum_stages!(
        z::AbstractArray, cache::MIRKCache{iip, T, use_both, DiffCacheNeeded},
        w, i::Int, dt = cache.mesh_dt[i]
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp, fᵢ₂_cache) = cache
    (; s_star) = cache.ITU

    fᵢ₂_cache .= zero(z)
    __maybe_matmul!(fᵢ₂_cache, k_discrete[i].du[:, 1:stage], w[1:stage])
    __maybe_matmul!(
        fᵢ₂_cache, k_interp.u[i][:, 1:(s_star - stage)], w[(stage + 1):s_star], true, true
    )
    z .= fᵢ₂_cache .* dt .+ cache.y₀.u[i]

    return nothing
end
@views function sum_stages!(
        z::AbstractArray, cache::MIRKCache{iip, T, use_both, NoDiffCacheNeeded},
        w, i::Int, dt = cache.mesh_dt[i]
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp, fᵢ₂_cache) = cache
    (; s_star) = cache.ITU

    fᵢ₂_cache .= zero(z)
    __maybe_matmul!(fᵢ₂_cache, k_discrete[i][:, 1:stage], w[1:stage])
    __maybe_matmul!(
        fᵢ₂_cache, k_interp.u[i][:, 1:(s_star - stage)], w[(stage + 1):s_star], true, true
    )
    z .= fᵢ₂_cache .* dt .+ cache.y₀.u[i]

    return nothing
end

@views function sum_stages!(
        z::AbstractArray, z′::AbstractArray,
        cache::MIRKCache{iip, T, use_both, DiffCacheNeeded}, w,
        w′, i::Int, dt = cache.mesh_dt[i]
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp) = cache
    (; s_star) = cache.ITU

    z .= zero(z)
    __maybe_matmul!(z, k_discrete[i].du[:, 1:stage], w[1:stage])
    __maybe_matmul!(
        z, k_interp.u[i][:, 1:(s_star - stage)], w[(stage + 1):s_star], true, true
    )
    z′ .= zero(z′)
    __maybe_matmul!(z′, k_discrete[i].du[:, 1:stage], w′[1:stage])
    __maybe_matmul!(
        z′, k_interp.u[i][:, 1:(s_star - stage)], w′[(stage + 1):s_star], true, true
    )
    z .= z .* dt .+ cache.y₀.u[i]

    return z, z′
end
@views function sum_stages!(
        z::AbstractArray, z′::AbstractArray,
        cache::MIRKCache{iip, T, use_both, NoDiffCacheNeeded}, w,
        w′, i::Int, dt = cache.mesh_dt[i]
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp) = cache
    (; s_star) = cache.ITU

    z .= zero(z)
    __maybe_matmul!(z, k_discrete[i][:, 1:stage], w[1:stage])
    __maybe_matmul!(
        z, k_interp.u[i][:, 1:(s_star - stage)], w[(stage + 1):s_star], true, true
    )
    z′ .= zero(z′)
    __maybe_matmul!(z′, k_discrete[i][:, 1:stage], w′[1:stage])
    __maybe_matmul!(
        z′, k_interp.u[i][:, 1:(s_star - stage)], w′[(stage + 1):s_star], true, true
    )
    z .= z .* dt .+ cache.y₀.u[i]

    return z, z′
end
