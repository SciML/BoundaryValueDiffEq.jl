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

# Device state interpolation on a refined or redistributed mesh.
@kernel function __mirk_device_refine_kernel!(
        new_y, new_mesh, y, K, KI, mesh, mesh_dt, algid, in_size
    )
    index = @index(Global, Linear)
    @inbounds begin
        row = (index - 1) % size(new_y, 1) + 1
        node = (index - 1) ÷ size(new_y, 1) + 1
        sol = EvalSol(__build_interpolation(y, K, KI, mesh, mesh_dt, algid, in_size))
        new_y[row, node] = sol(new_mesh[node])[row]
    end
end

function __mirk_device_refine!(
        platform, new_y, new_mesh, y, K, KI, mesh, mesh_dt, algid, in_size
    )
    __mirk_device_refine_kernel!(platform)(
        new_y, new_mesh, y, K, KI, mesh, mesh_dt, algid, in_size;
        ndrange = length(new_y)
    )
    synchronize(platform)
    return new_y
end

function half_mesh!(
        cache::MIRKCache{iip, T, U, D, P, Y}
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    # Reuse host mesh bisection, then interpolate the new state on the device.
    host_mesh = copy(cache.host_mesh)
    half_mesh!(host_mesh, diff(host_mesh))
    return __mirk_device_remesh!(cache, host_mesh)
end

function __mirk_device_remesh!(cache, host_mesh)
    platform = cache.alg.platform
    scratch = cache.new_stages
    resize!(scratch.mesh, length(host_mesh))
    copyto!(scratch.mesh, host_mesh)
    resize!(scratch.y, cache.M * length(host_mesh))
    y = reshape(scratch.y, cache.M, length(host_mesh))
    # Finish reading the old states, mesh and stages before resizing any owner.
    __mirk_device_refine!(
        platform, y, scratch.mesh, __mirk_states(cache), __mirk_stages(cache), __mirk_interp_stages(cache),
        cache.mesh, cache.mesh_dt, Val(nameof(typeof(cache.alg))), cache.in_size
    )
    empty!(cache.device_cache)
    resize!(cache.y, length(scratch.y))
    copyto!(cache.y, scratch.y)
    resize!(cache.mesh, length(host_mesh))
    copyto!(cache.mesh, scratch.mesh)
    resize!(cache.mesh_dt, length(host_mesh) - 1)
    copyto!(cache.mesh_dt, diff(host_mesh))
    resize!(cache.host_mesh, length(host_mesh))
    copyto!(cache.host_mesh, host_mesh)
    return __expand_cache!(cache)
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

@inline __mirk_scaled_defect(derivative, value) =
    (derivative - value) / (abs(value) + one(value))

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
# Global error control
@views function error_estimate!(
        cache::MIRKCache{iip, T}, controller::GlobalErrorControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T}
    return error_estimate!(
        cache, controller, controller.method, errors, sol, nlsolve_alg, abstol
    )
end

# Defect control
@views function error_estimate!(
        cache::MIRKCache{iip, T, use_both, diffcache}, controller::DefectControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T, use_both, diffcache}
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
            yᵢ₁ = __mirk_interp_values(cache.y[i], diffcache())
            f(yᵢ₁, z, cache.p, mesh[i] + τ_star * dt)
        else
            yᵢ₁ = f(z, cache.p, mesh[i] + τ_star * dt)
        end
        __apply_mass_matrix!(z′, cache.mass_matrix)
        yᵢ₁ .= __mirk_scaled_defect.(z′, yᵢ₁)
        est₁ = maximum(abs, yᵢ₁)

        z, z′ = sum_stages!(cache, w₂, w₂′, i)
        if iip
            yᵢ₂ = __mirk_interp_values(cache.y[i + 1], diffcache())
            f(yᵢ₂, z, cache.p, mesh[i] + (T(1) - τ_star) * dt)
        else
            yᵢ₂ = f(z, cache.p, mesh[i] + (T(1) - τ_star) * dt)
        end
        __apply_mass_matrix!(z′, cache.mass_matrix)
        yᵢ₂ .= __mirk_scaled_defect.(z′, yᵢ₂)
        est₂ = maximum(abs, yᵢ₂)

        errors.u[i] .= est₁ > est₂ ? yᵢ₁ : yᵢ₂
    end

    defect_norm = maximum(Base.Fix1(maximum, abs), errors.u)

    # The defect is greater than 10%, the solution is not acceptable
    info = ifelse(defect_norm > controller.defect_threshold, ReturnCode.Failure, ReturnCode.Success)
    return defect_norm, info
end

# Defect estimation
@inline __mirk_mass_derivative(mass::LinearAlgebra.UniformScaling, derivative, row) = derivative[row]
@inline function __mirk_mass_derivative(mass::AbstractMatrix, derivative, row)
    row > size(mass, 1) && return derivative[row]
    value = zero(eltype(derivative))
    @inbounds for column in axes(mass, 2)
        value += mass[row, column] * derivative[column]
    end
    return value
end

@kernel function __mirk_device_defect_kernel!(
        errors, tmp, rhs_tmp, K, KI, y, f, p, mesh, mesh_dt, algid,
        in_size, iip, singular_term, τ_star, mass_matrix
    )
    i = @index(Global, Linear)
    @inbounds begin
        sol = EvalSol(__build_interpolation(y, K, KI, mesh, mesh_dt, algid, in_size))
        z = view(tmp, :, i)
        du = view(rhs_tmp, :, i)
        estimate = zero(eltype(errors))
        for sample in 1:2
            τ = sample == 1 ? τ_star : one(τ_star) - τ_star
            t = mesh[i] + τ * mesh_dt[i]
            value = sol(t)
            derivative = sol(t, Val(1))
            for j in eachindex(z)
                z[j] = value[j]
            end
            __device_eval!(
                __device_reshape(du, in_size), f,
                (__device_reshape(z, in_size), p, t), iip
            )
            __device_singular!(du, singular_term, z, t)
            for j in eachindex(du)
                defect = abs(__mirk_scaled_defect(__mirk_mass_derivative(mass_matrix, derivative, j), du[j]))
                estimate = max(estimate, defect)
            end
        end
        errors[i] = estimate
    end
end

function __mirk_device_defect!(
        platform, errors, tmp, rhs_tmp, K, KI, y, f, p, mesh, mesh_dt,
        algid, in_size, iip, singular_term, τ_star, mass_matrix = LinearAlgebra.I
    )
    __mirk_device_defect_kernel!(platform)(
        errors, tmp, rhs_tmp, K, KI, y, f, p, mesh, mesh_dt,
        algid, in_size, iip, singular_term, τ_star, mass_matrix; ndrange = length(mesh_dt)
    )
    synchronize(platform)
    return maximum(errors)
end

function error_estimate!(
        cache::MIRKCache{iip, T, U, D, P, Y}, controller::DefectControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    defect = __mirk_device_defect!(
        cache.alg.platform, errors, __mirk_collocation(cache), __mirk_rhs_tmp(cache),
        __mirk_stages(cache), __mirk_interp_stages(cache), __mirk_states(cache), cache.f, cache.p,
        cache.mesh, cache.mesh_dt, Val(nameof(typeof(cache.alg))),
        cache.in_size, Val(iip), cache.singular_term, cache.ITU.τ_star, cache.mass_matrix
    )
    info = !isfinite(defect) ? ReturnCode.Unstable :
        defect > controller.defect_threshold ? ReturnCode.Failure : ReturnCode.Success
    return defect, info
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
                for i in 1:(length(err.u) - 1)
        ]
    )
end

"""
    sum_stages!(cache::MIRKCache, w, w′, i::Int)

sum_stages add the discrete solution, RK method stages and extra stages to construct interpolant.
"""
function sum_stages!(
        cache::MIRKCache{iip, T, use_both, diffcache}, w, w′,
        i::Int, dt = cache.mesh_dt[i]
    ) where {iip, T, use_both, diffcache}
    return sum_stages!(__mirk_interp_values(cache.fᵢ_cache, diffcache()), cache.fᵢ₂_cache, cache, w, w′, i, dt)
end

# Here we should not directly in-place change z in several steps
# because in final step we actually need to use the original z(which is cache.y₀.u[i])
# we use fᵢ₂_cache to avoid additional allocations.
@views function sum_stages!(
        z::AbstractArray, cache::MIRKCache{iip, T, use_both, diffcache},
        w, i::Int, dt = cache.mesh_dt[i]
    ) where {iip, T, use_both, diffcache}
    (; stage, k_discrete, k_interp, fᵢ₂_cache) = cache
    (; s_star) = cache.ITU

    fᵢ₂_cache .= zero(z)
    __maybe_matmul!(fᵢ₂_cache, __mirk_interp_values(k_discrete[i], diffcache())[:, 1:stage], w[1:stage])
    __maybe_matmul!(
        fᵢ₂_cache, k_interp.u[i][:, 1:(s_star - stage)], w[(stage + 1):s_star], true, true
    )
    z .= fᵢ₂_cache .* dt .+ cache.y₀.u[i]

    return nothing
end

@views function sum_stages!(
        z::AbstractArray, z′::AbstractArray,
        cache::MIRKCache{iip, T, use_both, diffcache}, w,
        w′, i::Int, dt = cache.mesh_dt[i]
    ) where {iip, T, use_both, diffcache}
    (; stage, k_discrete, k_interp) = cache
    (; s_star) = cache.ITU

    z .= zero(z)
    __maybe_matmul!(z, __mirk_interp_values(k_discrete[i], diffcache())[:, 1:stage], w[1:stage])
    __maybe_matmul!(
        z, k_interp.u[i][:, 1:(s_star - stage)], w[(stage + 1):s_star], true, true
    )
    z′ .= zero(z′)
    __maybe_matmul!(z′, __mirk_interp_values(k_discrete[i], diffcache())[:, 1:stage], w′[1:stage])
    __maybe_matmul!(
        z′, k_interp.u[i][:, 1:(s_star - stage)], w′[(stage + 1):s_star], true, true
    )
    z .= z .* dt .+ cache.y₀.u[i]

    return z, z′
end

# Device kernels evaluate the errors and interpolate states; the host chooses
# mesh locations from one scalar error estimate per interval.
function __mirk_device_mesh_selector!(cache, controller)
    T = eltype(cache)
    errors = cache.errors
    abstol = cache.kwargs.abstol
    n = length(cache.host_mesh) - 1
    power = controller isa GlobalErrorControl ? cache.order : cache.order + 1
    weights = (errors ./ abstol) .^ (one(T) / power)
    total, largest = sum(weights), maximum(weights)
    if !isfinite(total) || !isfinite(largest)
        return cache.mesh, cache.mesh_dt, n, ReturnCode.Unstable
    end
    density_ratio = controller isa DefectControl ? one(T) : T(2)
    uniform = largest <= density_ratio * total / n
    prediction = round(Int, T(1.3) * total + 1)
    abs(prediction - n) < T(0.1) * n && (prediction = round(Int, T(1.1) * n))
    intervals = uniform ? 2n : clamp(prediction, (n + 1) ÷ 2, 4n)
    # Never coarsen while the requested tolerance is still violated.
    intervals = max(n + 1, intervals)
    if intervals > cache.alg.max_num_subintervals
        return cache.mesh, cache.mesh_dt, intervals, ReturnCode.Failure
    end
    old_mesh, old_dt = copy(cache.mesh), copy(cache.mesh_dt)
    if uniform
        half_mesh!(cache)
    else
        # These are mesh-control metadata, not state or Jacobian transfers.
        host_weights = Array(weights)
        cumulative = cumsum(host_weights)
        host_mesh = similar(cache.host_mesh, intervals + 1)
        host_mesh[1], host_mesh[end] = first(cache.host_mesh), last(cache.host_mesh)
        for node in 1:(intervals - 1)
            target = node * total / intervals
            interval = min(searchsortedfirst(cumulative, target), n)
            previous = interval == 1 ? zero(total) : cumulative[interval - 1]
            fraction = (target - previous) / host_weights[interval]
            host_mesh[node + 1] = cache.host_mesh[interval] +
                fraction * (cache.host_mesh[interval + 1] - cache.host_mesh[interval])
        end
        __mirk_device_remesh!(cache, host_mesh)
    end
    return old_mesh, old_dt, intervals, ReturnCode.Success
end

for Controller in (DefectControl, GlobalErrorControl, SequentialErrorControl, HybridErrorControl)
    @eval function mesh_selector!(
            cache::MIRKCache{iip, T, U, D, P, Y}, controller::$Controller
        ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
        return __mirk_device_mesh_selector!(cache, controller)
    end
end

@kernel function __mirk_device_global_error_kernel!(errors, high, low, stride, factor)
    interval = @index(Global, Linear)
    estimate = zero(eltype(errors))
    @inbounds for node in interval:(interval + 1), row in axes(low, 1)
        value = low[row, node]
        difference = abs(high[row, (node - 1) * stride + 1] - value) / (1 + abs(value))
        estimate = max(estimate, difference)
    end
    @inbounds errors[interval] = factor * estimate
end

function __mirk_device_global_error!(cache, method, errors, abstol)
    alg = cache.alg
    # MIRK5/6/6I have no order+2 companion; use Richardson there.
    richardson = method isa REErrorControl || cache.order >= 5
    host_mesh = copy(cache.host_mesh)
    if richardson
        half_mesh!(host_mesh, diff(host_mesh))
        mesh = __device_parameter(alg.platform, host_mesh)
        guess = similar(__mirk_states(cache), cache.M, length(host_mesh))
        __mirk_device_refine!(
            alg.platform, guess, mesh, __mirk_states(cache), __mirk_stages(cache), __mirk_interp_stages(cache),
            cache.mesh, cache.mesh_dt, Val(nameof(typeof(alg))), cache.in_size
        )
    else
        constructor = getfield(@__MODULE__, Symbol(:MIRK, cache.order + 2))
        alg = constructor(;
            nlsolve = alg.nlsolve, optimize = alg.optimize, jac_alg = alg.jac_alg,
            platform = alg.platform, defect_threshold = alg.defect_threshold,
            max_num_subintervals = alg.max_num_subintervals
        )
        guess = copy(__mirk_states(cache))
    end
    nparameters = cache.kwargs.tune_parameters ? length(cache.p) : 0
    in_size = nparameters == 0 ? cache.in_size : (cache.M - nparameters,)
    states = [reshape(copy(view(guess, 1:(cache.M - nparameters), i)), in_size) for i in axes(guess, 2)]
    parameters = nparameters == 0 ? cache.p : copy(view(__mirk_states(cache), (cache.M - nparameters + 1):cache.M, 1))
    prob = remake(cache.prob; u0 = DiffEqArray(states, host_mesh), p = parameters)
    high_cache = __init_mirk_device(
        prob, alg, first(states); dt = zero(abstol), abstol, adaptive = false,
        controller = NoErrorControl(), nlsolve_kwargs = cache.nlsolve_kwargs,
        optimize_kwargs = cache.optimize_kwargs, verbose = cache.verbose
    )
    _, info, _ = __perform_mirk_iteration(high_cache, abstol, false, NoErrorControl())
    successful_retcode(info) || return oftype(abstol, Inf), info
    factor = richardson ? eltype(errors)(2^cache.order / (2^cache.order - 1)) : one(eltype(errors))
    __mirk_device_global_error_kernel!(alg.platform)(
        errors, __mirk_states(high_cache), __mirk_states(cache), richardson ? 2 : 1, factor;
        ndrange = length(errors)
    )
    synchronize(alg.platform)
    estimate = maximum(errors)
    return estimate, isfinite(estimate) ? ReturnCode.Success : ReturnCode.Unstable
end

function error_estimate!(
        cache::MIRKCache{iip, T, U, D, P, Y}, controller::GlobalErrorControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    return __mirk_device_global_error!(cache, controller.method, errors, abstol)
end

function error_estimate!(
        cache::MIRKCache{iip, T, U, D, P, Y}, controller::SequentialErrorControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    estimate, info = error_estimate!(cache, controller.defect, errors, sol, nlsolve_alg, abstol)
    if successful_retcode(info) && estimate <= abstol
        return error_estimate!(cache, controller.global_error, errors, sol, nlsolve_alg, abstol)
    end
    return estimate, info
end

function error_estimate!(
        cache::MIRKCache{iip, T, U, D, P, Y}, controller::HybridErrorControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    defect, info = error_estimate!(cache, controller.defect, errors, sol, nlsolve_alg, abstol)
    successful_retcode(info) || return defect, info
    global_errors = similar(errors)
    global_error, info = error_estimate!(
        cache, controller.global_error, global_errors, sol, nlsolve_alg, abstol
    )
    successful_retcode(info) || return global_error, info
    errors .= controller.DE .* errors .+ controller.GE .* global_errors
    return controller.DE * defect + controller.GE * global_error, info
end

function error_estimate!(
        cache::MIRKCache{iip, T, U, D, P, Y}, ::NoErrorControl,
        errors, sol, nlsolve_alg, abstol
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    fill!(errors, zero(eltype(errors)))
    return zero(abstol), ReturnCode.Success
end
