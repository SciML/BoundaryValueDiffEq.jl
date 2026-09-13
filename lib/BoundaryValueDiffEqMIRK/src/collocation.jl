function Φ!(residual, cache::MIRKCache, y, u, trait, constraint)
    return Φ!(
        residual, cache.collocation_cache, cache.k_discrete, cache.f, cache.TU, y, u,
        cache.p, cache.mesh, cache.mesh_dt, cache.stage, cache.f_prototype,
        cache.singular_term, trait, constraint, cache.alg.platform
    )
end

@inline _collocation_tmp(cache, u, ::DiffCacheNeeded) = get_tmp(cache, u)
@inline _collocation_tmp(cache, _, ::NoDiffCacheNeeded) = cache

@views function __mirk_collocation_interval!(
        i, residual, collocation_cache, k_discrete, f!, TU::MIRKTableau, y, u, p,
        mesh, mesh_dt, stage::Int, _, singular_term, trait, ::Val{false}
    )
    (; c, v, x, b) = TU
    tmp = _collocation_tmp(collocation_cache[i], u, trait)
    K = _collocation_tmp(k_discrete[i], u, trait)
    residᵢ = residual[i]
    h = mesh_dt[i]
    yᵢ = _collocation_tmp(y[i], u, trait)
    yᵢ₊₁ = _collocation_tmp(y[i + 1], u, trait)

    for r in 1:stage
        for j in eachindex(tmp)
            stage_sum = zero(eltype(tmp))
            for s in 1:(r - 1)
                stage_sum += K[j, s] * x[r, s]
            end
            tmp[j] = (1 - v[r]) * yᵢ[j] + v[r] * yᵢ₊₁[j] + h * stage_sum
        end
        t = mesh[i] + c[r] * h
        f!(K[:, r], tmp, p, t)
        __add_singular_term!(K[:, r], singular_term, tmp, t)
    end

    for j in eachindex(residᵢ)
        stage_sum = zero(eltype(residᵢ))
        for r in 1:stage
            stage_sum += K[j, r] * b[r]
        end
        residᵢ[j] = yᵢ₊₁[j] - yᵢ[j] - h * stage_sum
    end
    return nothing
end

@views function __mirk_collocation_interval!(
        i, residual, collocation_cache, k_discrete, f!, TU::MIRKTableau, y, u, p,
        mesh, mesh_dt, stage::Int, f_prototype, _, trait, ::Val{true}
    )
    (; c, v, x, b) = TU
    L_f_prototype = length(f_prototype)
    tmp = _collocation_tmp(collocation_cache[i], u, trait)
    tmpy = tmp[1:L_f_prototype]
    tmpu = tmp[(L_f_prototype + 1):end]
    K = _collocation_tmp(k_discrete[i], u, trait)
    residᵢ = residual[i]
    h = mesh_dt[i]
    yᵢ = _collocation_tmp(y[i], u, trait)
    yᵢ₊₁ = _collocation_tmp(y[i + 1], u, trait)
    yᵢ_state = yᵢ[1:L_f_prototype]
    yᵢ_control = yᵢ[(L_f_prototype + 1):end]
    yᵢ₊₁_state = yᵢ₊₁[1:L_f_prototype]
    yᵢ₊₁_control = yᵢ₊₁[(L_f_prototype + 1):end]

    for r in 1:stage
        for j in eachindex(tmpy)
            stage_sum = zero(eltype(tmpy))
            for s in 1:(r - 1)
                stage_sum += K[j, s] * x[r, s]
            end
            tmpy[j] = (1 - v[r]) * yᵢ_state[j] + v[r] * yᵢ₊₁_state[j] + h * stage_sum
        end
        for j in eachindex(tmpu)
            tmpu[j] = (1 - v[r]) * yᵢ_control[j] + v[r] * yᵢ₊₁_control[j]
        end
        f!(K[:, r], tmp, p, mesh[i] + c[r] * h)
    end

    for j in eachindex(residᵢ)
        stage_sum = zero(eltype(residᵢ))
        for r in 1:stage
            stage_sum += K[j, r] * b[r]
        end
        residᵢ[j] = yᵢ₊₁_state[j] - yᵢ_state[j] - h * stage_sum
    end
    return nothing
end

@kernel function __mirk_collocation_kernel!(
        residual, collocation_cache, k_discrete, f!, TU, y, u, p, mesh, mesh_dt,
        stage, f_prototype, singular_term, trait, constraint
    )
    i = @index(Global, Linear)
    __mirk_collocation_interval!(
        i, residual, collocation_cache, k_discrete, f!, TU, y, u, p, mesh, mesh_dt,
        stage, f_prototype, singular_term, trait, constraint
    )
end

function Φ!(
        residual, collocation_cache, k_discrete, f!, TU::MIRKTableau, y, u, p,
        mesh, mesh_dt, stage::Int, f_prototype, singular_term, trait, constraint,
        platform::Backend
    )
    kernel! = __mirk_collocation_kernel!(platform)
    kernel!(
        residual, collocation_cache, k_discrete, f!, TU, y, u, p, mesh, mesh_dt,
        stage, f_prototype, singular_term, trait, constraint;
        ndrange = length(k_discrete)
    )
    synchronize(platform)
    return nothing
end

function Φ(cache::MIRKCache, y, u, trait)
    return Φ(
        cache.collocation_cache, cache.k_discrete, cache.f, cache.TU, y, u,
        cache.p, cache.mesh, cache.mesh_dt, cache.stage, cache.singular_term, trait,
        cache.alg.platform
    )
end

@views function __mirk_collocation_oop_interval!(
        i, residuals, collocation_cache, k_discrete, f, TU::MIRKTableau, y, u, p,
        mesh, mesh_dt, stage::Int, singular_term, trait
    )
    (; c, v, x, b) = TU
    tmp = _collocation_tmp(collocation_cache[i], u, trait)
    K = _collocation_tmp(k_discrete[i], u, trait)
    residᵢ = residuals[i]
    h = mesh_dt[i]
    yᵢ = _collocation_tmp(y[i], u, trait)
    yᵢ₊₁ = _collocation_tmp(y[i + 1], u, trait)

    for r in 1:stage
        for j in eachindex(tmp)
            stage_sum = zero(eltype(tmp))
            for s in 1:(r - 1)
                stage_sum += K[j, s] * x[r, s]
            end
            tmp[j] = (1 - v[r]) * yᵢ[j] + v[r] * yᵢ₊₁[j] + h * stage_sum
        end
        t = mesh[i] + c[r] * h
        K[:, r] .= f(tmp, p, t)
        __add_singular_term!(K[:, r], singular_term, tmp, t)
    end

    for j in eachindex(residᵢ)
        stage_sum = zero(eltype(residᵢ))
        for r in 1:stage
            stage_sum += K[j, r] * b[r]
        end
        residᵢ[j] = yᵢ₊₁[j] - yᵢ[j] - h * stage_sum
    end
    return nothing
end

@kernel function __mirk_collocation_oop_kernel!(
        residuals, collocation_cache, k_discrete, f, TU, y, u, p, mesh, mesh_dt,
        stage, singular_term, trait
    )
    i = @index(Global, Linear)
    __mirk_collocation_oop_interval!(
        i, residuals, collocation_cache, k_discrete, f, TU, y, u, p, mesh, mesh_dt,
        stage, singular_term, trait
    )
end

function Φ(
        collocation_cache, k_discrete, f, TU::MIRKTableau, y, u, p,
        mesh, mesh_dt, stage::Int, singular_term, trait, platform::Backend
    )
    residuals = [safe_similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    kernel! = __mirk_collocation_oop_kernel!(platform)
    kernel!(
        residuals, collocation_cache, k_discrete, f, TU, y, u, p, mesh, mesh_dt,
        stage, singular_term, trait;
        ndrange = length(k_discrete)
    )
    synchronize(platform)
    return residuals
end
