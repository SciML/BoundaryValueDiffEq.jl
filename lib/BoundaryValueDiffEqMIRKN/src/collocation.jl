function Φ!(residual, cache::MIRKNCache, y, u, p = cache.p)
    platform = cache.alg.platform
    kernel! = __mirkn_collocation_kernel!(platform)
    kernel!(
        residual, cache.collocation_cache, cache.k_discrete, cache.f, cache.TU,
        y, u, p, cache.mesh, cache.mesh_dt, cache.stage;
        ndrange = length(cache.k_discrete)
    )
    synchronize(platform)
    return nothing
end

@views function __mirkn_collocation_interval!(
        i, residual, collocation_cache, k_discrete, f!,
        TU::MIRKNTableau, y, u, p, mesh, mesh_dt, stage::Int
    )
    (; c, v, w, b, x, vp, bp, xp) = TU
    L = length(mesh)
    tmp = get_tmp(collocation_cache[i][1], u)
    tmpd = get_tmp(collocation_cache[i][2], u)
    dtᵢ = mesh_dt[i]
    yᵢ = get_tmp(y[i], u)
    yᵢ₊₁ = get_tmp(y[i + 1], u)
    yₗ₊ᵢ = get_tmp(y[L + i], u)
    yₗ₊ᵢ₊₁ = get_tmp(y[L + i + 1], u)
    K = get_tmp(k_discrete[i], u)

    for r in 1:stage
        for j in eachindex(tmp)
            stage_sum = zero(eltype(tmp))
            for s in 1:(r - 1)
                stage_sum += K[j, s] * x[r, s]
            end
            tmp[j] = (1 - v[r]) * yᵢ[j] + v[r] * yᵢ₊₁[j] +
                dtᵢ * ((c[r] - v[r] - w[r]) * yₗ₊ᵢ[j] + w[r] * yₗ₊ᵢ₊₁[j]) +
                dtᵢ^2 * stage_sum
        end
        for j in eachindex(tmpd)
            stage_sum = zero(eltype(tmpd))
            for s in 1:(r - 1)
                stage_sum += K[j, s] * xp[r, s]
            end
            tmpd[j] = (1 - vp[r]) * yₗ₊ᵢ[j] + vp[r] * yₗ₊ᵢ₊₁[j] + dtᵢ * stage_sum
        end
        f!(K[:, r], tmpd, tmp, p, mesh[i] + c[r] * dtᵢ)
    end

    residᵢ = residual[i]
    for j in eachindex(residᵢ)
        stage_sum = zero(eltype(residᵢ))
        for r in 1:stage
            stage_sum += K[j, r] * b[r]
        end
        residᵢ[j] = yᵢ₊₁[j] - yᵢ[j] - dtᵢ * yₗ₊ᵢ[j] - dtᵢ^2 * stage_sum
    end

    residₗᵢ = residual[L + i - 1]
    for j in eachindex(residₗᵢ)
        stage_sum = zero(eltype(residₗᵢ))
        for r in 1:stage
            stage_sum += K[j, r] * bp[r]
        end
        residₗᵢ[j] = yₗ₊ᵢ₊₁[j] - yₗ₊ᵢ[j] - dtᵢ * stage_sum
    end
    return nothing
end

@kernel function __mirkn_collocation_kernel!(
        residual, collocation_cache, k_discrete, f!,
        TU, y, u, p, mesh, mesh_dt, stage
    )
    i = @index(Global, Linear)
    __mirkn_collocation_interval!(
        i, residual, collocation_cache, k_discrete, f!,
        TU, y, u, p, mesh, mesh_dt, stage
    )
end

function Φ(cache::MIRKNCache, y, u, p = cache.p)
    residuals = [similar(yᵢ) for yᵢ in y[1:(end - 2)]]
    platform = cache.alg.platform
    kernel! = __mirkn_collocation_oop_kernel!(platform)
    kernel!(
        residuals, cache.collocation_cache, cache.k_discrete, cache.f, cache.TU,
        y, u, p, cache.mesh, cache.mesh_dt, cache.stage;
        ndrange = length(cache.k_discrete)
    )
    synchronize(platform)
    return residuals
end

@views function __mirkn_collocation_oop_interval!(
        i, residuals, collocation_cache, k_discrete, f,
        TU::MIRKNTableau, y, u, p, mesh, mesh_dt, stage::Int
    )
    (; c, v, w, b, x, vp, bp, xp) = TU
    L = length(mesh)
    tmp = get_tmp(collocation_cache[i][1], u)
    tmpd = get_tmp(collocation_cache[i][2], u)
    dtᵢ = mesh_dt[i]
    yᵢ = get_tmp(y[i], u)
    yᵢ₊₁ = get_tmp(y[i + 1], u)
    yₗ₊ᵢ = get_tmp(y[L + i], u)
    yₗ₊ᵢ₊₁ = get_tmp(y[L + i + 1], u)
    K = get_tmp(k_discrete[i], u)

    for r in 1:stage
        for j in eachindex(tmp)
            stage_sum = zero(eltype(tmp))
            for s in 1:(r - 1)
                stage_sum += K[j, s] * x[r, s]
            end
            tmp[j] = (1 - v[r]) * yᵢ[j] + v[r] * yᵢ₊₁[j] +
                dtᵢ * ((c[r] - v[r] - w[r]) * yₗ₊ᵢ[j] + w[r] * yₗ₊ᵢ₊₁[j]) +
                dtᵢ^2 * stage_sum
        end
        for j in eachindex(tmpd)
            stage_sum = zero(eltype(tmpd))
            for s in 1:(r - 1)
                stage_sum += K[j, s] * xp[r, s]
            end
            tmpd[j] = (1 - vp[r]) * yₗ₊ᵢ[j] + vp[r] * yₗ₊ᵢ₊₁[j] + dtᵢ * stage_sum
        end
        K[:, r] .= f(tmpd, tmp, p, mesh[i] + c[r] * dtᵢ)
    end

    residᵢ = residuals[i]
    for j in eachindex(residᵢ)
        stage_sum = zero(eltype(residᵢ))
        for r in 1:stage
            stage_sum += K[j, r] * b[r]
        end
        residᵢ[j] = yᵢ₊₁[j] - yᵢ[j] - dtᵢ * yₗ₊ᵢ[j] - dtᵢ^2 * stage_sum
    end

    residₗᵢ = residuals[L + i - 1]
    for j in eachindex(residₗᵢ)
        stage_sum = zero(eltype(residₗᵢ))
        for r in 1:stage
            stage_sum += K[j, r] * bp[r]
        end
        residₗᵢ[j] = yₗ₊ᵢ₊₁[j] - yₗ₊ᵢ[j] - dtᵢ * stage_sum
    end
    return nothing
end

@kernel function __mirkn_collocation_oop_kernel!(
        residuals, collocation_cache, k_discrete, f,
        TU, y, u, p, mesh, mesh_dt, stage
    )
    i = @index(Global, Linear)
    __mirkn_collocation_oop_interval!(
        i, residuals, collocation_cache, k_discrete, f,
        TU, y, u, p, mesh, mesh_dt, stage
    )
end
