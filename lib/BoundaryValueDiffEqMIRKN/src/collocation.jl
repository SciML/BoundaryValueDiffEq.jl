@views function Φ!(residual, cache::MIRKNCache, y, u, p = cache.p)
    return Φ!(residual, cache.fᵢ_cache, cache.fᵢ₂_cache, cache.k_discrete,
        cache.f, cache.TU, y, u, p, cache.mesh, cache.mesh_dt, cache.stage)
end

@views function Φ!(residual, fᵢ_cache, fᵢ₂_cache, k_discrete, f!,
        TU::MIRKNTableau, y, u, p, mesh, mesh_dt, stage::Int)
    (; c, v, w, b, x, vp, bp, xp) = TU
    L = length(mesh)
    T = eltype(u)
    tmp = get_tmp(fᵢ_cache, u)
    tmpd = get_tmp(fᵢ₂_cache, u)
    for i in 1:(L - 1)
        dtᵢ = mesh_dt[i]
        yᵢ = get_tmp(y[i], u)
        yᵢ₊₁ = get_tmp(y[i + 1], u)
        yₗ₊ᵢ = get_tmp(y[L + i], u)
        yₗ₊ᵢ₊₁ = get_tmp(y[L + i + 1], u)
        K = get_tmp(k_discrete[i], u)
        for r in 1:stage
            @. tmp = (1 - v[r]) * yᵢ +
                     v[r] * yᵢ₊₁ +
                     dtᵢ * ((c[r] - v[r] - w[r]) * yₗ₊ᵢ + w[r] * yₗ₊ᵢ₊₁)
            @. tmpd = (1 - vp[r]) * yₗ₊ᵢ + vp[r] * yₗ₊ᵢ₊₁
            __maybe_matmul!(tmp, K[:, 1:(r - 1)], x[r, 1:(r - 1)], dtᵢ^2, T(1))
            __maybe_matmul!(tmpd, K[:, 1:(r - 1)], xp[r, 1:(r - 1)], dtᵢ, T(1))
            f!(K[:, r], tmpd, tmp, p, mesh[i] + c[r] * dtᵢ)
        end

        # Update residual
        @. residual[i] = yᵢ₊₁ - yᵢ - dtᵢ * yₗ₊ᵢ
        __maybe_matmul!(residual[i], K[:, 1:stage], b[1:stage], -dtᵢ^2, T(1))
        @. residual[L + i - 1] = yₗ₊ᵢ₊₁ - yₗ₊ᵢ
        __maybe_matmul!(residual[L + i - 1], K[:, 1:stage], bp[1:stage], -dtᵢ, T(1))
    end
end

function Φ(cache::MIRKNCache, y, u, p = cache.p)
    return Φ(cache.fᵢ_cache, cache.fᵢ₂_cache, cache.k_discrete, cache.f,
        cache.TU, y, u, p, cache.mesh, cache.mesh_dt, cache.stage)
end

@views function Φ(fᵢ_cache, fᵢ₂_cache, k_discrete, f, TU::MIRKNTableau,
        y, u, p, mesh, mesh_dt, stage::Int)
    (; c, v, w, b, x, vp, bp, xp) = TU
    residual = [similar(yᵢ) for yᵢ in y[1:(end - 2)]]
    L = length(mesh)
    T = eltype(u)
    tmp = get_tmp(fᵢ_cache, u)
    tmpd = get_tmp(fᵢ₂_cache, u)
    for i in 1:(L - 1)
        dtᵢ = mesh_dt[i]
        yᵢ = get_tmp(y[i], u)
        yᵢ₊₁ = get_tmp(y[i + 1], u)
        yₗ₊ᵢ = get_tmp(y[L + i], u)
        yₗ₊ᵢ₊₁ = get_tmp(y[L + i + 1], u)
        K = get_tmp(k_discrete[i], u)
        for r in 1:stage
            @. tmp = (1 - v[r]) * yᵢ +
                     v[r] * yᵢ₊₁ +
                     dtᵢ * ((c[r] - v[r] - w[r]) * yₗ₊ᵢ + w[r] * yₗ₊ᵢ₊₁)
            @. tmpd = (1 - vp[r]) * yₗ₊ᵢ + vp[r] * yₗ₊ᵢ₊₁
            __maybe_matmul!(tmp, K[:, 1:(r - 1)], x[r, 1:(r - 1)], dtᵢ^2, T(1))
            __maybe_matmul!(tmpd, K[:, 1:(r - 1)], xp[r, 1:(r - 1)], dtᵢ, T(1))

            K[:, r] .= f(tmpd, tmp, p, mesh[i] + c[r] * dtᵢ)
        end
        @. residual[i] = yᵢ₊₁ - yᵢ - dtᵢ * yₗ₊ᵢ
        __maybe_matmul!(residual[i], K[:, 1:stage], b[1:stage], -dtᵢ^2, T(1))
        @. residual[L + i - 1] = yₗ₊ᵢ₊₁ - yₗ₊ᵢ
        __maybe_matmul!(residual[L + i - 1], K[:, 1:stage], bp[1:stage], -dtᵢ, T(1))
    end
    return residual
end
