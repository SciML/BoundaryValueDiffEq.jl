function Φ!(residual, cache::MIRKCache, y, u, p = cache.p)
    return Φ!(residual, cache.fᵢ_cache, cache.k_discrete, cache.f,
        cache.TU, y, u, p, cache.mesh, cache.mesh_dt, cache.stage)
end

@views function Φ!(residual, fᵢ_cache, k_discrete, f!, TU::MIRKTableau,
        y, u, p, mesh, mesh_dt, stage::Int)
    (; c, v, x, b) = TU

    tmp = get_tmp(fᵢ_cache, u)
    T = eltype(u)
    for i in eachindex(k_discrete)
        K = get_tmp(k_discrete[i], u)
        residᵢ = residual[i]
        h = mesh_dt[i]

        yᵢ = get_tmp(y[i], u)
        yᵢ₊₁ = get_tmp(y[i + 1], u)

        for r in 1:stage
            @. tmp = (1 - v[r]) * yᵢ + v[r] * yᵢ₊₁
            __maybe_matmul!(tmp, K[:, 1:(r - 1)], x[r, 1:(r - 1)], h, T(1))
            f!(K[:, r], tmp, p, mesh[i] + c[r] * h)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, K[:, 1:stage], b[1:stage], -h, T(1))
    end
end

function Φ(cache::MIRKCache, y, u, p = cache.p)
    return Φ(cache.fᵢ_cache, cache.k_discrete, cache.f, cache.TU,
        y, u, p, cache.mesh, cache.mesh_dt, cache.stage)
end

@views function Φ(
        fᵢ_cache, k_discrete, f, TU::MIRKTableau, y, u, p, mesh, mesh_dt, stage::Int)
    (; c, v, x, b) = TU
    residuals = [__similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    tmp = get_tmp(fᵢ_cache, u)
    T = eltype(u)
    for i in eachindex(k_discrete)
        K = get_tmp(k_discrete[i], u)
        residᵢ = residuals[i]
        h = mesh_dt[i]

        yᵢ = get_tmp(y[i], u)
        yᵢ₊₁ = get_tmp(y[i + 1], u)

        for r in 1:stage
            @. tmp = (1 - v[r]) * yᵢ + v[r] * yᵢ₊₁
            __maybe_matmul!(tmp, K[:, 1:(r - 1)], x[r, 1:(r - 1)], h, T(1))
            K[:, r] .= f(tmp, p, mesh[i] + c[r] * h)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, K[:, 1:stage], b[1:stage], -h, T(1))
    end

    return residuals
end
