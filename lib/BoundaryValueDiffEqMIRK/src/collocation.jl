function Φ!(residual, cache::MIRKCache, y, u, trait, constraint)
    return Φ!(
        residual, cache.fᵢ_cache, cache.k_discrete, cache.f, cache.TU, y, u, cache.p,
        cache.mesh, cache.mesh_dt, cache.stage, cache.f_prototype, cache.singular_term, trait, constraint
    )
end

@views function Φ!(
        residual, fᵢ_cache, k_discrete, f!, TU::MIRKTableau, y, u, p, mesh,
        mesh_dt, stage::Int, f_prototype, singular_term, ::DiffCacheNeeded, ::Val{true}
    )
    (; c, v, x, b) = TU
    L_f_prototype = length(f_prototype)

    tmpy,
        tmpu = get_tmp(fᵢ_cache, u)[1:L_f_prototype],
        get_tmp(fᵢ_cache, u)[(L_f_prototype + 1):end]

    T = eltype(u)
    for i in eachindex(k_discrete)
        K = get_tmp(k_discrete[i], u)
        residᵢ = residual[i]
        h = mesh_dt[i]

        yᵢ = get_tmp(y[i], u)
        yᵢ₊₁ = get_tmp(y[i + 1], u)

        yᵢ, uᵢ = yᵢ[1:L_f_prototype], yᵢ[(L_f_prototype + 1):end]
        yᵢ₊₁, uᵢ₊₁ = yᵢ₊₁[1:L_f_prototype], yᵢ₊₁[(L_f_prototype + 1):end]

        for r in 1:stage
            @. tmpy = (1 - v[r]) * yᵢ + v[r] * yᵢ₊₁
            @. tmpu = (1 - v[r]) * uᵢ + v[r] * uᵢ₊₁
            __maybe_matmul!(tmpy, K[:, 1:(r - 1)], x[r, 1:(r - 1)], h, T(1))
            f!(K[:, r], vcat(tmpy, tmpu), p, mesh[i] + c[r] * h)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, K[:, 1:stage], b[1:stage], -h, T(1))
    end
end

@views function Φ!(
        residual, fᵢ_cache, k_discrete, f!, TU::MIRKTableau, y, u, p, mesh,
        mesh_dt, stage::Int, _, singular_term, ::DiffCacheNeeded, constraint::Val{false}
    )
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
            t = mesh[i] + c[r] * h
            f!(K[:, r], tmp, p, t)
            __add_singular_term!(K[:, r], singular_term, tmp, t)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, K[:, 1:stage], b[1:stage], -h, T(1))
    end
end

@views function Φ!(
        residual, fᵢ_cache, k_discrete, f!, TU::MIRKTableau, y, u, p,
        mesh, mesh_dt, stage::Int, _, singular_term, ::NoDiffCacheNeeded, ::Val{false}
    )
    (; c, v, x, b) = TU

    tmp = similar(fᵢ_cache)
    T = eltype(u)
    for i in eachindex(k_discrete)
        K = k_discrete[i]
        residᵢ = residual[i]
        h = mesh_dt[i]

        yᵢ = y[i]
        yᵢ₊₁ = y[i + 1]

        for r in 1:stage
            @. tmp = (1 - v[r]) * yᵢ + v[r] * yᵢ₊₁
            __maybe_matmul!(tmp, K[:, 1:(r - 1)], x[r, 1:(r - 1)], h, T(1))
            t = mesh[i] + c[r] * h
            f!(K[:, r], tmp, p, t)
            __add_singular_term!(K[:, r], singular_term, tmp, t)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, K[:, 1:stage], b[1:stage], -h, T(1))
    end
end

function Φ(cache::MIRKCache, y, u, trait)
    return Φ(
        cache.fᵢ_cache, cache.k_discrete, cache.f, cache.TU, y, u,
        cache.p, cache.mesh, cache.mesh_dt, cache.stage, cache.singular_term, trait
    )
end

@views function Φ(
        fᵢ_cache, k_discrete, f, TU::MIRKTableau, y, u,
        p, mesh, mesh_dt, stage::Int, singular_term, ::DiffCacheNeeded
    )
    (; c, v, x, b) = TU
    residuals = [safe_similar(yᵢ) for yᵢ in y[1:(end - 1)]]
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
            t = mesh[i] + c[r] * h
            K[:, r] .= f(tmp, p, t)
            __add_singular_term!(K[:, r], singular_term, tmp, t)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, K[:, 1:stage], b[1:stage], -h, T(1))
    end

    return residuals
end

@views function Φ(
        fᵢ_cache, k_discrete, f, TU::MIRKTableau, y, u, p,
        mesh, mesh_dt, stage::Int, singular_term, ::NoDiffCacheNeeded
    )
    (; c, v, x, b) = TU
    residuals = [safe_similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    tmp = similar(fᵢ_cache)
    T = eltype(u)
    for i in eachindex(k_discrete)
        K = k_discrete[i]
        residᵢ = residuals[i]
        h = mesh_dt[i]

        yᵢ = y[i]
        yᵢ₊₁ = y[i + 1]

        for r in 1:stage
            @. tmp = (1 - v[r]) * yᵢ + v[r] * yᵢ₊₁
            __maybe_matmul!(tmp, K[:, 1:(r - 1)], x[r, 1:(r - 1)], h, T(1))
            t = mesh[i] + c[r] * h
            K[:, r] .= f(tmp, p, t)
            __add_singular_term!(K[:, r], singular_term, tmp, t)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, K[:, 1:stage], b[1:stage], -h, T(1))
    end

    return residuals
end
