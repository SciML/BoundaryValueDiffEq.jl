function Φ!(residual, cache::MIRKCache, y, u, trait, constraint)
    return Φ!(
        residual, cache.fᵢ_cache, cache.k_discrete, cache.f, cache.TU, y, u, cache.p,
        cache.mesh, cache.mesh_dt, cache.stage, cache.f_prototype, cache.singular_term, trait, constraint
    )
end

function Φ!(
        residual, fᵢ_cache, k_discrete, f!, TU::MIRKTableau, y, u, p, mesh,
        mesh_dt, stage::Int, f_prototype, singular_term, ::DiffCacheNeeded, ::Val{true}
    )
    (; c, v, x, b) = TU
    L_f_prototype = length(f_prototype)

    tmpy = @view get_tmp(fᵢ_cache, u)[1:L_f_prototype]
    tmpu = @view get_tmp(fᵢ_cache, u)[(L_f_prototype + 1):end]

    T = eltype(u)
    for i in eachindex(k_discrete)
        residᵢ = _maybe_get_tmp(residual[i], u)
        h = mesh_dt[i]

        yᵢ = get_tmp(y[i], u)
        yᵢ₊₁ = get_tmp(y[i + 1], u)

        yᵢ_f = @view yᵢ[1:L_f_prototype]
        uᵢ = @view yᵢ[(L_f_prototype + 1):end]
        yᵢ₊₁_f = @view yᵢ₊₁[1:L_f_prototype]
        uᵢ₊₁ = @view yᵢ₊₁[(L_f_prototype + 1):end]

        for r in 1:stage
            @. tmpy = (1 - v[r]) * yᵢ_f + v[r] * yᵢ₊₁_f
            @. tmpu = (1 - v[r]) * uᵢ + v[r] * uᵢ₊₁
            @inbounds for j in 1:(r - 1)
                Kⱼ = get_tmp(k_discrete[i][j], u)
                xrj = h * x[r, j]
                @simd ivdep for k in 1:L_f_prototype
                    tmpy[k] += Kⱼ[k] * xrj
                end
            end
            K_r = get_tmp(k_discrete[i][r], u)
            f!(K_r, vcat(tmpy, tmpu), p, mesh[i] + c[r] * h)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁_f - yᵢ_f
        @inbounds for j in 1:stage
            Kⱼ = get_tmp(k_discrete[i][j], u)
            mbhj = -h * b[j]
            @simd ivdep for k in 1:L_f_prototype
                residᵢ[k] += Kⱼ[k] * mbhj
            end
        end
    end
end

function Φ!(
        residual, fᵢ_cache, k_discrete, f!, TU::MIRKTableau, y, u, p, mesh,
        mesh_dt, stage::Int, _, singular_term, ::DiffCacheNeeded, constraint::Val{false}
    )
    (; c, v, x, b) = TU

    tmp = get_tmp(fᵢ_cache, u)
    N = length(tmp)
    T = eltype(u)
    for i in eachindex(k_discrete)
        residᵢ = _maybe_get_tmp(residual[i], u)
        h = mesh_dt[i]

        yᵢ = get_tmp(y[i], u)
        yᵢ₊₁ = get_tmp(y[i + 1], u)

        for r in 1:stage
            @. tmp = (1 - v[r]) * yᵢ + v[r] * yᵢ₊₁
            @inbounds for j in 1:(r - 1)
                Kⱼ = get_tmp(k_discrete[i][j], u)
                xrj = h * x[r, j]
                @simd ivdep for k in 1:N
                    tmp[k] += Kⱼ[k] * xrj
                end
            end
            t = mesh[i] + c[r] * h
            K_r = get_tmp(k_discrete[i][r], u)
            f!(K_r, tmp, p, t)
            __add_singular_term!(K_r, singular_term, tmp, t)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        @inbounds for j in 1:stage
            Kⱼ = get_tmp(k_discrete[i][j], u)
            mbhj = -h * b[j]
            @simd ivdep for k in 1:N
                residᵢ[k] += Kⱼ[k] * mbhj
            end
        end
    end
end

function Φ!(
        residual, fᵢ_cache, k_discrete, f!, TU::MIRKTableau, y, u, p,
        mesh, mesh_dt, stage::Int, _, singular_term, ::NoDiffCacheNeeded, ::Val{false}
    )
    (; c, v, x, b) = TU

    tmp = fᵢ_cache
    N = length(tmp)
    T = eltype(u)
    for i in eachindex(k_discrete)
        residᵢ = residual[i]
        h = mesh_dt[i]

        yᵢ = y[i]
        yᵢ₊₁ = y[i + 1]

        for r in 1:stage
            @. tmp = (1 - v[r]) * yᵢ + v[r] * yᵢ₊₁
            @inbounds for j in 1:(r - 1)
                Kⱼ = k_discrete[i][j]
                xrj = h * x[r, j]
                @simd ivdep for k in 1:N
                    tmp[k] += Kⱼ[k] * xrj
                end
            end
            t = mesh[i] + c[r] * h
            K_r = k_discrete[i][r]
            f!(K_r, tmp, p, t)
            __add_singular_term!(K_r, singular_term, tmp, t)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        @inbounds for j in 1:stage
            Kⱼ = k_discrete[i][j]
            mbhj = -h * b[j]
            @simd ivdep for k in 1:N
                residᵢ[k] += Kⱼ[k] * mbhj
            end
        end
    end
end

function Φ(cache::MIRKCache, y, u, trait)
    return Φ(
        cache.fᵢ_cache, cache.k_discrete, cache.f, cache.TU, y, u,
        cache.p, cache.mesh, cache.mesh_dt, cache.stage, cache.singular_term, trait
    )
end

function Φ(
        fᵢ_cache, k_discrete, f, TU::MIRKTableau, y, u,
        p, mesh, mesh_dt, stage::Int, singular_term, ::DiffCacheNeeded
    )
    (; c, v, x, b) = TU
    residuals = [safe_similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    tmp = get_tmp(fᵢ_cache, u)
    N = length(tmp)
    T = eltype(u)
    for i in eachindex(k_discrete)
        residᵢ = residuals[i]
        h = mesh_dt[i]

        yᵢ = get_tmp(y[i], u)
        yᵢ₊₁ = get_tmp(y[i + 1], u)

        for r in 1:stage
            @. tmp = (1 - v[r]) * yᵢ + v[r] * yᵢ₊₁
            @inbounds for j in 1:(r - 1)
                Kⱼ = get_tmp(k_discrete[i][j], u)
                xrj = h * x[r, j]
                @simd ivdep for k in 1:N
                    tmp[k] += Kⱼ[k] * xrj
                end
            end
            t = mesh[i] + c[r] * h
            K_r = get_tmp(k_discrete[i][r], u)
            K_r .= f(tmp, p, t)
            __add_singular_term!(K_r, singular_term, tmp, t)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        @inbounds for j in 1:stage
            Kⱼ = get_tmp(k_discrete[i][j], u)
            mbhj = -h * b[j]
            @simd ivdep for k in 1:N
                residᵢ[k] += Kⱼ[k] * mbhj
            end
        end
    end

    return residuals
end

function Φ(
        fᵢ_cache, k_discrete, f, TU::MIRKTableau, y, u, p,
        mesh, mesh_dt, stage::Int, singular_term, ::NoDiffCacheNeeded
    )
    (; c, v, x, b) = TU
    residuals = [safe_similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    tmp = similar(fᵢ_cache)
    N = length(tmp)
    T = eltype(u)
    for i in eachindex(k_discrete)
        residᵢ = residuals[i]
        h = mesh_dt[i]

        yᵢ = y[i]
        yᵢ₊₁ = y[i + 1]

        for r in 1:stage
            @. tmp = (1 - v[r]) * yᵢ + v[r] * yᵢ₊₁
            @inbounds for j in 1:(r - 1)
                Kⱼ = k_discrete[i][j]
                xrj = h * x[r, j]
                @simd ivdep for k in 1:N
                    tmp[k] += Kⱼ[k] * xrj
                end
            end
            t = mesh[i] + c[r] * h
            k_discrete[i][r] .= f(tmp, p, t)
            __add_singular_term!(k_discrete[i][r], singular_term, tmp, t)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        @inbounds for j in 1:stage
            Kⱼ = k_discrete[i][j]
            mbhj = -h * b[j]
            @simd ivdep for k in 1:N
                residᵢ[k] += Kⱼ[k] * mbhj
            end
        end
    end

    return residuals
end
