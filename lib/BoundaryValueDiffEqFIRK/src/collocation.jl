function Φ!(residual, cache::FIRKCacheExpand, y, u, p = cache.p)
    return Φ!(residual, cache.fᵢ_cache, cache.k_discrete, cache.f,
        cache.TU, y, u, p, cache.mesh, cache.mesh_dt, cache.stage)
end

function Φ!(residual, cache::FIRKCacheNested, y, u, p = cache.p)
    return Φ!(residual, cache.fᵢ_cache, cache.k_discrete, cache.f, cache.TU,
        y, u, p, cache.mesh, cache.mesh_dt, cache.stage, cache)
end

@views function Φ!(residual, fᵢ_cache, k_discrete, f!, TU::FIRKTableau{false},
        y, u, p, mesh, mesh_dt, stage::Int)
    (; c, a, b) = TU
    tmp1 = get_tmp(fᵢ_cache, u)
    K = get_tmp(k_discrete[1], u) # Not optimal # TODO
    T = eltype(u)
    ctr = 1

    for i in eachindex(mesh_dt)
        h = mesh_dt[i]
        yᵢ = get_tmp(y[ctr], u)
        yᵢ₊₁ = get_tmp(y[ctr + stage + 1], u)

        # Load interpolation residual
        for j in 1:stage
            K[:, j] = get_tmp(y[ctr + j], u)
        end

        # Update interpolation residual
        for r in 1:stage
            @. tmp1 = yᵢ
            __maybe_matmul!(tmp1, K, a[:, r], h, T(1))
            f!(residual[ctr + r], tmp1, p, mesh[i] + c[r] * h)
            residual[ctr + r] .-= K[:, r]
        end

        # Update mesh point residual
        residᵢ = residual[ctr]
        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, K, b, -h, T(1))
        ctr += stage + 1
    end
end

function FIRK_nlsolve!(res, K, p_nlsolve, f!, TU::FIRKTableau{true}, p_f!)
    (; a, c, s) = TU
    mesh_i = p_nlsolve[1]
    h = p_nlsolve[2]
    yᵢ = @view p_nlsolve[3:end]

    T = promote_type(eltype(K), eltype(yᵢ))
    tmp1 = similar(K, T, size(K, 1))

    for r in 1:s
        @. tmp1 = yᵢ
        __maybe_matmul!(tmp1, K, a[:, r], h, T(1))

        f!(@view(res[:, r]), tmp1, p_f!, mesh_i + c[r] * h)
        @views res[:, r] .-= K[:, r]
    end
    return nothing
end

function FIRK_nlsolve(K, p_nlsolve, f!, TU::FIRKTableau{true}, p_f!)
    (; a, c, s) = TU
    mesh_i = p_nlsolve[1]
    h = p_nlsolve[2]
    yᵢ = @view p_nlsolve[3:end]

    T = promote_type(eltype(K), eltype(yᵢ))
    tmp1 = similar(K, T, size(K, 1))
    res = similar(K, T, size(K))

    for r in 1:s
        @. tmp1 = yᵢ
        __maybe_matmul!(tmp1, K, a[:, r], h, T(1))
        @views res[:, r] = f!(tmp1, p_f!, mesh_i + c[r] * h)
        @views res[:, r] .-= K[:, r]
    end
    return res
end

@views function Φ!(residual, fᵢ_cache, k_discrete, f!, TU::FIRKTableau{true},
        y, u, p, mesh, mesh_dt, stage::Int, cache)
    (; b) = TU
    (; nest_prob, nest_tol) = cache

    T = eltype(u)
    nestprob_p = vcat(T(mesh[1]), T(mesh_dt[1]), get_tmp(y[1], u))
    nest_nlsolve_alg = __concrete_nonlinearsolve_algorithm(nest_prob, cache.alg.nlsolve)

    for i in eachindex(k_discrete)
        residᵢ = residual[i]
        h = mesh_dt[i]

        yᵢ = get_tmp(y[i], u)
        yᵢ₊₁ = get_tmp(y[i + 1], u)

        nestprob_p[1] = T(mesh[i])
        nestprob_p[2] = T(mesh_dt[i])
        nestprob_p[3:end] = yᵢ

        K = get_tmp(k_discrete[i], u)

        _nestprob = remake(nest_prob, p = nestprob_p)
        nestsol = solve(_nestprob, nest_nlsolve_alg; abstol = nest_tol)
        @. K = nestsol.u
        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, nestsol.u, b, -h, T(1))
    end
end

function Φ(cache::FIRKCacheExpand, y, u, p = cache.p)
    return Φ(cache.fᵢ_cache, cache.k_discrete, cache.f, cache.TU,
        y, u, p, cache.mesh, cache.mesh_dt, cache.stage)
end

function Φ(cache::FIRKCacheNested, y, u, p = cache.p)
    return Φ(cache.fᵢ_cache, cache.k_discrete, cache.f, cache.TU, y,
        u, p, cache.mesh, cache.mesh_dt, cache.stage, cache)
end

@views function Φ(
        fᵢ_cache, k_discrete, f, TU::FIRKTableau{false}, y, u, p, mesh, mesh_dt, stage::Int)
    (; c, a, b) = TU
    residuals = [__similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    tmp1 = get_tmp(fᵢ_cache, u)
    K = get_tmp(k_discrete[1], u) # Not optimal # TODO
    T = eltype(u)
    ctr = 1

    for i in eachindex(mesh_dt)
        h = mesh_dt[i]
        yᵢ = get_tmp(y[ctr], u)
        yᵢ₊₁ = get_tmp(y[ctr + stage + 1], u)

        # Load interpolation residual
        for j in 1:stage
            K[:, j] = get_tmp(y[ctr + j], u)
        end

        # Update interpolation residual
        for r in 1:stage
            @. tmp1 = yᵢ
            __maybe_matmul!(tmp1, K, a[:, r], h, T(1))
            residuals[ctr + r] = f(tmp1, p, mesh[i] + c[r] * h)
            residuals[ctr + r] .-= K[:, r]
        end

        # Update mesh point residual
        residᵢ = residuals[ctr]
        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, K, b, -h, T(1))
        ctr += stage + 1
    end
    return residuals
end

@views function Φ(fᵢ_cache, k_discrete, f!, TU::FIRKTableau{true},
        y, u, p, mesh, mesh_dt, stage::Int, cache)
    (; b) = TU
    (; nest_prob, alg, nest_tol) = cache

    residuals = [__similar(yᵢ) for yᵢ in y[1:(end - 1)]]

    T = eltype(u)
    nestprob_p = vcat(T(mesh[1]), T(mesh_dt[1]), get_tmp(y[1], u))
    nest_nlsolve_alg = __concrete_nonlinearsolve_algorithm(nest_prob, alg.nlsolve)

    for i in eachindex(k_discrete)
        residᵢ = residuals[i]
        h = mesh_dt[i]

        yᵢ = get_tmp(y[i], u)
        yᵢ₊₁ = get_tmp(y[i + 1], u)

        nestprob_p[1] = T(mesh[i])
        nestprob_p[2] = T(mesh_dt[i])
        nestprob_p[3:end] = yᵢ

        _nestprob = remake(nest_prob, p = nestprob_p)
        nestsol = solve(_nestprob, nest_nlsolve_alg, abstol = nest_tol)

        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, nestsol.u, b, -h, T(1))
    end
    return residuals
end
