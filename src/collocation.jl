__initial_state_from_prob(prob::BVProblem, mesh) = __initial_state_from_prob(prob.u0, mesh)
__initial_state_from_prob(u0::AbstractArray, mesh) = [copy(vec(u0)) for _ in mesh]
function __initial_state_from_prob(u0::AbstractVector{<:AbstractVector}, _)
    [copy(vec(u)) for u in u0]
end

function Φ!(residual, cache::RKCache, y, u, p = cache.p)
    return Φ!(residual, cache.fᵢ_cache, cache.k_discrete, cache.f, cache.TU,
              y, u, p, cache.mesh, cache.mesh_dt, cache.stage)
end

@views function Φ!(residual, fᵢ_cache, k_discrete, f!, TU::MIRKTableau, y, u, p,
                   mesh, mesh_dt, stage::Int)
    @unpack c, v, x, b = TU

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

@views function Φ!(residual, fᵢ_cache, k_discrete, f!, TU::RKTableau, y, u, p,
                   mesh, mesh_dt, stage::Int)
    @unpack c, a, b = TU
    tmp1 = get_tmp(fᵢ_cache, u)
    tmp2 = get_tmp(fᵢ_cache, u)
    K = get_tmp(k_discrete[1], u) # Not optimal
    T = eltype(u)
    ctr = 1
    for i in eachindex(k_discrete)
        h = mesh_dt[i]
        yᵢ = get_tmp(y[ctr], u)
        yᵢ₊₁ = get_tmp(y[ctr + stage + 1], u)

        # Load interpolation residual
        for j in 1:stage
            K[:,j] = get_tmp(y[ctr + j], u)
        end

        # Update interpolation residual
        for r in 1:stage
            @. tmp1 = yᵢ
            __maybe_matmul!(tmp1, K[:, 1:stage], a[r, 1:stage], h, T(1))
            f!(residual[ctr + r], tmp1, p, mesh[i] + c[r] * h)
            residual[ctr + r] -= K[:,r]
        end

        # Update mesh point residual
        residᵢ = residual[ctr]
        @. residᵢ = yᵢ₊₁ - yᵢ
        __maybe_matmul!(residᵢ, K[:, 1:stage], b[1:stage], -h, T(1))
        ctr += stage + 1
    end
end

function Φ(cache::RKCache, y, u, p = cache.p)
    return Φ(cache.fᵢ_cache, cache.k_discrete, cache.f, cache.TU, y, u, p, cache.mesh,
             cache.mesh_dt, cache.stage)
end

@views function Φ(fᵢ_cache, k_discrete, f, TU::MIRKTableau, y, u, p, mesh, mesh_dt,
                  stage::Int)
    @unpack c, v, x, b = TU
    residuals = [similar(yᵢ) for yᵢ in y[1:(end - 1)]]
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
