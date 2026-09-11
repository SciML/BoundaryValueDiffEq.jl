function Φ!(residual, cache::FIRKCacheExpand, y, u, trait, constraint)
    return __firk_collocation!(residual, cache, y, u, trait, Val(true), constraint)
end

function Φ!(residual, cache::FIRKCacheNested, y, u, trait, constraint)
    return __firk_collocation!(residual, cache, y, u, trait)
end

@inline _collocation_tmp(cache, u, ::DiffCacheNeeded) = get_tmp(cache, u)
@inline _collocation_tmp(cache, _, ::NoDiffCacheNeeded) = cache

@views function __firk_collocation_interval!(
        i, residual, collocation_cache, k_discrete, f, TU::FIRKTableau{false}, y, u,
        p, mesh, mesh_dt, stage::Int, f_prototype, singular_term, trait,
        ::Val{iip}, ::Val{constraint}
    ) where {iip, constraint}
    (; c, a, b) = TU
    tmp = _collocation_tmp(collocation_cache[i], u, trait)
    K = _collocation_tmp(k_discrete[i], u, trait)
    ctr = (i - 1) * (stage + 1) + 1
    h = mesh_dt[i]
    yᵢ = _collocation_tmp(y[ctr], u, trait)
    yᵢ₊₁ = _collocation_tmp(y[ctr + stage + 1], u, trait)
    nstate = constraint ? length(f_prototype) : length(yᵢ)

    # Each interval owns its stage matrix and writes a disjoint residual block.
    for r in 1:stage
        stage_y = _collocation_tmp(y[ctr + r], u, trait)
        for j in 1:nstate
            K[j, r] = stage_y[j]
        end
    end

    for r in 1:stage
        for j in 1:nstate
            stage_sum = zero(eltype(tmp))
            for s in 1:stage
                stage_sum += K[j, s] * a[s, r]
            end
            tmp[j] = yᵢ[j] + h * stage_sum
        end
        if constraint
            for j in (nstate + 1):length(tmp)
                tmp[j] = yᵢ[j]
            end
        end
        t = mesh[i] + c[r] * h
        stage_resid = residual[ctr + r]
        if iip
            f(stage_resid, tmp, p, t)
        else
            stage_resid .= f(tmp, p, t)
        end
        if !constraint
            __add_singular_term!(stage_resid, singular_term, tmp, t)
        end
        for j in eachindex(stage_resid)
            stage_resid[j] -= K[j, r]
        end
    end

    residᵢ = residual[ctr]
    for j in eachindex(residᵢ)
        stage_sum = zero(eltype(residᵢ))
        for r in 1:stage
            stage_sum += K[j, r] * b[r]
        end
        residᵢ[j] = yᵢ₊₁[j] - yᵢ[j] - h * stage_sum
    end
    return nothing
end

@kernel function __firk_collocation_kernel!(
        residual, collocation_cache, k_discrete, f, TU, y, u, p, mesh, mesh_dt,
        stage, f_prototype, singular_term, trait, iip, constraint
    )
    i = @index(Global, Linear)
    __firk_collocation_interval!(
        i, residual, collocation_cache, k_discrete, f, TU, y, u, p, mesh, mesh_dt,
        stage, f_prototype, singular_term, trait, iip, constraint
    )
end

function __firk_collocation!(residual, cache::FIRKCacheExpand, y, u, trait, iip, constraint)
    platform = cache.alg.platform
    kernel! = __firk_collocation_kernel!(platform)
    kernel!(
        residual, cache.collocation_cache, cache.k_discrete, cache.f, cache.TU, y, u,
        cache.p, cache.mesh, cache.mesh_dt, cache.stage, cache.f_prototype,
        cache.singular_term, trait, iip, constraint;
        ndrange = length(cache.mesh_dt)
    )
    synchronize(platform)
    return nothing
end

@views function __firk_collocation_nested_interval!(
        i, residual, collocation_cache, k_discrete, TU::FIRKTableau{true}, y, u,
        mesh, mesh_dt, nest_prob, nest_nlsolve_alg, nested_nlsolve_kwargs, trait
    )
    (; b) = TU
    nestprob_p = _collocation_tmp(collocation_cache[i], u, trait)
    yᵢ = _collocation_tmp(y[i], u, trait)
    yᵢ₊₁ = _collocation_tmp(y[i + 1], u, trait)
    h = mesh_dt[i]
    nestprob_p[1] = mesh[i]
    nestprob_p[2] = h
    nestprob_p[3:end] .= yᵢ

    # Keep the initial guess private even when the nested solver aliases u0.
    nestprob = remake(nest_prob; u0 = copy(nest_prob.u0), p = nestprob_p)
    nestsol = if trait isa DiffCacheNeeded
        __solve(nestprob, nest_nlsolve_alg; nested_nlsolve_kwargs...)
    else
        solve(nestprob, nest_nlsolve_alg; nested_nlsolve_kwargs...)
    end
    K = _collocation_tmp(k_discrete[i], u, trait)
    K .= nestsol.u
    residᵢ = residual[i]
    for j in eachindex(residᵢ)
        stage_sum = zero(eltype(residᵢ))
        for r in eachindex(b)
            stage_sum += K[j, r] * b[r]
        end
        residᵢ[j] = yᵢ₊₁[j] - yᵢ[j] - h * stage_sum
    end
    return nothing
end

@kernel function __firk_collocation_nested_kernel!(
        residual, collocation_cache, k_discrete, TU, y, u, mesh, mesh_dt,
        nest_prob, nest_nlsolve_alg, nested_nlsolve_kwargs, trait
    )
    i = @index(Global, Linear)
    __firk_collocation_nested_interval!(
        i, residual, collocation_cache, k_discrete, TU, y, u, mesh, mesh_dt,
        nest_prob, nest_nlsolve_alg, nested_nlsolve_kwargs, trait
    )
end

function __firk_collocation!(residual, cache::FIRKCacheNested, y, u, trait)
    (; alg, nest_prob) = cache
    nest_nlsolve_alg = __concrete_solve_algorithm(nest_prob, alg.nlsolve)
    kernel! = __firk_collocation_nested_kernel!(alg.platform)
    kernel!(
        residual, cache.collocation_cache, cache.k_discrete, cache.TU, y, u,
        cache.mesh, cache.mesh_dt, nest_prob, nest_nlsolve_alg,
        alg.nested_nlsolve_kwargs, trait;
        ndrange = length(cache.mesh_dt)
    )
    synchronize(alg.platform)
    return nothing
end

function Φ(cache::FIRKCacheExpand, y, u, trait)
    residuals = [safe_similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    __firk_collocation!(residuals, cache, y, u, trait, Val(false), Val(false))
    return residuals
end

function Φ(cache::FIRKCacheNested, y, u, trait)
    residuals = [safe_similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    __firk_collocation!(residuals, cache, y, u, trait)
    return residuals
end

function FIRK_nlsolve!(res, K, p_nlsolve, f!, TU::FIRKTableau{true}, p_f!)
    (; a, c, s) = TU
    mesh_i = p_nlsolve[1]
    h = p_nlsolve[2]
    yᵢ = @view p_nlsolve[3:end]

    T = promote_type(eltype(K), eltype(yᵢ))
    tmp1 = similar(K, T, size(K, 1))

    for r in 1:s
        @. tmp1 = T.(yᵢ)
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
