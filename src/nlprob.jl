import SparseDiffTools: __init_𝒥

function construct_nlproblem(cache::RKCache, y::AbstractVector)
    function loss_bc!(resid::AbstractVector, u::AbstractVector, p = cache.p)
        y_ = recursive_unflatten!(cache.y, u)
        eval_bc_residual!(resid, cache.problem_type, cache.bc!, y_, p, cache.mesh, u)
        return resid
    end

    function loss_collocation!(resid::AbstractVector, u::AbstractVector, p = cache.p)
        y_ = recursive_unflatten!(cache.y, u)
        resids = [get_tmp(r, u) for r in cache.residual[2:end]]
        Φ!(resids, cache, y_, u, p)
        recursive_flatten!(resid, resids)
        return resid
    end

    function loss!(resid::AbstractVector, u::AbstractVector, p)
        y_ = recursive_unflatten!(cache.y, u)
        resids = [get_tmp(r, u) for r in cache.residual]
        eval_bc_residual!(resids[1], cache.problem_type, cache.bc!, y_, p, cache.mesh, u)
        Φ!(resids[2:end], cache, y_, u, p)
        recursive_flatten!(resid, resids)
        return resid
    end

    @unpack nlsolve, jac_alg = cache.alg

    resid = similar(y)

    resid_bc, resid_collocation = @view(resid[1:(cache.M)]), @view(resid[(cache.M + 1):end])

    sd_bc = jac_alg.bc_diffmode isa AbstractSparseADType ? SymbolicsSparsityDetection() :
            NoSparsityDetection()
    cache_bc = sparse_jacobian_cache(jac_alg.bc_diffmode, sd_bc, loss_bc!, resid_bc, y)

    N = length(cache.mesh)
    sd_collocation = if jac_alg.collocation_diffmode isa AbstractSparseADType
        Jₛ = construct_sparse_banded_jac_prototype(y, cache.M, N)
        JacPrototypeSparsityDetection(; jac_prototype = Jₛ)
    else
        NoSparsityDetection()
    end
    cache_collocation = sparse_jacobian_cache(jac_alg.collocation_diffmode, sd_collocation,
        loss_collocation!, resid_collocation, y)

    jac_prototype = vcat(__init_𝒥(cache_bc), __init_𝒥(cache_collocation))

    function jac!(J, x, p)
        # TODO: Pass `p` into `loss_bc!` and `loss_collocation!`
        sparse_jacobian!(@view(J[1:(cache.M), :]), jac_alg.bc_diffmode, cache_bc, loss_bc!,
            resid_bc, x)
        sparse_jacobian!(@view(J[(cache.M + 1):end, :]), jac_alg.collocation_diffmode,
            cache_collocation, loss_collocation!, resid_collocation, x)
        return J
    end

    # TODO: Enable sparse jacobian for RK tableau
    if typeof(cache.TU) == MIRKTableau
        return NonlinearProblem(NonlinearFunction{true}(loss!; jac = jac!, jac_prototype),
            y, cache.p) 
    else
        return NonlinearProblem(NonlinearFunction{true}(loss!),
        y, cache.p)
    end  
end

function construct_sparse_banded_jac_prototype(y, M, N)
    l = sum(i -> min(2M + i, M * N) - max(1, i - 1) + 1, 1:(M * (N - 1)))
    Is = Vector{Int}(undef, l)
    Js = Vector{Int}(undef, l)
    idx = 1
    for i in 1:(M * (N - 1)), j in max(1, i - 1):min(2M + i, M * N)
        Is[idx] = i
        Js[idx] = j
        idx += 1
    end
    y_ = similar(y, length(Is))
    return sparse(adapt(parameterless_type(y), Is), adapt(parameterless_type(y), Js),
        y_, M * (N - 1), M * N)
end
