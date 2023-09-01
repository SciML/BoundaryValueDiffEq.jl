import SparseDiffTools: __init_𝒥

function construct_nlproblem(cache::MIRKCache, y::AbstractVector)
    function loss_bc!(resid::AbstractVector, u::AbstractVector, p = cache.p)
        recursive_unflatten!(cache.y, u)
        eval_bc_residual!(cache.residual[1], cache.problem_type, cache.bc!, cache.y, p,
            cache.mesh, u)
        copyto!(resid, get_tmp(cache.residual[1], u))
        return resid
    end

    function loss_collocation!(resid::AbstractVector, u::AbstractVector, p = cache.p)
        recursive_unflatten!(cache.y, u)
        Φ!(cache, u, p)
        recursive_flatten!(resid, cache.residual; skip_first = true)
        return resid
    end

    function loss!(resid::AbstractVector, u::AbstractVector, p)
        recursive_unflatten!(cache.y, u)
        eval_bc_residual!(cache.residual[1], cache.problem_type, cache.bc!, cache.y, p,
            cache.mesh, u)
        Φ!(cache, u, p)
        recursive_flatten!(resid, cache.residual)
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
        Jₛ = sparse(BandedMatrix(similar(y, (cache.M * (N - 1), cache.M * N)),
            (1, 2 * cache.M)))
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

    return NonlinearProblem(NonlinearFunction{true}(loss!; jac = jac!, jac_prototype),
        y, cache.p)
end
