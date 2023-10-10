function construct_nlproblem(cache::RKCache{iip}, y::AbstractVector) where {iip}
    loss_bc = if iip
        function loss_bc_internal!(resid::AbstractVector, u::AbstractVector, p = cache.p)
            y_ = recursive_unflatten!(cache.y, u)
            eval_bc_residual!(resid, cache.problem_type, cache.bc, y_, p, cache.mesh)
            return resid
        end
    else
        function loss_bc_internal(u::AbstractVector, p = cache.p)
            y_ = recursive_unflatten!(cache.y, u)
            return eval_bc_residual(cache.problem_type, cache.bc, y_, p, cache.mesh)
        end
    end

    loss_collocation = if iip
        function loss_collocation_internal!(resid::AbstractVector, u::AbstractVector,
                                            p = cache.p)
            y_ = recursive_unflatten!(cache.y, u)
            resids = [get_tmp(r, u) for r in cache.residual[2:end]]
            Φ!(resids, cache, y_, u, p)
            recursive_flatten!(resid, resids)
            return resid
        end
    else
        function loss_collocation_internal(u::AbstractVector, p = cache.p)
            y_ = recursive_unflatten!(cache.y, u)
            resids = Φ(cache, y_, u, p)
            xxx = mapreduce(vec, vcat, resids)
            return xxx
        end
    end

    loss = if !(cache.problem_type isa TwoPointBVProblem)
        if iip
            function loss_internal!(resid::AbstractVector, u::AbstractVector, p = cache.p)
                y_ = recursive_unflatten!(cache.y, u)
                resids = [get_tmp(r, u) for r in cache.residual]
                eval_bc_residual!(resids[1], cache.problem_type, cache.bc, y_, p,
                                  cache.mesh)
                Φ!(@view(resids[2:end]), cache, y_, u, p)
                recursive_flatten!(resid, resids)
                return resid
            end
        else
            function loss_internal(u::AbstractVector, p = cache.p)
                y_ = recursive_unflatten!(cache.y, u)
                resid_bc = eval_bc_residual(cache.problem_type, cache.bc, y_, p, cache.mesh)
                resid_co = Φ(cache, y_, u, p)
                return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
            end
        end
    else
        # Reordering for 2 point BVP
        if iip
            function loss_internal_2point!(resid::AbstractVector, u::AbstractVector,
                                           p = cache.p)
                y_ = recursive_unflatten!(cache.y, u)
                resids = [get_tmp(r, u) for r in cache.residual]
                eval_bc_residual!(resids[1], cache.problem_type, cache.bc, y_, p,
                                  cache.mesh)
                Φ!(resids[2:end], cache, y_, u, p)
                recursive_flatten_twopoint!(resid, resids)
                return resid
            end
        else
            function loss_internal_2point(u::AbstractVector, p = cache.p)
                y_ = recursive_unflatten!(cache.y, u)
                resid_bc = eval_bc_residual(cache.problem_type, cache.bc, y_, p, cache.mesh)
                resid_co = Φ(cache, y_, u, p)
                return vcat(resid_bc.x[1], mapreduce(vec, vcat, resid_co), resid_bc.x[2])
            end
        end
    end

    return generate_nlprob(cache, y, loss_bc, loss_collocation, loss, cache.problem_type)
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
    col_colorvec = Vector{Int}(undef, M * N)
    for i in eachindex(col_colorvec)
        col_colorvec[i] = mod1(i, min(2M + 1, M * N) + 1)
    end
    row_colorvec = Vector{Int}(undef, M * (N - 1))
    for i in eachindex(row_colorvec)
        row_colorvec[i] = mod1(i, min(2M + 1, M * N) + 1)
    end

    y_ = similar(y, length(Is))
    return (sparse(adapt(parameterless_type(y), Is), adapt(parameterless_type(y), Js),
                   y_, M * (N - 1), M * N), col_colorvec, row_colorvec)
end

# Two Point Specialization
function construct_sparse_banded_jac_prototype(y::ArrayPartition, M, N)
    l = sum(i -> min(2M + i, M * N) - max(1, i - 1) + 1, 1:(M * (N - 1)))
    l_top = M * length(y.x[1].x[1])
    l_bot = M * length(y.x[1].x[2])

    Is = Vector{Int}(undef, l + l_top + l_bot)
    Js = Vector{Int}(undef, l + l_top + l_bot)
    idx = 1

    for i in 1:length(y.x[1].x[1]), j in 1:M
        Is[idx] = i
        Js[idx] = j
        idx += 1
    end

    for i in 1:(M * (N - 1)), j in max(1, i - 1):min(2M + i, M * N)
        Is[idx] = i + length(y.x[1].x[1])
        Js[idx] = j
        idx += 1
    end

    for i in 1:length(y.x[1].x[2]), j in 1:M
        Is[idx] = i + length(y.x[1].x[1]) + M * (N - 1)
        Js[idx] = j + M * (N - 1)
        idx += 1
    end

    col_colorvec = Vector{Int}(undef, M * N)
    for i in eachindex(col_colorvec)
        col_colorvec[i] = mod1(i, min(2M + 1, M * N) + 1)
    end
    row_colorvec = Vector{Int}(undef, M * N)
    for i in eachindex(row_colorvec)
        row_colorvec[i] = mod1(i, min(2M + 1, M * N) + 1)
    end

    y_ = similar(y, length(Is))
    return (sparse(adapt(parameterless_type(y), Is), adapt(parameterless_type(y), Js),
                   y_, M * N, M * N), col_colorvec, row_colorvec)
end

function generate_nlprob(cache::RKCache{iip}, y, loss_bc, loss_collocation, loss,
                         _) where {iip}
    @unpack nlsolve, jac_alg = cache.alg
    N = length(cache.mesh)

    stage = alg_stage(cache.alg)

    resid_bc = cache.prob.f.bcresid_prototype === nothing ? similar(y, cache.M) :
               cache.prob.f.bcresid_prototype
    expanded_jac = isa(cache.TU, RKTableau{false})
    resid_collocation = expanded_jac ? similar(y, cache.M * (N - 1) * (stage + 1)) :
                        similar(y, cache.M * (N - 1))

    sd_bc = jac_alg.bc_diffmode isa AbstractSparseADType ? SymbolicsSparsityDetection() :
            NoSparsityDetection()

    if iip
        cache_bc = sparse_jacobian_cache(jac_alg.bc_diffmode, sd_bc, loss_bc, resid_bc, y)
    else
        cache_bc = sparse_jacobian_cache(jac_alg.bc_diffmode, sd_bc, loss_bc, y;
                                         fx = resid_bc)
    end

    sd_collocation = if jac_alg.collocation_diffmode isa AbstractSparseADType
        Jₛ, cvec, rvec = construct_sparse_banded_jac_prototype(y, cache.M, N)
        PrecomputedJacobianColorvec(; jac_prototype = Jₛ, row_colorvec = rvec,
                                    col_colorvec = cvec)
    else
        NoSparsityDetection()
    end

    if iip
        cache_collocation = sparse_jacobian_cache(jac_alg.collocation_diffmode,
                                                  sd_collocation, loss_collocation,
                                                  resid_collocation, y)
    else
        cache_collocation = sparse_jacobian_cache(jac_alg.collocation_diffmode,
                                                  sd_collocation, loss_collocation, y;
                                                  fx = resid_collocation)
    end

    jac_prototype = vcat(init_jacobian(cache_bc),
                         jac_alg.collocation_diffmode isa AbstractSparseADType ? Jₛ :
                         init_jacobian(cache_collocation))

    # TODO: Pass `p` into `loss_bc` and `loss_collocation`. Currently leads to a Tag
    #       mismatch for ForwardDiff
    jac = if iip
        function jac_internal!(J, x, p)
            sparse_jacobian!(@view(J[1:(cache.M), :]), jac_alg.bc_diffmode, cache_bc,
                             loss_bc, resid_bc, x)
            sparse_jacobian!(@view(J[(cache.M + 1):end, :]), jac_alg.collocation_diffmode,
                             cache_collocation, loss_collocation, resid_collocation, x)
            return J
        end
    else
        J_ = jac_prototype
        function jac_internal(x, p)
            sparse_jacobian!(@view(J_[1:(cache.M), :]), jac_alg.bc_diffmode, cache_bc,
                             loss_bc, x)
            sparse_jacobian!(@view(J_[(cache.M + 1):end, :]), jac_alg.collocation_diffmode,
                             cache_collocation, loss_collocation, x)
            return J_
        end
    end

    return NonlinearProblem(NonlinearFunction{iip}(loss; jac, jac_prototype), y,
                            cache.p)
end

function generate_nlprob(cache::RKCache{iip}, y, loss_bc, loss_collocation, loss,
                         ::TwoPointBVProblem) where {iip}
    @unpack nlsolve, jac_alg = cache.alg
    N = length(cache.mesh)

    if !iip && cache.prob.f.bcresid_prototype === nothing
        y_ = recursive_unflatten!(cache.y, y)
        resid_ = cache.bc((y_[1], y_[end]), cache.p)
        resid = ArrayPartition(ArrayPartition(resid_),
                               similar(y, cache.M * (N - 1) * (stage + 1)))
    else
        resid = ArrayPartition(cache.prob.f.bcresid_prototype,
                               similar(y, cache.M * (N - 1) * (stage + 1)))
    end

    sd = if jac_alg.diffmode isa AbstractSparseADType
        Jₛ, cvec, rvec = construct_sparse_banded_jac_prototype(resid, cache.M, N)
        PrecomputedJacobianColorvec(; jac_prototype = Jₛ, row_colorvec = rvec,
                                    col_colorvec = cvec)
    else
        NoSparsityDetection()
    end

    if iip
        diffcache = sparse_jacobian_cache(jac_alg.diffmode, sd, loss, resid, y)
    else
        diffcache = sparse_jacobian_cache(jac_alg.diffmode, sd, loss, y; fx = resid)
    end

    jac_prototype = jac_alg.diffmode isa AbstractSparseADType ? Jₛ :
                    init_jacobian(diffcache)

    # TODO: Pass `p` into `loss_bc` and `loss_collocation`. Currently leads to a Tag
    #       mismatch for ForwardDiff
    jac = if iip
        function jac_internal!(J, x, p)
            sparse_jacobian!(J, jac_alg.diffmode, diffcache, loss, resid, x)
            return J
        end
    else
        J_ = jac_prototype
        function jac_internal(x, p)
            sparse_jacobian!(J_, jac_alg.diffmode, diffcache, loss, x)
            return J_
        end
    end

    return NonlinearProblem(NonlinearFunction{iip}(loss; jac, jac_prototype), y,
                            cache.p)
end
