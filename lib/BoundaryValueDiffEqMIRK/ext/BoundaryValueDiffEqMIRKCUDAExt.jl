module BoundaryValueDiffEqMIRKCUDAExt

using BoundaryValueDiffEqMIRK: BoundaryValueDiffEqMIRK, MIRKCache, DiffCacheNeeded,
    NoDiffCacheNeeded, recursive_unflatten!, get_tmp,
    eval_bc_residual!, recursive_flatten!,
    safe_similar, __restructure_sol, EvalSol, interval,
    interp_weights, __maybe_matmul!
using RecursiveArrayTools
using SciMLBase
using CUDA
using FastClosures
using ADTypes
import DifferentiationInterface as DI
using Adapt: adapt
using GPUArraysCore: AbstractGPUArray
using BandedMatrices: BandedMatrix, Ones
using SparseArrays: sparse
using CUDSS
using KernelAbstractions

const backend = CUDA.CUDABackend()

function BoundaryValueDiffEqMIRK.__construct_problem(cache::MIRKCache{iip}, y::CuArray{T, 1, D}, y₀::AbstractVectorOfArray) where {iip, T, D}
    constraint = (!isnothing(cache.prob.f.inequality)) ||
        (!isnothing(cache.prob.f.equality)) ||
        (!isnothing(cache.prob.lb)) ||
        (!isnothing(cache.prob.ub))
    return BoundaryValueDiffEqMIRK.__construct_problem(cache, y, y₀, Val(constraint))
end


function BoundaryValueDiffEqMIRK.__construct_problem(
        cache::MIRKCache{iip}, y::CuArray{T, 1, D},
        y₀::AbstractVectorOfArray, constraint
    ) where {iip, T, D}
    pt = cache.problem_type
    (; jac_alg) = cache.alg

    eval_sol = EvalSol(__restructure_sol(y₀.u, cache.in_size), cache.mesh, cache)

    trait = BoundaryValueDiffEqMIRK.__cache_trait(jac_alg)

    loss_bc = if iip
        @closure (
            du,
            u,
            p,
        ) -> BoundaryValueDiffEqMIRK.__mirk_loss_bc!(du, u, p, pt, cache.bc, cache.y, cache.mesh, cache, trait)
    else
        @closure (
            u, p,
        ) -> BoundaryValueDiffEqMIRK.__mirk_loss_bc(u, p, pt, cache.bc, cache.y, cache.mesh, cache, trait)
    end

    loss_collocation = if iip
        @closure (
            du,
            u,
            p,
        ) -> BoundaryValueDiffEqMIRK.__mirk_loss_collocation!(
            du, u, p, cache.y, cache.mesh, cache.residual, cache, trait, constraint
        )
    else
        @closure (
            u,
            p,
        ) -> BoundaryValueDiffEqMIRK.__mirk_loss_collocation(
            u, p, cache.y, cache.mesh, cache.residual, cache, trait
        )
    end

    loss = if iip
        @closure (
            du,
            u,
            p,
        ) -> BoundaryValueDiffEqMIRK.__mirk_loss!(
            du, u, p, cache.y, pt, cache.bc, cache.residual,
            cache.mesh, cache, eval_sol, trait, constraint
        )
    else
        @closure (
            u,
            p,
        ) -> BoundaryValueDiffEqMIRK.__mirk_loss(
            u, p, cache.y, pt, cache.bc, cache.mesh, cache, eval_sol, trait
        )
    end

    if !isnothing(cache.alg.optimize)
        loss = @closure (
            du,
            u,
            p,
        ) -> BoundaryValueDiffEqMIRK.__mirk_loss!(
            du, u, p, cache.y, pt, cache.bc, cache.residual,
            cache.bcresid_prototype, cache.mesh, cache, eval_sol, trait, constraint
        )
    end

    return BoundaryValueDiffEqMIRK.__construct_problem(cache, y, loss_bc, loss_collocation, loss, pt, constraint)
end

function BoundaryValueDiffEqMIRK.__construct_problem(
        cache::MIRKCache{iip, T, UB, DC, tune_parameters}, y::CuArray{T, 1, D}, loss_bc::BC, loss_collocation::C, loss::LF,
        ::SciMLBase.StandardBVProblem, constraint::Val{false}
    ) where {iip, T, UB, DC, tune_parameters, BC, C, LF, D}
    (; jac_alg) = cache.alg
    (; f_prototype, bcresid_prototype, prob) = cache
    (; bc_diffmode) = jac_alg
    N = length(cache.mesh)

    resid_bc = bcresid_prototype
    L = length(resid_bc)
    L_f_prototype = length(f_prototype)
    resid_collocation = safe_similar(y, L_f_prototype * (N - 1))

    # Prepare Jacobian caches for BC on CPU only
    y_cpu = Array(y)
    resid_bc_cpu = Array(resid_bc)
    cache_bc = if iip
        DI.prepare_jacobian(
            loss_bc, resid_bc, bc_diffmode, y, DI.Constant(cache.p); strict = Val(false)
        )
    else
        DI.prepare_jacobian(
            loss_bc, bc_diffmode, y, DI.Constant(cache.p); strict = Val(false)
        )
    end

    nonbc_diffmode = AutoSparse(
        get_dense_ad(jac_alg.nonbc_diffmode),
        sparsity_detector = __default_sparsity_detector(jac_alg.nonbc_diffmode),
        coloring_algorithm = __default_coloring_algorithm(jac_alg.nonbc_diffmode)
    )
    cache_collocation = if iip
        DI.prepare_jacobian(
            loss_collocation, resid_collocation, nonbc_diffmode, y, DI.Constant(cache.p);
            strict = Val(false)
        )
    else
        DI.prepare_jacobian(
            loss_collocation, nonbc_diffmode, y, DI.Constant(cache.p); strict = Val(false)
        )
    end

    J_bc = if iip
        DI.jacobian(loss_bc, resid_bc_cpu, cache_bc, bc_diffmode, y, DI.Constant(cache.p))
    else
        DI.jacobian(loss_bc, cache_bc, bc_diffmode, y_cpu, DI.Constant(cache.p))
    end
    J_bc = adapt(backend, J_bc)
    J_c = if iip
        DI.jacobian(
            loss_collocation, resid_collocation, cache_collocation,
            nonbc_diffmode, y, DI.Constant(cache.p)
        )
    else
        DI.jacobian(
            loss_collocation, cache_collocation, nonbc_diffmode, y, DI.Constant(cache.p)
        )
    end
    jac_prototype = vcat(J_bc, J_c)

    jac = if iip
        @closure (
            J,
            u,
            p,
        ) -> __mirk_mpoint_jacobian!(
            J, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc, cache_collocation,
            loss_bc, loss_collocation, resid_bc, resid_collocation, L, cache.p
        )
    else
        @closure (
            u,
            p,
        ) -> __mirk_mpoint_jacobian(
            jac_prototype, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc,
            cache_collocation, loss_bc, loss_collocation, L, cache.p
        )
    end

    cost_fun = __build_cost(
        prob.f.cost, cache, cache.mesh, cache.M;
        tune_parameters, p = cache.p
    )

    resid_prototype = vcat(resid_bc, resid_collocation)
    return __construct_internal_problem(
        prob, cache.problem_type, cache.alg, loss, jac, jac_prototype, resid_prototype,
        bcresid_prototype, f_prototype, y, cache.p, cache.M, N, cost_fun
    )
end

function BoundaryValueDiffEqMIRK.__construct_problem(
        cache::MIRKCache{iip, T, UB, DC, tune_parameters}, y::CuArray{T, 1, D}, loss_bc::BC, loss_collocation::C, loss::LF,
        ::SciMLBase.TwoPointBVProblem, constraint::Val{false}
    ) where {iip, T, UB, DC, tune_parameters, BC, C, LF, D}
    (; jac_alg) = cache.alg
    (; f_prototype, bcresid_prototype, prob) = cache
    N = length(cache.mesh)

    len_a = prod(cache.resid_size[1])
    len_b = length(cache.bcresid_prototype) - len_a
    resid_a = safe_similar(y, len_a)
    copyto!(resid_a, 1, cache.bcresid_prototype, 1, len_a)
    resid_collocation = safe_similar(y, cache.M * (N - 1))
    resid_b = safe_similar(y, len_b)
    copyto!(resid_b, 1, cache.bcresid_prototype, len_a + 1, len_b)
    resid = _concat_like(y, resid_a, resid_collocation, resid_b)

    diffmode = if jac_alg.diffmode isa AutoSparse
        sparse_jacobian_prototype = BoundaryValueDiffEqMIRK.__generate_sparse_jacobian_prototype(
            cache, cache.problem_type,
            @view(bcresid_prototype[1:prod(cache.resid_size[1])]),
            @view(bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]), cache.M, N, nothing
        )

        AutoSparse(
            BoundaryValueDiffEqMIRK.get_dense_ad(jac_alg.diffmode);
            sparsity_detector = ADTypes.KnownJacobianSparsityDetector(sparse_jacobian_prototype),
            coloring_algorithm = BoundaryValueDiffEqMIRK.__default_coloring_algorithm(jac_alg.diffmode)
        )
    else
        jac_alg.diffmode
    end

    diffcache = if iip
        DI.prepare_jacobian(
            loss, resid, diffmode, y, DI.Constant(cache.p); strict = Val(false)
        )
    else
        DI.prepare_jacobian(
            loss, diffmode, y, DI.Constant(cache.p); strict = Val(false)
        )
    end

    jac_prototype = if iip
        DI.jacobian(loss, resid, diffcache, diffmode, y, DI.Constant(cache.p))
    else
        DI.jacobian(loss, diffcache, diffmode, y, DI.Constant(cache.p))
    end

    jac = if iip
        @closure (
            J, u, p,
        ) -> BoundaryValueDiffEqMIRK.__mirk_2point_jacobian!(J, u, diffmode, diffcache, loss, resid, p)
    else
        @closure (
            u, p,
        ) -> BoundaryValueDiffEqMIRK.__mirk_2point_jacobian(u, jac_prototype, diffmode, diffcache, loss, p)
    end

    cost_fun = BoundaryValueDiffEqMIRK.__build_cost(
        prob.f.cost, cache, cache.mesh, cache.M;
        tune_parameters, p = cache.p
    )

    resid_prototype = copy(resid)
    return BoundaryValueDiffEqMIRK.__construct_internal_problem(
        cache.prob, cache.problem_type, cache.alg, loss, jac, jac_prototype,
        resid_prototype, bcresid_prototype, f_prototype, y, cache.p, cache.M, N, cost_fun
    )
end

@views function BoundaryValueDiffEqMIRK.__mirk_loss!(
        resid, u, p, y, pt::SciMLBase.StandardBVProblem, bc!::BC, residual, mesh,
        cache::MIRKCache, _, trait::DiffCacheNeeded, constraint,
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    __gpu_collocation!(resids[2:end], cache, y_, u)
    soly_ = __boundary_condition_input(pt, cache, y_, u, mesh)
    eval_bc_residual!(resids[1], pt, bc!, soly_, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function BoundaryValueDiffEqMIRK.__mirk_loss!(
        resid, u, p, y, pt::SciMLBase.StandardBVProblem, bc!::BC, residual, mesh,
        cache::MIRKCache, _, trait::NoDiffCacheNeeded, constraint,
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    __gpu_collocation!(residual[2:end], cache, y_, u)
    soly_ = __boundary_condition_input(pt, cache, y_, u, mesh)
    eval_bc_residual!(residual[1], pt, bc!, soly_, p, mesh)
    recursive_flatten!(resid, residual)
    return nothing
end


@views function BoundaryValueDiffEqMIRK.__mirk_loss!(
        resid, u, p, y, pt::SciMLBase.TwoPointBVProblem,
        bc!::Tuple{BC1, BC2}, residual, mesh, cache::MIRKCache, _,
        trait::DiffCacheNeeded, constraint
    ) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    __gpu_collocation!(resids[2:end], cache, y_, u)
    len_a = prod(cache.resid_size[1])
    len_b = prod(cache.resid_size[2])
    bc_tmp_host = Array(resids[1])

    # Two-point BC callbacks run on CPU; evaluate them on host arrays and copy back.
    ua_host = Array(first(y_))
    ub_host = Array(last(y_))
    resida_host = copy(@view bc_tmp_host[1:len_a])
    residb_host = copy(@view bc_tmp_host[(len_a + 1):end])
    first(bc!)(resida_host, ua_host, p)
    last(bc!)(residb_host, ub_host, p)
    copyto!(resids[1], 1, resida_host, 1, len_a)
    copyto!(resids[1], len_a + 1, residb_host, 1, len_b)

    # Flatten on host first to avoid ReinterpretArray -> MtlVector scalar indexing.
    resid_host = Vector{eltype(resid)}(undef, length(resid))
    copyto!(resid_host, 1, resida_host, 1, len_a)
    idx = len_a + 1
    for r in resids[2:end]
        ri_host = Array(r)
        n = length(ri_host)
        copyto!(resid_host, idx, ri_host, 1, n)
        idx += n
    end
    copyto!(resid_host, idx, residb_host, 1, len_b)
    copyto!(resid, 1, resid_host, 1, length(resid_host))
    return nothing
end

@views function BoundaryValueDiffEqMIRK.__mirk_loss!(
        resid, u, p, y, pt::SciMLBase.TwoPointBVProblem,
        bc!::Tuple{BC1, BC2}, residual, mesh, cache::MIRKCache, _,
        trait::NoDiffCacheNeeded, constraint
    ) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    __gpu_collocation!(residual[2:end], cache, y_, u)
    len_a = prod(cache.resid_size[1])
    len_b = prod(cache.resid_size[2])
    bc_tmp_host = Array(residual[1])

    ua_host = Array(first(y_))
    ub_host = Array(last(y_))
    resida_host = copy(@view bc_tmp_host[1:len_a])
    residb_host = copy(@view bc_tmp_host[(len_a + 1):end])
    first(bc!)(resida_host, ua_host, p)
    last(bc!)(residb_host, ub_host, p)

    resid_host = Vector{eltype(resid)}(undef, length(resid))
    copyto!(resid_host, 1, resida_host, 1, len_a)
    idx = len_a + 1
    for r in residual[2:end]
        ri_host = Array(r)
        n = length(ri_host)
        copyto!(resid_host, idx, ri_host, 1, n)
        idx += n
    end
    copyto!(resid_host, idx, residb_host, 1, len_b)
    copyto!(resid, 1, resid_host, 1, length(resid_host))
    return nothing
end

@views function BoundaryValueDiffEqMIRK.__mirk_loss!(
        resid, u, p, y, pt::SciMLBase.TwoPointBVProblem,
        bc!::Tuple{BC1, BC2}, residual, bcresid_prototype, mesh, cache::MIRKCache, _,
        trait, constraint
    ) where {BC1, BC2}
    BoundaryValueDiffEqMIRK.__mirk_loss!(
        resid, u, p, y, pt, bc!, residual, mesh, cache, nothing, trait, constraint
    )
    return nothing
end

@inline function _tuple_from(vec, n)
    return ntuple(i -> vec[i], n)
end

@inline function _matrix_tuple(mat, n)
    return ntuple(i -> ntuple(j -> mat[i, j], n), n)
end

function _concat_like(x::AbstractVector, parts::AbstractVector...)
    total = mapreduce(length, +, parts; init = 0)
    out = safe_similar(x, total)
    idx = 1
    for part in parts
        n = length(part)
        part_ = part
        if typeof(part_) !== typeof(out)
            # Ensure slices/views are materialized on the same device as `out`
            # before concatenation to avoid host-side scalar iteration.
            part_ = safe_similar(out, n)
            copyto!(part_, 1, collect(part), 1, n)
        end
        copyto!(out, idx, part_, 1, n)
        idx += n
    end
    return out
end

function _pack_states(y_, M::Int)
    npts = length(y_)
    T = eltype(first(y_))
    data = Array{T}(undef, M, npts)
    for j in 1:npts
        yj_host = Array(y_[j])
        copyto!(vec(data), (j - 1) * M + 1, yj_host, 1, M)
    end
    return data
end

function _pack_stages(cache::MIRKCache, u, M::Int, stage::Int, intervals::Int)
    T = eltype(get_tmp(cache.k_discrete[1], u))
    data = Array{T}(undef, M, stage, intervals)
    for i in 1:intervals
        Ki = get_tmp(cache.k_discrete[i], u)
        Ki_host = Array(Ki)
        copyto!(vec(data), (i - 1) * M * stage + 1, Ki_host, 1, M * stage)
    end
    return data
end

@kernel function gpu_collocation_kernel!(
        residual, k_discrete, tmp_stage, y, mesh, mesh_dt,
        c_vals, v_vals, x_vals, b_vals, stage::Int, f!, p
    )
    i = @index(Global, Linear)

    M = size(residual, 1)
    h = mesh_dt[i]
    tL = mesh[i]
    tmp = @view tmp_stage[:, i]
    one_T = one(eltype(y))
    zero_T = zero(eltype(residual))
    @inbounds for r in 1:stage
        vr = v_vals[r]
        cr = c_vals[r]
        for m in 1:M
            val = (one_T - vr) * y[m, i] + vr * y[m, i + 1]
            for s in 1:(r - 1)
                val += x_vals[r][s] * k_discrete[m, s, i] * h
            end
            tmp[m] = val
        end
        k_col = @view k_discrete[:, r, i]
        f!(k_col, tmp, p, tL + cr * h)
    end

    resid = @view residual[:, i]
    @inbounds for m in 1:M
        acc = zero_T
        for r in 1:stage
            acc += k_discrete[m, r, i] * b_vals[r]
        end
        resid[m] = (y[m, i + 1] - y[m, i]) - h * acc
    end
end

function __gpu_collocation!(resids_slice, cache::MIRKCache, y_, u)
    intervals = length(cache.mesh) - 1
    intervals <= 0 && return
    M = cache.M
    stage = cache.stage
    y_host = _pack_states(y_, M)
    k_host = _pack_stages(cache, u, M, stage, intervals)
    T = eltype(y_host)

    y_dev = adapt(backend, y_host)
    k_dev = adapt(backend, k_host)
    tmp_dev = adapt(backend, zeros(T, M, intervals))
    resid_dev = adapt(backend, zeros(T, M, intervals))
    mesh_dev = adapt(backend, cache.mesh)
    meshdt_dev = adapt(backend, cache.mesh_dt)
    c_tuple = _tuple_from(cache.TU.c, stage)
    v_tuple = _tuple_from(cache.TU.v, stage)
    b_tuple = _tuple_from(cache.TU.b, stage)
    x_tuple = _matrix_tuple(cache.TU.x, stage)

    kernel! = gpu_collocation_kernel!(backend, intervals)
    kernel!(
        resid_dev, k_dev, tmp_dev, y_dev, mesh_dev, meshdt_dev,
        c_tuple, v_tuple, x_tuple, b_tuple, stage, cache.f.f, cache.p;
        ndrange = (intervals,)
    )
    KernelAbstractions.synchronize(backend)

    resid_host = Array(resid_dev)
    k_host = Array(k_dev)
    for i in 1:intervals
        resid_col = copy(@view resid_host[:, i])
        copyto!(resids_slice[i], 1, resid_col, 1, M)
        Ki = get_tmp(cache.k_discrete[i], u)
        k_block = vec(copy(@view k_host[:, :, i]))
        copyto!(Ki, 1, k_block, 1, M * stage)
    end
    return
end


struct HostEvalSol{T, U, A, K}
    t::T
    u::U
    alg::A
    stage::Int
    k_discrete::K
end

function (s::HostEvalSol)(tval::Number)
    (; t, u, alg, stage, k_discrete) = s
    (tval == t[1]) && return first(u)
    (tval == t[end]) && return last(u)
    z = zero(last(u))
    ii = interval(t, tval)
    dt = t[ii + 1] - t[ii]
    τ = (tval - t[ii]) / dt
    w, _ = interp_weights(τ, alg)
    __maybe_matmul!(z, @view(k_discrete[ii][:, 1:stage]), w[1:stage])
    z .= z .* dt .+ u[ii]
    return z
end

function _boundary_values_on_host(cache::MIRKCache, y_, u)
    y_host = Array.(y_)
    k_host = [Array(get_tmp(cache.k_discrete[i], u)) for i in 1:(length(cache.mesh) - 1)]
    return y_host, k_host
end

function __boundary_condition_input(
        pt::SciMLBase.StandardBVProblem, cache::MIRKCache, y_, u,
        mesh
    )
    y_host, k_host = _boundary_values_on_host(cache, y_, u)
    return HostEvalSol(
        mesh, __restructure_sol(y_host, cache.in_size), cache.alg,
        cache.stage, k_host
    )
end

function __boundary_condition_input(
        ::SciMLBase.TwoPointBVProblem, cache::MIRKCache, y_, u,
        mesh
    )
    y_host, _ = _boundary_values_on_host(cache, y_, u)
    return VectorOfArray(y_host)
end

"""
    __generate_sparse_jacobian_prototype(
        cache::MIRKCache, problem::SciMLBase.StandardBVProblem, ya, yb, M, N)

Generate a sparse Jacobian prototype on GPU
"""
function BoundaryValueDiffEqMIRK.__generate_sparse_jacobian_prototype(
        ::MIRKCache, ::SciMLBase.StandardBVProblem, ya, yb, M, N, _
    )
    #fast_scalar_indexing(ya) ||
    #    error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    # Wait on the SMC structured colorings
    J_c = sparse(CuArray(BandedMatrix(Ones{eltype(ya)}(M * (N - 1), M * N), (1, 2M - 1))); fmt = :csr)
    return J_c
end

function BoundaryValueDiffEqMIRK.__generate_sparse_jacobian_prototype(
        ::MIRKCache, ::SciMLBase.TwoPointBVProblem, ya, yb, M, N, _
    )
    #fast_scalar_indexing(ya) ||
    #    error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J₁ = length(ya) + length(yb) + M * (N - 1)
    J₂ = M * N
    # Wait on the SMC structured colorings
    J = sparse(CuArray(BandedMatrix(Ones{eltype(ya)}(J₁, J₂), (M + 1, M + 1))); fmt = :csr)
    # for underdetermined systems we don't have banded qr implemented. use sparse
    J₁ < J₂ && return sparse(J)
    return J
end

end
