@concrete struct MIRKCache{iip, T}
    order::Int                 # The order of MIRK method
    stage::Int                 # The state of MIRK method
    M::Int                     # The number of equations
    in_size
    f
    bc
    prob                       # BVProblem
    problem_type               # StandardBVProblem
    p                          # Parameters
    alg                        # MIRK methods
    TU                         # MIRK Tableau
    ITU                        # MIRK Interpolation Tableau
    bcresid_prototype
    # Everything below gets resized in adaptive methods
    mesh                       # Discrete mesh
    mesh_dt                    # Step size
    k_discrete                 # Stage information associated with the discrete Runge-Kutta method
    k_interp                   # Stage information associated with the discrete Runge-Kutta method
    y
    y₀
    residual
    # The following 2 caches are never resized
    fᵢ_cache
    fᵢ₂_cache
    defect
    new_stages
    resid_size
    kwargs
end

Base.eltype(::MIRKCache{iip, T}) where {iip, T} = T

function SciMLBase.__init(prob::BVProblem, alg::AbstractMIRK; dt = 0.0,
        abstol = 1e-3, adaptive = true, kwargs...)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    iip = isinplace(prob)

    _, T, M, n, X = __extract_problem_details(prob; dt, check_positive_dt = true)
    # NOTE: Assumes the user provided initial guess is on a uniform mesh
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
    mesh_dt = diff(mesh)

    chunksize = pickchunksize(M * (n + 1))

    __alloc = x -> __maybe_allocate_diffcache(vec(x), chunksize, alg.jac_alg)

    fᵢ_cache = __alloc(similar(X))
    fᵢ₂_cache = vec(similar(X))

    defect_threshold = T(0.1)  # TODO: Allow user to specify these
    MxNsub = 3000              # TODO: Allow user to specify these

    # Don't flatten this here, since we need to expand it later if needed
    y₀ = __initial_state_from_prob(prob, mesh)
    y = __alloc.(copy.(y₀))
    TU, ITU = constructMIRK(alg, T)
    stage = alg_stage(alg)

    k_discrete = [__maybe_allocate_diffcache(similar(X, M, stage), chunksize, alg.jac_alg)
                  for _ in 1:n]
    k_interp = [similar(X, ifelse(adaptive, M, 0), ifelse(adaptive, ITU.s_star - stage, 0))
                for _ in 1:n]

    bcresid_prototype, resid₁_size = __get_bcresid_prototype(prob.problem_type, prob, X)

    residual = if iip
        if prob.problem_type isa TwoPointBVProblem
            vcat([__alloc(__vec(bcresid_prototype))], __alloc.(copy.(@view(y₀[2:end]))))
        else
            vcat([__alloc(bcresid_prototype)], __alloc.(copy.(@view(y₀[2:end]))))
        end
    else
        nothing
    end

    defect = [similar(X, ifelse(adaptive, M, 0)) for _ in 1:n]
    new_stages = [similar(X, ifelse(adaptive, M, 0)) for _ in 1:n]

    # Transform the functions to handle non-vector inputs
    bcresid_prototype = __vec(bcresid_prototype)
    f, bc = if X isa AbstractVector
        prob.f, prob.f.bc
    elseif iip
        vecf! = (du, u, p, t) -> __vec_f!(du, u, p, t, prob.f, size(X))
        vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
            (r, u, p, t) -> __vec_bc!(r, u, p, t, prob.f.bc, resid₁_size, size(X))
        else
            ((r, u, p) -> __vec_bc!(r, u, p, prob.f.bc[1], resid₁_size[1], size(X)),
                (r, u, p) -> __vec_bc!(r, u, p, prob.f.bc[2], resid₁_size[2], size(X)))
        end
        vecf!, vecbc!
    else
        vecf = (u, p, t) -> __vec_f(u, p, t, prob.f, size(X))
        vecbc = if !(prob.problem_type isa TwoPointBVProblem)
            (u, p, t) -> __vec_bc(u, p, t, prob.f.bc, size(X))
        else
            ((u, p) -> __vec_bc(u, p, prob.f.bc[1], size(X))),
            (u, p) -> __vec_bc(u, p, prob.f.bc[2], size(X))
        end
        vecf, vecbc
    end

    prob_ = !(prob.u0 isa AbstractArray) ? remake(prob; u0 = X) : prob

    return MIRKCache{iip, T}(alg_order(alg), stage, M, size(X), f, bc, prob_,
        prob.problem_type, prob.p, alg, TU, ITU, bcresid_prototype, mesh, mesh_dt,
        k_discrete, k_interp, y, y₀, residual, fᵢ_cache, fᵢ₂_cache, defect, new_stages,
        resid₁_size, (; defect_threshold, MxNsub, abstol, dt, adaptive, kwargs...))
end

"""
    __expand_cache!(cache::MIRKCache)

After redistributing or halving the mesh, this function expands the required vectors to
match the length of the new mesh.
"""
function __expand_cache!(cache::MIRKCache)
    Nₙ = length(cache.mesh)
    __append_similar!(cache.k_discrete, Nₙ - 1, cache.M)
    __append_similar!(cache.k_interp, Nₙ - 1, cache.M)
    __append_similar!(cache.y, Nₙ, cache.M)
    __append_similar!(cache.y₀, Nₙ, cache.M)
    __append_similar!(cache.residual, Nₙ, cache.M)
    __append_similar!(cache.defect, Nₙ - 1, cache.M)
    __append_similar!(cache.new_stages, Nₙ - 1, cache.M)
    return cache
end

function __split_mirk_kwargs(; defect_threshold, MxNsub, abstol, dt, adaptive = true,
        kwargs...)
    return ((defect_threshold, MxNsub, abstol, adaptive, dt),
        (; abstol, adaptive, kwargs...))
end

function SciMLBase.solve!(cache::MIRKCache)
    (defect_threshold, MxNsub, abstol, adaptive, _), kwargs = __split_mirk_kwargs(;
        cache.kwargs...)
    @unpack y, y₀, prob, alg, mesh, mesh_dt, TU, ITU = cache
    info::ReturnCode.T = ReturnCode.Success
    defect_norm = 2 * abstol

    while SciMLBase.successful_retcode(info) && defect_norm > abstol
        nlprob = __construct_nlproblem(cache, recursive_flatten(y₀))
        sol_nlprob = __solve(nlprob, alg.nlsolve; abstol, kwargs...)
        recursive_unflatten!(cache.y₀, sol_nlprob.u)

        info = sol_nlprob.retcode

        !adaptive && break

        if info == ReturnCode.Success
            defect_norm = defect_estimate!(cache)
            # The defect is greater than 10%, the solution is not acceptable
            defect_norm > defect_threshold && (info = ReturnCode.Failure)
        end

        if info == ReturnCode.Success
            if defect_norm > abstol
                # We construct a new mesh to equidistribute the defect
                mesh, mesh_dt, _, info = mesh_selector!(cache)
                if info == ReturnCode.Success
                    __append_similar!(cache.y₀, length(cache.mesh), cache.M)
                    for (i, m) in enumerate(cache.mesh)
                        interp_eval!(cache.y₀[i], cache, m, mesh, mesh_dt)
                    end
                    __expand_cache!(cache)
                end
            end
        else
            #  We cannot obtain a solution for the current mesh
            if 2 * (length(cache.mesh) - 1) > MxNsub
                # New mesh would be too large
                info = ReturnCode.Failure
            else
                half_mesh!(cache)
                __expand_cache!(cache)
                recursive_fill!(cache.y₀, 0)
                info = ReturnCode.Success # Force a restart
                defect_norm = 2 * abstol
            end
        end
    end

    u = [reshape(y, cache.in_size) for y in cache.y₀]
    return DiffEqBase.build_solution(prob, alg, cache.mesh,
        u; interp = MIRKInterpolation(cache.mesh, u, cache), retcode = info)
end

# Constructing the Nonlinear Problem
function __construct_nlproblem(cache::MIRKCache{iip}, y::AbstractVector) where {iip}
    pt = cache.problem_type

    loss_bc = if iip
        (du, u, p) -> __mirk_loss_bc!(du, u, p, pt, cache.bc, cache.y, cache.mesh)
    else
        (u, p) -> __mirk_loss_bc(u, p, pt, cache.bc, cache.y, cache.mesh)
    end

    loss_collocation = if iip
        (du, u, p) -> __mirk_loss_collocation!(du, u, p, cache.y, cache.mesh,
            cache.residual, cache)
    else
        (u, p) -> __mirk_loss_collocation(u, p, cache.y, cache.mesh, cache.residual, cache)
    end

    loss = if iip
        (du, u, p) -> __mirk_loss!(du, u, p, cache.y, pt, cache.bc, cache.residual,
            cache.mesh, cache)
    else
        (u, p) -> __mirk_loss(u, p, cache.y, pt, cache.bc, cache.mesh, cache)
    end

    return __construct_nlproblem(cache, y, loss_bc, loss_collocation, loss, pt)
end

function __mirk_loss!(resid, u, p, y, pt::StandardBVProblem, bc!::BC, residual, mesh,
        cache) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    eval_bc_residual!(resids[1], pt, bc!, y_, p, mesh)
    Φ!(resids[2:end], cache, y_, u, p)
    recursive_flatten!(resid, resids)
    return nothing
end

function __mirk_loss!(resid, u, p, y, pt::TwoPointBVProblem, bc!::Tuple{BC1, BC2}, residual,
        mesh, cache) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    resida = @view resids[1][1:prod(cache.resid_size[1])]
    residb = @view resids[1][(prod(cache.resid_size[1]) + 1):end]
    eval_bc_residual!((resida, residb), pt, bc!, y_, p, mesh)
    Φ!(resids[2:end], cache, y_, u, p)
    recursive_flatten_twopoint!(resid, resids, cache.resid_size)
    return nothing
end

function __mirk_loss(u, p, y, pt::StandardBVProblem, bc::BC, mesh, cache) where {BC}
    y_ = recursive_unflatten!(y, u)
    resid_bc = eval_bc_residual(pt, bc, y_, p, mesh)
    resid_co = Φ(cache, y_, u, p)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end

function __mirk_loss(u, p, y, pt::TwoPointBVProblem, bc::Tuple{BC1, BC2}, mesh,
        cache) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    resid_bca, resid_bcb = eval_bc_residual(pt, bc, y_, p, mesh)
    resid_co = Φ(cache, y_, u, p)
    return vcat(resid_bca, mapreduce(vec, vcat, resid_co), resid_bcb)
end

function __mirk_loss_bc!(resid, u, p, pt, bc!::BC, y, mesh) where {BC}
    y_ = recursive_unflatten!(y, u)
    eval_bc_residual!(resid, pt, bc!, y_, p, mesh)
    return nothing
end

function __mirk_loss_bc(u, p, pt, bc!::BC, y, mesh) where {BC}
    y_ = recursive_unflatten!(y, u)
    return eval_bc_residual(pt, bc!, y_, p, mesh)
end

function __mirk_loss_collocation!(resid, u, p, y, mesh, residual, cache)
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual[2:end]]
    Φ!(resids, cache, y_, u, p)
    recursive_flatten!(resid, resids)
    return nothing
end

function __mirk_loss_collocation(u, p, y, mesh, residual, cache)
    y_ = recursive_unflatten!(y, u)
    resids = Φ(cache, y_, u, p)
    return mapreduce(vec, vcat, resids)
end

function __construct_nlproblem(cache::MIRKCache{iip}, y, loss_bc::BC, loss_collocation::C,
        loss::LF, ::StandardBVProblem) where {iip, BC, C, LF}
    @unpack nlsolve, jac_alg = cache.alg
    N = length(cache.mesh)

    resid_bc = cache.bcresid_prototype
    L = length(resid_bc)
    resid_collocation = similar(y, cache.M * (N - 1))

    loss_bcₚ = iip ? ((du, u) -> loss_bc(du, u, cache.p)) : (u -> loss_bc(u, cache.p))
    loss_collocationₚ = iip ? ((du, u) -> loss_collocation(du, u, cache.p)) :
                        (u -> loss_collocation(u, cache.p))

    sd_bc = jac_alg.bc_diffmode isa AbstractSparseADType ? SymbolicsSparsityDetection() :
            NoSparsityDetection()
    cache_bc = __sparse_jacobian_cache(Val(iip), jac_alg.bc_diffmode, sd_bc, loss_bcₚ,
        resid_bc, y)

    sd_collocation = if jac_alg.nonbc_diffmode isa AbstractSparseADType
        __sparsity_detection_alg(__generate_sparse_jacobian_prototype(cache,
            cache.problem_type, y, y, cache.M, N))
    else
        NoSparsityDetection()
    end
    cache_collocation = __sparse_jacobian_cache(Val(iip), jac_alg.nonbc_diffmode,
        sd_collocation, loss_collocationₚ, resid_collocation, y)

    jac_prototype = vcat(init_jacobian(cache_bc), init_jacobian(cache_collocation))

    jac = if iip
        (J, u, p) -> __mirk_mpoint_jacobian!(J, u, jac_alg.bc_diffmode,
            jac_alg.nonbc_diffmode, cache_bc, cache_collocation, loss_bcₚ,
            loss_collocationₚ, resid_bc, resid_collocation, L)
    else
        (u, p) -> __mirk_mpoint_jacobian(jac_prototype, u, jac_alg.bc_diffmode,
            jac_alg.nonbc_diffmode, cache_bc, cache_collocation, loss_bcₚ,
            loss_collocationₚ, L)
    end

    nlf = NonlinearFunction{iip}(loss; resid_prototype = vcat(resid_bc, resid_collocation),
        jac, jac_prototype)
    return (L == cache.M ? NonlinearProblem : NonlinearLeastSquaresProblem)(nlf, y, cache.p)
end

function __mirk_mpoint_jacobian!(J, x, bc_diffmode, nonbc_diffmode, bc_diffcache,
        nonbc_diffcache, loss_bc::BC, loss_collocation::C, resid_bc, resid_collocation,
        L::Int) where {BC, C}
    sparse_jacobian!(@view(J[1:L, :]), bc_diffmode, bc_diffcache, loss_bc, resid_bc, x)
    sparse_jacobian!(@view(J[(L + 1):end, :]), nonbc_diffmode, nonbc_diffcache,
        loss_collocation, resid_collocation, x)
    return nothing
end

function __mirk_mpoint_jacobian(J, x, bc_diffmode, nonbc_diffmode, bc_diffcache,
        nonbc_diffcache, loss_bc::BC, loss_collocation::C, L::Int) where {BC, C}
    sparse_jacobian!(@view(J[1:L, :]), bc_diffmode, bc_diffcache, loss_bc, x)
    sparse_jacobian!(@view(J[(L + 1):end, :]), nonbc_diffmode, nonbc_diffcache,
        loss_collocation, x)
    return J
end

function __construct_nlproblem(cache::MIRKCache{iip}, y, loss_bc::BC, loss_collocation::C,
        loss::LF, ::TwoPointBVProblem) where {iip, BC, C, LF}
    @unpack nlsolve, jac_alg = cache.alg
    N = length(cache.mesh)

    lossₚ = iip ? ((du, u) -> loss(du, u, cache.p)) : (u -> loss(u, cache.p))

    resid = vcat(@view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
        similar(y, cache.M * (N - 1)),
        @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]))
    L = length(cache.bcresid_prototype)

    sd = if jac_alg.diffmode isa AbstractSparseADType
        __sparsity_detection_alg(__generate_sparse_jacobian_prototype(cache,
            cache.problem_type, @view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
            @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]), cache.M,
            N))
    else
        NoSparsityDetection()
    end
    diffcache = __sparse_jacobian_cache(Val(iip), jac_alg.diffmode, sd, lossₚ, resid, y)
    jac_prototype = init_jacobian(diffcache)

    jac = if iip
        (J, u, p) -> __mirk_2point_jacobian!(J, u, jac_alg.diffmode, diffcache, lossₚ,
            resid)
    else
        (u, p) -> __mirk_2point_jacobian(u, jac_prototype, jac_alg.diffmode, diffcache,
            lossₚ)
    end

    nlf = NonlinearFunction{iip}(loss; resid_prototype = copy(resid), jac, jac_prototype)

    return (L == cache.M ? NonlinearProblem : NonlinearLeastSquaresProblem)(nlf, y, cache.p)
end

function __mirk_2point_jacobian!(J, x, diffmode, diffcache, loss_fn::L, resid) where {L}
    sparse_jacobian!(J, diffmode, diffcache, loss_fn, resid, x)
    return J
end

function __mirk_2point_jacobian(x, J, diffmode, diffcache, loss_fn::L) where {L}
    sparse_jacobian!(J, diffmode, diffcache, loss_fn, x)
    return J
end
