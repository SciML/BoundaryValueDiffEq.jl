function SciMLBase.__init(prob::BVProblem, alg::AbstractMIRK; dt = 0.0,
    abstol = 1e-3, adaptive = true, kwargs...)
    has_initial_guess = prob.u0 isa AbstractVector{<:AbstractArray}
    iip = isinplace(prob)
    (T, M, n) = if has_initial_guess
        # If user provided a vector of initial guesses
        _u0 = first(prob.u0)
        eltype(_u0), length(_u0), (length(prob.u0) - 1)
    else
        dt ≤ 0 && throw(ArgumentError("dt must be positive"))
        eltype(prob.u0), length(prob.u0), Int(cld((prob.tspan[2] - prob.tspan[1]), dt))
    end
    chunksize = pickchunksize(M * (n + 1))
    if has_initial_guess
        fᵢ_cache = maybe_allocate_diffcache(vec(similar(_u0)), chunksize, alg.jac_alg)
        fᵢ₂_cache = vec(similar(_u0))
    else
        fᵢ_cache = maybe_allocate_diffcache(vec(similar(prob.u0)), chunksize, alg.jac_alg)
        fᵢ₂_cache = vec(similar(prob.u0))
    end

    # Without this, boxing breaks type stability
    X = has_initial_guess ? _u0 : prob.u0

    # NOTE: Assumes the user provided initial guess is on a uniform mesh
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
    mesh_dt = diff(mesh)

    defect_threshold = T(0.1)  # TODO: Allow user to specify these
    MxNsub = 3000              # TODO: Allow user to specify these

    # Don't flatten this here, since we need to expand it later if needed
    y₀ = __initial_state_from_prob(prob, mesh)
    y = [maybe_allocate_diffcache(vec(copy(yᵢ)), chunksize, alg.jac_alg) for yᵢ in y₀]
    TU, ITU = constructMIRK(alg, T)
    stage = alg_stage(alg)

    k_discrete = [maybe_allocate_diffcache(similar(X, M, stage), chunksize, alg.jac_alg)
                  for _ in 1:n]
    k_interp = adaptive ? [similar(X, M, ITU.s_star - stage) for _ in 1:n] :
               [similar(X, 0, 0) for _ in 1:n]

    resid₁_size = if prob.f.bcresid_prototype === nothing
        size(X)
    elseif prob.f.bcresid_prototype isa ArrayPartition
        size.(prob.f.bcresid_prototype.x)
    else
        size(prob.f.bcresid_prototype)
    end

    if iip
        if prob.f.bcresid_prototype === nothing
            residual = [maybe_allocate_diffcache(vec(copy(yᵢ)), chunksize, alg.jac_alg)
                        for yᵢ in y₀]
        else
            residual = vcat([
                    maybe_allocate_diffcache(vec(copy(prob.f.bcresid_prototype)),
                        chunksize, alg.jac_alg)],
                [maybe_allocate_diffcache(vec(copy(yᵢ)), chunksize, alg.jac_alg)
                 for yᵢ in y₀[2:end]])
        end
    else
        residual = nothing
    end

    defect = adaptive ? [similar(X, M) for _ in 1:n] : [similar(X, 0) for _ in 1:n]

    new_stages = adaptive ? [similar(X, M) for _ in 1:n] : [similar(X, 0) for _ in 1:n]

    # Transform the functions to handle non-vector inputs
    f, bc = if X isa AbstractVector
        prob.f, prob.f.bc
    elseif iip
        function vecf!(du, u, p, t)
            du_ = reshape(du, size(X))
            x_ = reshape(u, size(X))
            prob.f(du_, x_, p, t)
            return du
        end
        vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
            function __vecbc!(resid, sol, p, t)
                resid_ = reshape(resid, resid₁_size)
                sol_ = map(s -> reshape(s, size(X)), sol)
                prob.f.bc(resid_, sol_, p, t)
                return resid
            end
        else
            function __vecbc_a!(resida, ua, p)
                resida_ = reshape(resida, resid₁_size[1])
                ua_ = reshape(ua, size(X))
                prob.f.bc[1](resida_, ua_, p)
                return nothing
            end
            function __vecbc_b!(residb, ub, p)
                residb_ = reshape(residb, resid₁_size[2])
                ub_ = reshape(ub, size(X))
                prob.f.bc[2](residb_, ub_, p)
                return nothing
            end
            (__vecbc_a!, __vecbc_b!)
        end
        vecf!, vecbc!
    else
        function vecf(u, p, t)
            x_ = reshape(u, size(X))
            return vec(prob.f(x_, p, t))
        end
        vecbc = if !(prob.problem_type isa TwoPointBVProblem)
            function __vecbc(sol, p, t)
                sol_ = map(s -> reshape(s, size(X)), sol)
                return vec(prob.f.bc(sol_, p, t))
            end
        else
            __vecbc_a(ua, p) = vec(prob.f.bc[1](reshape(ua, size(X)), p))
            __vecbc_b(ub, p) = vec(prob.f.bc[2](reshape(ub, size(X)), p))
            (__vecbc_a, __vecbc_b)
        end
        vecf, vecbc
    end

    return MIRKCache{iip, T}(alg_order(alg), stage, M, size(X), f, bc, prob,
        prob.problem_type, prob.p, alg, TU, ITU, mesh, mesh_dt, k_discrete, k_interp, y, y₀,
        residual, fᵢ_cache, fᵢ₂_cache, defect, new_stages,
        (; defect_threshold, MxNsub, abstol, dt, adaptive, kwargs...))
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
        nlprob = construct_nlproblem(cache, recursive_flatten(y₀))
        sol_nlprob = solve(nlprob, alg.nlsolve; abstol, kwargs...)
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
                    expand_cache!(cache)
                end
            end
        else
            #  We cannot obtain a solution for the current mesh
            if 2 * (length(cache.mesh) - 1) > MxNsub
                # New mesh would be too large
                info = ReturnCode.Failure
            else
                half_mesh!(cache)
                expand_cache!(cache)
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
function construct_nlproblem(cache::MIRKCache{iip}, y::AbstractVector) where {iip}
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
                Φ!(resids[2:end], cache, y_, u, p)
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

function generate_nlprob(cache::MIRKCache{iip}, y, loss_bc, loss_collocation, loss,
    _) where {iip}
    @unpack nlsolve, jac_alg = cache.alg
    N = length(cache.mesh)

    resid_bc = cache.prob.f.bcresid_prototype === nothing ? similar(y, cache.M) :
               cache.prob.f.bcresid_prototype
    resid_collocation = similar(y, cache.M * (N - 1))

    sd_bc = jac_alg.bc_diffmode isa AbstractSparseADType ? SymbolicsSparsityDetection() :
            NoSparsityDetection()

    cache_bc = __sparse_jacobian_cache(Val(iip), jac_alg.bc_diffmode, sd_bc, loss_bc,
        resid_bc, y)

    sd_collocation = if jac_alg.collocation_diffmode isa AbstractSparseADType
        Jₛ, cvec, rvec = construct_sparse_banded_jac_prototype(y, cache.M, N)
        PrecomputedJacobianColorvec(; jac_prototype = Jₛ, row_colorvec = rvec,
            col_colorvec = cvec)
    else
        NoSparsityDetection()
    end

    cache_collocation = __sparse_jacobian_cache(Val(iip), jac_alg.collocation_diffmode,
        sd_collocation, loss_collocation, resid_collocation, y)

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

    return NonlinearProblem(NonlinearFunction{iip}(loss; jac, jac_prototype), y, cache.p)
end

function generate_nlprob(cache::MIRKCache{iip}, y, loss_bc, loss_collocation, loss,
    ::TwoPointBVProblem) where {iip}
    @unpack nlsolve, jac_alg = cache.alg
    N = length(cache.mesh)

    if !iip && cache.prob.f.bcresid_prototype === nothing
        y_ = recursive_unflatten!(cache.y, y)
        resid_ = cache.bc((y_[1], y_[end]), cache.p)
        resid = ArrayPartition(ArrayPartition(resid_), similar(y, cache.M * (N - 1)))
    else
        resid = ArrayPartition(cache.prob.f.bcresid_prototype,
            similar(y, cache.M * (N - 1)))
    end

    sd = if jac_alg.diffmode isa AbstractSparseADType
        Jₛ, cvec, rvec = construct_sparse_banded_jac_prototype(resid, cache.M, N)
        PrecomputedJacobianColorvec(; jac_prototype = Jₛ, row_colorvec = rvec,
            col_colorvec = cvec)
    else
        NoSparsityDetection()
    end

    diffcache = __sparse_jacobian_cache(Val(iip), jac_alg.diffmode, sd, loss, resid, y)

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

    return NonlinearProblem(NonlinearFunction{iip}(loss; jac, jac_prototype), y, cache.p)
end
