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
    has_initial_guess, T, M, n, X = __extract_problem_details(prob; dt,
        check_positive_dt = true)
    chunksize = pickchunksize(M * (n + 1))

    __alloc_diffcache = x -> __maybe_allocate_diffcache(vec(x), chunksize, alg.jac_alg)

    fᵢ_cache = __alloc_diffcache(similar(X))
    fᵢ₂_cache = vec(similar(X))

    # NOTE: Assumes the user provided initial guess is on a uniform mesh
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
    mesh_dt = diff(mesh)

    defect_threshold = T(0.1)  # TODO: Allow user to specify these
    MxNsub = 3000              # TODO: Allow user to specify these

    # Don't flatten this here, since we need to expand it later if needed
    y₀ = __initial_state_from_prob(prob, mesh)
    y = __alloc_diffcache.(copy.(y₀))
    TU, ITU = constructMIRK(alg, T)
    stage = alg_stage(alg)

    k_discrete = [__maybe_allocate_diffcache(similar(X, M, stage), chunksize, alg.jac_alg)
                  for _ in 1:n]
    k_interp = [similar(X, ifelse(adaptive, M, 0), ifelse(adaptive, ITU.s_star - stage, 0))
                for _ in 1:n]

    bcresid_prototype, resid₁_size = __get_bcresid_prototype(prob.problem_type, prob, X)

    residual = if iip
        if prob.problem_type isa TwoPointBVProblem
            vcat([__alloc_diffcache(__vec(bcresid_prototype))],
                __alloc_diffcache.(copy.(@view(y₀[2:end]))))
        else
            vcat([__alloc_diffcache(bcresid_prototype)],
                __alloc_diffcache.(copy.(@view(y₀[2:end]))))
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
        vecf!(du, u, p, t) = prob.f(reshape(du, size(X)), reshape(u, size(X)), p, t)
        vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
            function __vecbc!(resid, sol, p, t)
                prob.f.bc(reshape(resid, resid₁_size),
                    map(Base.Fix2(reshape, size(X)), sol), p, t)
            end
        else
            function __vecbc_a!(resida, ua, p)
                prob.f.bc[1](reshape(resida, resid₁_size[1]), reshape(ua, size(X)), p)
            end
            function __vecbc_b!(residb, ub, p)
                prob.f.bc[2](reshape(residb, resid₁_size[2]), reshape(ub, size(X)), p)
            end
            (__vecbc_a!, __vecbc_b!)
        end
        vecf!, vecbc!
    else
        vecf(u, p, t) = vec(prob.f(reshape(u, size(X)), p, t))
        vecbc = if !(prob.problem_type isa TwoPointBVProblem)
            __vecbc(sol, p, t) = vec(prob.f.bc(map(Base.Fix2(reshape, size(X)), sol), p, t))
        else
            __vecbc_a(ua, p) = vec(prob.f.bc[1](reshape(ua, size(X)), p))
            __vecbc_b(ub, p) = vec(prob.f.bc[2](reshape(ub, size(X)), p))
            (__vecbc_a, __vecbc_b)
        end
        vecf, vecbc
    end

    return MIRKCache{iip, T}(alg_order(alg), stage, M, size(X), f, bc, prob,
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
                        interp_eval!(cache.y₀[i], cache, cache.y₀, m, mesh, mesh_dt)
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
<<<<<<< HEAD
function __construct_nlproblem(cache::MIRKCache{iip}, y::AbstractVector) where {iip}
    loss_bc = if iip
        function loss_bc_internal!(resid::AbstractVector, u::AbstractVector, p = cache.p)
            y_ = recursive_unflatten!(cache.y, u)
            bc_sol_ = DiffEqBase.build_solution(cache.prob,
                cache.alg,
                cache.mesh,
                y_;
                interp = MIRKInterpolation(cache.mesh, y_, cache)) # build solution @ any time
            eval_bc_residual!(resid, cache.problem_type, cache.bc, bc_sol_, p, cache.mesh)
            return resid
=======
function construct_nlproblem(cache::MIRKCache{iip}, y::AbstractVector) where {iip}
    loss_bc = if !(cache.problem_type isa TwoPointBVProblem)
        if iip
            function loss_bc_internal!(resid::AbstractVector,
                u::AbstractVector,
                p = cache.p)
                y_ = recursive_unflatten!(cache.y, u)
                bc_sol_ = DiffEqBase.build_solution(cache.prob,
                    cache.alg,
                    cache.mesh,
                    y_;
                    interp = MIRKInterpolation(cache.mesh, y_, cache)) # build solution @ any time
                eval_bc_residual!(resid,
                    cache.problem_type,
                    cache.bc,
                    bc_sol_,
                    p,
                    cache.mesh)
                return resid
            end
        else
            function loss_bc_internal(u::AbstractVector, p = cache.p)
                y_ = recursive_unflatten!(cache.y, u)
                return eval_bc_residual(cache.problem_type, cache.bc, y_, p, cache.mesh)
            end
>>>>>>> 991147b (add conditional for multipoint problem bc)
        end
    else
        if iip
            function loss_bc_internal_2point!(resid::AbstractVector,
                u::AbstractVector,
                p = cache.p)
                y_ = recursive_unflatten!(cache.y, u)
                eval_bc_residual!(resid, cache.problem_type, cache.bc, y_, p, cache.mesh)
                return resid
            end
        else
            function loss_bc_internal_2point(u::AbstractVector, p = cache.p)
                y_ = recursive_unflatten!(cache.y, u)
                return eval_bc_residual(cache.problem_type, cache.bc, y_, p, cache.mesh)
            end
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
            return mapreduce(vec, vcat, resids)
        end
    end

<<<<<<< HEAD
    loss = if iip
        @views function loss_internal!(resid::AbstractVector,
            u::AbstractVector,
            p = cache.p)
            y_ = recursive_unflatten!(cache.y, u)
            resids = [get_tmp(r, u) for r in cache.residual]
            resid_bc = if cache.problem_type isa TwoPointBVProblem
                (resids[1][1:prod(cache.resid_size[1])],
                    resids[1][(prod(cache.resid_size[1]) + 1):end])
            else
                resids[1]
            end
            eval_bc_residual!(resid_bc, cache.problem_type, cache.bc, y_, p, cache.mesh)
            Φ!(resids[2:end], cache, y_, u, p)
            if cache.problem_type isa TwoPointBVProblem
                recursive_flatten_twopoint!(resid, resids, cache.resid_size)
            else
=======
    loss = if !(cache.problem_type isa TwoPointBVProblem)
        if iip
            function loss_internal!(resid::AbstractVector, u::AbstractVector, p = cache.p)
                y_ = recursive_unflatten!(cache.y, u)
                resids = [get_tmp(r, u) for r in cache.residual]
                intern_sol_ = DiffEqBase.build_solution(cache.prob,
                    cache.alg,
                    cache.mesh,
                    y_;
                    interp = MIRKInterpolation(cache.mesh, y_, cache)) # build solution @ any time
                eval_bc_residual!(resids[1], cache.problem_type, cache.bc, intern_sol_, p,
                    cache.mesh)
                Φ!(resids[2:end], cache, y_, u, p)
>>>>>>> 5a37ad4 (build solution when constructing nonliear problem)
                recursive_flatten!(resid, resids)
            end
            return resid
        end
    else
        function loss_internal(u::AbstractVector, p = cache.p)
            y_ = recursive_unflatten!(cache.y, u)
            resid_bc = eval_bc_residual(cache.problem_type, cache.bc, y_, p, cache.mesh)
            resid_co = Φ(cache, y_, u, p)
            if cache.problem_type isa TwoPointBVProblem
                return vcat(resid_bc[1], mapreduce(vec, vcat, resid_co), resid_bc[2])
            else
                return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
            end
        end
    end

    return __construct_nlproblem(cache, y, loss_bc, loss_collocation, loss,
        cache.problem_type)
end

function __construct_nlproblem(cache::MIRKCache{iip}, y, loss_bc, loss_collocation, loss,
    ::StandardBVProblem) where {iip}
    @unpack nlsolve, jac_alg = cache.alg
    N = length(cache.mesh)

    resid_bc = cache.bcresid_prototype
    resid_collocation = similar(y, cache.M * (N - 1))

    sd_bc = jac_alg.bc_diffmode isa AbstractSparseADType ? SymbolicsSparsityDetection() :
            NoSparsityDetection()
    cache_bc = __sparse_jacobian_cache(Val(iip), jac_alg.bc_diffmode, sd_bc, loss_bc,
        resid_bc, y)

    sd_collocation = if jac_alg.nonbc_diffmode isa AbstractSparseADType
        PrecomputedJacobianColorvec(__generate_sparse_jacobian_prototype(cache,
            cache.problem_type, y, y, cache.M, N))
    else
        NoSparsityDetection()
    end
    cache_collocation = __sparse_jacobian_cache(Val(iip), jac_alg.nonbc_diffmode,
        sd_collocation, loss_collocation, resid_collocation, y)

    jac_prototype = vcat(init_jacobian(cache_bc), init_jacobian(cache_collocation))

    jac = if iip
        function jac_internal!(J, x, p)
            sparse_jacobian!(@view(J[1:(cache.M), :]), jac_alg.bc_diffmode, cache_bc,
                loss_bc, resid_bc, x)
            sparse_jacobian!(@view(J[(cache.M + 1):end, :]), jac_alg.nonbc_diffmode,
                cache_collocation, loss_collocation, resid_collocation, x)
            return J
        end
    else
        J_ = jac_prototype
        function jac_internal(x, p)
            sparse_jacobian!(@view(J_[1:(cache.M), :]), jac_alg.bc_diffmode, cache_bc,
                loss_bc, x)
            sparse_jacobian!(@view(J_[(cache.M + 1):end, :]), jac_alg.nonbc_diffmode,
                cache_collocation, loss_collocation, x)
            return J_
        end
    end

    return NonlinearProblem(NonlinearFunction{iip}(loss; jac, jac_prototype), y, cache.p)
end

function __construct_nlproblem(cache::MIRKCache{iip}, y, loss_bc, loss_collocation,
    loss, ::TwoPointBVProblem) where {iip}
    @unpack nlsolve, jac_alg = cache.alg
    N = length(cache.mesh)

    resid = vcat(cache.bcresid_prototype[1:prod(cache.resid_size[1])],
        similar(y, cache.M * (N - 1)),
        cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end])

    sd = if jac_alg.diffmode isa AbstractSparseADType
        PrecomputedJacobianColorvec(__generate_sparse_jacobian_prototype(cache,
            cache.problem_type, @view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
            @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]), cache.M,
            N))
    else
        NoSparsityDetection()
    end
    diffcache = __sparse_jacobian_cache(Val(iip), jac_alg.diffmode, sd, loss, resid, y)
    jac_prototype = init_jacobian(diffcache)

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
