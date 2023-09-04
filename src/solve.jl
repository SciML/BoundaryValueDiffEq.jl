function SciMLBase.__solve(prob::BVProblem, alg; kwargs...)
    # If dispatch not directly defined
    cache = init(prob, alg; kwargs...)
    return solve!(cache)
end

# Shooting Methods

function SciMLBase.__solve(prob::BVProblem, alg::Shooting; kwargs...)
    iip = isinplace(prob)
    bc = prob.bc
    u0 = deepcopy(prob.u0)
    function loss!(resid, u0, p)
        tmp_prob = ODEProblem{iip}(prob.f, u0, prob.tspan, p)
        internal_sol = solve(tmp_prob, alg.ode_alg; kwargs...)
        bc(resid, internal_sol, prob.p, internal_sol.t)
        return nothing
    end
    opt = solve(NonlinearProblem(NonlinearFunction{true}(loss!), u0, prob.p), alg.nlsolve;
        kwargs...)
    sol_prob = ODEProblem{iip}(prob.f, opt.u, prob.tspan, prob.p)
    sol = solve(sol_prob, alg.ode_alg; kwargs...)
    return DiffEqBase.solution_new_retcode(sol,
        sol.retcode == opt.retcode ? ReturnCode.Success : ReturnCode.Failure)
end

# MIRK Methods

function SciMLBase.__init(prob::BVProblem, alg::AbstractMIRK; dt = 0.0, abstol = 1e-3,
    adaptive = true, kwargs...)
    has_initial_guess = prob.u0 isa AbstractVector{<:AbstractArray}
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
        fᵢ_cache = maybe_allocate_diffcache(similar(_u0), chunksize, alg.jac_alg)
        fᵢ₂_cache = similar(_u0)
    else
        fᵢ_cache = maybe_allocate_diffcache(similar(prob.u0), chunksize, alg.jac_alg)
        fᵢ₂_cache = similar(prob.u0)
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
    y = [maybe_allocate_diffcache(copy(yᵢ), chunksize, alg.jac_alg) for yᵢ in y₀]
    TU, ITU = constructMIRK(alg, T)
    stage = alg_stage(alg)

    k_discrete = [maybe_allocate_diffcache(similar(X, M, stage), chunksize, alg.jac_alg)
                  for _ in 1:n]
    k_interp = adaptive ? [similar(X, M, ITU.s_star - stage) for _ in 1:n] :
               [similar(X, 0, 0) for _ in 1:n]

    # FIXME: Here we are making the assumption that size(first(residual)) == size(first(y))
    #        This won't hold true for underconstrained or overconstrained problems
    residual = [maybe_allocate_diffcache(copy(yᵢ), chunksize, alg.jac_alg) for yᵢ in y₀]

    defect = adaptive ? [similar(X, M) for _ in 1:n] : [similar(X, 0) for _ in 1:n]

    new_stages = adaptive ? [similar(X, M) for _ in 1:n] : [similar(X, 0) for _ in 1:n]

    return MIRKCache{T}(alg_order(alg), stage, M, size(X), prob.f, prob.bc, prob,
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
    (defect_threshold, MxNsub, abstol, adaptive, dt), kwargs = __split_mirk_kwargs(;
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

    return DiffEqBase.build_solution(prob, alg, mesh, cache.y₀; retcode = info)
end
