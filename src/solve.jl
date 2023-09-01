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
    kwargs...)
    # TODO: Avoid allocating a few of these if adaptive is false
    # TODO: Only allocate as DualCache if jac_alg has ForwardDiff
    _u0 = first(prob.u0)
    if _u0 isa AbstractArray
        # If user provided a vector of initial guesses
        T, M, n = eltype(_u0), length(_u0), (length(prob.u0) - 1)
        fᵢ_cache = DiffCache(similar(_u0), pickchunksize(M * (n + 1)))
        fᵢ₂_cache = similar(prob.u0)
    else
        dt ≤ 0 && throw(ArgumentError("dt must be positive"))
        T = eltype(prob.u0)
        M = length(prob.u0)
        n = Int(cld((prob.tspan[2] - prob.tspan[1]), dt))
        fᵢ_cache = DiffCache(similar(prob.u0), pickchunksize(M * (n + 1)))
        fᵢ₂_cache = similar(prob.u0)
    end
    # NOTE: Assumes the user provided initial guess is on a uniform mesh
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
    mesh_dt = diff(mesh)

    defect_threshold = T(0.1)  # TODO: Allow user to specify these
    MxNsub = 3000              # TODO: Allow user to specify these

    # Don't flatten this here, since we need to expand it later if needed
    y₀ = __initial_state_from_prob(prob, mesh)
    y = [DiffCache(yᵢ, pickchunksize(M * (n + 1))) for yᵢ in y₀]
    TU, ITU = constructMIRK(alg, T)
    stage = alg_stage(alg)

    k_discrete = [DiffCache(similar(fᵢ_cache.du, M, stage), pickchunksize(M * (n + 1)))
                  for _ in 1:n]
    k_interp = [similar(fᵢ_cache.du, M, ITU.s_star - stage) for _ in 1:n]

    # FIXME: Here we are making the assumption that size(first(residual)) == size(first(y))
    #        This won't hold true for underconstrained or overconstrained problems
    residual = [DiffCache(copy(yᵢ), pickchunksize(M * (n + 1))) for yᵢ in y₀]

    defect = [similar(fᵢ_cache.du, M) for _ in 1:n]

    new_stages = [similar(fᵢ_cache.du, M) for _ in 1:n]

    return MIRKCache{T}(alg_order(alg), stage, M, size(fᵢ_cache.du), prob.f, prob.bc,
        prob, prob.problem_type, prob.p, alg, TU, ITU, mesh, mesh_dt, k_discrete, k_interp,
        y, y₀, residual, fᵢ_cache, fᵢ₂_cache, defect, new_stages,
        (; defect_threshold, MxNsub, abstol, dt, kwargs...))
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

    vec_y = recursive_flatten(y₀)
    while SciMLBase.successful_retcode(info) && defect_norm > abstol
        nlprob = construct_nlproblem(cache, vec_y)
        sol_nlprob = solve(nlprob, alg.nlsolve; abstol, kwargs...)
        copyto!(vec_y, sol_nlprob.u)

        info = sol_nlprob.retcode

        !adaptive && break



        if info == ReturnCode.Success
            # FIXME: Defect Estimate is incorrect
            defect_norm = defect_estimate!(cache)
            # The defect is greater than 10%, the solution is not acceptable
            defect_norm > defect_threshold && (info = ReturnCode.Failure)
            @show defect_norm, info
        end

        # break
    end

    recursive_unflatten!(cache.y₀, vec_y)
    return DiffEqBase.build_solution(prob, alg, mesh, cache.y₀; retcode = info)
end

#=
function DiffEqBase.__solve(prob::BVProblem, alg::AbstractMIRK; dt = 0.0, abstol = 1e-3,
    adaptive::Bool = true, kwargs...)
    while info == ReturnCode.Success && defect_norm > abstol
        cache = alg_cache(alg, S, vec_y)

        #     if info == ReturnCode.Success
        #         if defect_norm > abstol
        #             # We construct a new mesh to equidistribute the defect
        #             mesh_new, Nsub_star, info = mesh_selector(S, alg, defect, abstol, mesh,
        #                 mesh_dt)
        #             mesh_dt_new = diff(mesh_new)
        #             # println("New mesh size would be: ", Nsub_star)
        #             if info == ReturnCode.Success
        #                 y__ = similar(y, S.M, Nsub_star + 1)
        #                 for (i, m) in enumerate(mesh_new)
        #                     y__[:, i] .= first(interp_eval(S, cache, alg, ITU, m, k_interp,
        #                         mesh, y, mesh_dt))
        #                 end
        #                 y = y__
        #                 S = BVPSystem(prob, mesh_new, alg, y)
        #                 mesh = mesh_new
        #                 mesh_dt = mesh_dt_new
        #             end
        #         end
        #     else
        #         #  We cannot obtain a solution for the current mesh
        #         if 2 * (S.N - 1) > MxNsub
        #             # New mesh would be too large
        #             info = ReturnCode.Failure
        #         else
        #             mesh = half_mesh(mesh)
        #             mesh_dt = diff(mesh)
        #             S = BVPSystem(prob, mesh, alg, y)
        #             y = similar(y, S.M, S.N)
        #             fill!(y, 0)
        #             info = ReturnCode.Success # Force a restart
        #             defect_norm = 2 * abstol
        #         end
        #     end
    end

    # return DiffEqBase.build_solution(prob, alg, mesh, collect(eachcol(y)); retcode = info)

    return
end
=#
