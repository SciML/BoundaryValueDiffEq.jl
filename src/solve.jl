function DiffEqBase.__solve(prob::BVProblem, alg::Shooting; kwargs...)
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
        sol.retcode == opt.retcode ? ReturnCode.Success :
        ReturnCode.Failure)
end

function __mirk_reorder!(resid)
    # reorder the Jacobian matrix such that it is banded
    tmp_last = resid[end]
    idxs = (lastindex(resid) - 1):-1:firstindex(resid)
    resid[idxs .+ 1] .= resid[idxs]
    resid[firstindex(resid)], resid[end] = resid[end], tmp_last
    return nothing
end

function construct_MIRK_loss_function(S::BVPSystem,
    prob::BVProblem,
    TU,
    cache)
    function loss!(resid, u, p)
        nest_vector!(S.y, u)
        S.p = p
        Φ!(S, TU, cache)
        eval_bc_residual!(prob.problem_type, S)
        flatten_vector!(resid, S.residual)
        __mirk_reorder!(resid)
        return nothing
    end
    return loss!
end

function DiffEqBase.__solve(prob::BVProblem,
    alg::Union{GeneralMIRK, MIRK};
    dt = 0.0,
    abstol = 1e-3,
    kwargs...)
    dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    T = eltype(prob.u0)
    n = Int(cld((prob.tspan[2] - prob.tspan[1]), dt))
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))

    # Initialization
    defect_threshold = T(0.1)
    info::ReturnCode.T = ReturnCode.Success
    defect_norm = 2 * abstol
    MxNsub = 3000
    S = BVPSystem(prob, mesh, alg)

    while info == ReturnCode.Success && defect_norm > abstol
        TU, ITU = constructMIRK(S)
        cache = alg_cache(alg, S)
        # Upper-level iteration
        vec_y = similar(first(S.y), S.M * S.N)

        loss! = construct_MIRK_loss_function(S, prob, TU, cache)
        jac_wrapper = BVPJacobianWrapper(loss!)

        flatten_vector!(vec_y, S.y)
        nlprob = _construct_nonlinear_problem_with_jacobian(jac_wrapper, S, vec_y, prob.p)
        opt = solve(nlprob, alg.nlsolve; kwargs...)
        nest_vector!(S.y, opt.u)

        info = opt.retcode

        if info == ReturnCode.Success
            defect, defect_norm, k_interp = defect_estimate(S, cache, alg, ITU)
            # The defect is greater than 10%, the solution is not acceptable
            defect_norm > defect_threshold && (info = ReturnCode.Failure)
        end

        if info == ReturnCode.Success
            if defect_norm > abstol
                # We construct a new mesh to equidistribute the defect
                mesh_new, Nsub_star, info = mesh_selector(S, alg, defect, abstol)
                # println("New mesh size would be: ", Nsub_star)
                if info == ReturnCode.Success
                    new_Y = map(m -> first(interp_eval(S, cache, alg, ITU, m, k_interp)),
                        mesh_new)
                    S.x = mesh_new
                    S.N = length(S.x)
                    S.y = new_Y
                    S.residual = vector_alloc(eltype(S.x), S.M, S.N)
                end
            end
        else
            #  We cannot obtain a solution for the current mesh
            if 2 * (S.N - 1) > MxNsub
                # New mesh would be too large
                info = ReturnCode.Failure
            else
                S.x = half_mesh(S.x)
                S.N = length(S.x)
                S.y = vector_alloc(eltype(S.x), S.M, S.N)
                S.residual = vector_alloc(eltype(S.x), S.M, S.N)
                info = ReturnCode.Success # Force a restart
                defect_norm = 2 * abstol
            end
        end
    end

    return DiffEqBase.build_solution(prob, alg, S.x, S.y; retcode = info)
end
