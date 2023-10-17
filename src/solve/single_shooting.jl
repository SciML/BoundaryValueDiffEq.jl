function __solve(prob::BVProblem, alg::Shooting; odesolve_kwargs = (;),
    nlsolve_kwargs = (;), verbose = true, kwargs...)
    ig, T, _, _, u0 = __extract_problem_details(prob; dt = 0.1)
    known(ig) && verbose &&
        @warn "Initial guess provided, but will be ignored for Shooting!"

    bcresid_prototype, resid_size = __get_bcresid_prototype(prob, u0)
    iip, bc, u0, u0_size = isinplace(prob), prob.f.bc, deepcopy(u0), size(u0)

    bcresid_prototype_data = __safe_getdata(bcresid_prototype)
    resid_axes = __safe_getaxes(bcresid_prototype)

    loss_fn = if iip
        function loss!(resid, u0_, p)
            resid_ = __maybe_componentarray(resid, resid_axes)
            odeprob = ODEProblem{true}(prob.f, reshape(u0_, u0_size), prob.tspan, p)
            odesol = __solve(odeprob, alg.ode_alg; odesolve_kwargs..., verbose, kwargs...)
            eval_bc_residual!(__safe_reshape(resid_, resid_size), prob.problem_type, bc,
                odesol, p)
            return nothing
        end
    else
        function loss(u0_, p)
            odeprob = ODEProblem{false}(prob.f, reshape(u0_, u0_size), prob.tspan, p)
            odesol = __solve(odeprob, alg.ode_alg; odesolve_kwargs..., verbose, kwargs...)
            return __safe_vec(eval_bc_residual(prob.problem_type, bc, odesol, p))
        end
    end
    opt = __solve(NonlinearProblem(NonlinearFunction{iip}(loss_fn; prob.f.jac_prototype,
                resid_prototype = bcresid_prototype_data), vec(u0), prob.p),
        alg.nlsolve; nlsolve_kwargs..., verbose, kwargs...)
    newprob = ODEProblem{iip}(prob.f, reshape(opt.u, u0_size), prob.tspan, prob.p)
    sol = __solve(newprob, alg.ode_alg; odesolve_kwargs..., verbose, kwargs...)

    if !SciMLBase.successful_retcode(opt)
        return SciMLBase.solution_new_retcode(sol, ReturnCode.Failure)
    end
    return sol
end
