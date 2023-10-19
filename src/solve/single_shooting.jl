function __solve(prob::BVProblem, alg::Shooting; odesolve_kwargs = (;),
    nlsolve_kwargs = (;), verbose = true, kwargs...)
    ig, T, _, _, u0 = __extract_problem_details(prob; dt = 0.1)
    _unwrap_val(ig) && verbose &&
        @warn "Initial guess provided, but will be ignored for Shooting!"

    bcresid_prototype, resid_size = __get_bcresid_prototype(prob, u0)
    iip, bc, u0, u0_size = isinplace(prob), prob.f.bc, deepcopy(u0), size(u0)
    resid_prototype = __vec(bcresid_prototype)

    loss_fn = if iip
        function loss!(resid_, u0_, p)
            if prob.problem_type isa TwoPointBVProblem
                resida = @view resid_[1:prod(resid_size[1])]
                residb = @view resid_[(prod(resid_size[1]) + 1):end]
                resid = (reshape(resida, resid_size[1]), reshape(residb, resid_size[2]))
            else
                resid = reshape(resid_, resid_size)
            end
            odeprob = ODEProblem{true}(prob.f, reshape(u0_, u0_size), prob.tspan, p)
            odesol = __solve(odeprob, alg.ode_alg; odesolve_kwargs..., verbose, kwargs...)
            eval_bc_residual!(resid, prob.problem_type, bc, odesol, p)
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
                resid_prototype), vec(u0), prob.p), alg.nlsolve;
        nlsolve_kwargs..., verbose, kwargs...)
    newprob = ODEProblem{iip}(prob.f, reshape(opt.u, u0_size), prob.tspan, prob.p)
    sol = __solve(newprob, alg.ode_alg; odesolve_kwargs..., verbose, kwargs...)

    if !SciMLBase.successful_retcode(opt)
        return SciMLBase.solution_new_retcode(sol, ReturnCode.Failure)
    end
    return sol
end
