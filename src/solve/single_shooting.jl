function __solve(prob::BVProblem, alg::Shooting; odesolve_kwargs = (;),
    nlsolve_kwargs = (;), verbose = true, kwargs...)
    ig, T, _, _, u0 = __extract_problem_details(prob; dt = 0.1)
    _unwrap_val(ig) && verbose &&
        @warn "Initial guess provided, but will be ignored for Shooting!"

    bcresid_prototype, resid_size = __get_bcresid_prototype(prob, u0)
    iip, bc, u0, u0_size = isinplace(prob), prob.f.bc, deepcopy(u0), size(u0)
    resid_prototype = __vec(bcresid_prototype)

    ode_kwargs = (; kwargs..., verbose, odesolve_kwargs...)
    loss_fn = if iip
        (du, u, p) -> __single_shooting_loss!(du, u, p, prob.f, bc, u0_size, prob.tspan,
            prob.problem_type, resid_size, alg, ode_kwargs)
    else
        (u, p) -> __single_shooting_loss(u, p, prob.f, bc, u0_size, prob.tspan,
            prob.problem_type, alg, ode_kwargs)
    end

    opt = __solve(NonlinearProblem(NonlinearFunction{iip}(loss_fn; prob.f.jac_prototype,
                resid_prototype), vec(u0), prob.p), alg.nlsolve;
        nlsolve_kwargs..., verbose, kwargs...)
    newprob = ODEProblem{iip}(prob.f, reshape(opt.u, u0_size), prob.tspan, prob.p)
    sol = __solve(newprob, alg.ode_alg; odesolve_kwargs..., verbose, kwargs...)

    !SciMLBase.successful_retcode(opt) &&
        return SciMLBase.solution_new_retcode(sol, ReturnCode.Failure)
    return sol
end

function __single_shooting_loss!(resid_, u0_, p, f, bc, u0_size, tspan,
    pt::TwoPointBVProblem, (resida_size, residb_size), alg::Shooting, kwargs)
    resida = @view resid_[1:prod(resida_size)]
    residb = @view resid_[(prod(resida_size) + 1):end]
    resid = (reshape(resida, resida_size), reshape(residb, residb_size))

    odeprob = ODEProblem{true}(f, reshape(u0_, u0_size), tspan, p)
    odesol = __solve(odeprob, alg.ode_alg; kwargs...)
    eval_bc_residual!(resid, pt, bc, odesol, p)

    return nothing
end

function __single_shooting_loss!(resid_, u0_, p, f, bc, u0_size, tspan,
    pt::StandardBVProblem, resid_size, alg::Shooting, kwargs)
    resid = reshape(resid_, resid_size)

    odeprob = ODEProblem{true}(f, reshape(u0_, u0_size), tspan, p)
    odesol = __solve(odeprob, alg.ode_alg; kwargs...)
    eval_bc_residual!(resid, pt, bc, odesol, p)

    return nothing
end

function __single_shooting_loss(u0_, p, f, bc, u0_size, tspan, pt, alg::Shooting, kwargs)
    odeprob = ODEProblem{false}(f, reshape(u0_, u0_size), tspan, p)
    odesol = __solve(odeprob, alg.ode_alg; kwargs...)
    return __safe_vec(eval_bc_residual(pt, bc, odesol, p))
end
