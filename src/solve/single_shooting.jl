function SciMLBase.__solve(prob::BVProblem, alg::Shooting; odesolve_kwargs = (;),
    nlsolve_kwargs = (;), kwargs...)
    iip, bc, u0, u0_size = isinplace(prob), prob.f.bc, deepcopy(prob.u0), size(prob.u0)
    resid_size = prob.f.bcresid_prototype === nothing ? u0_size :
                 size(prob.f.bcresid_prototype)
    loss_fn = if iip
        function loss!(resid, u0_, p)
            u0_internal = reshape(u0_, u0_size)
            tmp_prob = ODEProblem{iip}(prob.f, u0_internal, prob.tspan, p)
            internal_sol = solve(tmp_prob, alg.ode_alg; odesolve_kwargs..., kwargs...)
            eval_bc_residual!(reshape(resid, resid_size), prob.problem_type, bc,
                internal_sol, p)
            return nothing
        end
    else
        function loss(u0_, p)
            u0_internal = reshape(u0_, u0_size)
            tmp_prob = ODEProblem(prob.f, u0_internal, prob.tspan, p)
            internal_sol = solve(tmp_prob, alg.ode_alg; odesolve_kwargs..., kwargs...)
            return vec(eval_bc_residual(prob.problem_type, bc, internal_sol, p))
        end
    end
    opt = solve(NonlinearProblem(NonlinearFunction{iip}(loss_fn; prob.f.jac_prototype,
                resid_prototype = prob.f.bcresid_prototype), vec(u0), prob.p), alg.nlsolve;
        nlsolve_kwargs..., kwargs...)
    newprob = ODEProblem{iip}(prob.f, reshape(opt.u, u0_size), prob.tspan, prob.p)
    sol = solve(newprob, alg.ode_alg; odesolve_kwargs..., kwargs...)

    if !SciMLBase.successful_retcode(opt)
        return SciMLBase.solution_new_retcode(sol, ReturnCode.Failure)
    end
    return sol
end
