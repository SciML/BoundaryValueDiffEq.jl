# TODO: Differentiate between nlsolve kwargs and odesolve kwargs
# TODO: Support Non-Vector Inputs
function SciMLBase.__solve(prob::BVProblem, alg::Shooting; kwargs...)
    iip = isinplace(prob)
    bc = prob.bc
    u0 = deepcopy(prob.u0)
    loss_fn = if iip
        function loss!(resid, u0, p)
            tmp_prob = ODEProblem{iip}(prob.f, u0, prob.tspan, p)
            internal_sol = solve(tmp_prob, alg.ode_alg; kwargs...)
            eval_bc_residual!(resid, prob.problem_type, bc, internal_sol, p)
            return nothing
        end
    else
        function loss(u0, p)
            tmp_prob = ODEProblem(prob.f, u0, prob.tspan, p)
            internal_sol = solve(tmp_prob, alg.ode_alg; kwargs...)
            return eval_bc_residual(prob.problem_type, bc, internal_sol, p)
        end
    end
    opt = solve(NonlinearProblem(NonlinearFunction{iip}(loss_fn; prob.f.jac_prototype,
                resid_prototype = prob.f.bcresid_prototype), u0, prob.p), alg.nlsolve;
        kwargs...)
    sol_prob = ODEProblem{iip}(prob.f, opt.u, prob.tspan, prob.p)
    sol = solve(sol_prob, alg.ode_alg; kwargs...)
    return DiffEqBase.solution_new_retcode(sol,
        sol.retcode == opt.retcode ? ReturnCode.Success : ReturnCode.Failure)
end
