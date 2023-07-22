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

function DiffEqBase.__solve(prob::BVProblem, alg::Union{GeneralMIRK, MIRK}; dt = 0.0,
    kwargs...)
    dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    n = Int(cld((prob.tspan[2] - prob.tspan[1]), dt))
    x = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
    S = BVPSystem(prob, x, alg)

    tableau = constructMIRK(S)
    cache = alg_cache(alg, S)
    # Upper-level iteration
    vec_y = Array{eltype(first(S.y))}(undef, S.M * S.N)              # Vector
    function reorder!(resid)
        # reorder the Jacobian matrix such that it is banded
        tmp_last = resid[end]
        for i in (length(resid) - 1):-1:1
            resid[i + 1] = resid[i]
        end
        resid[1], resid[end] = resid[end], tmp_last
        return nothing
    end
    function loss!(resid, u0, p)
        nest_vector!(S.y, u0)
        @set! S.p = p
        Φ!(S, tableau, cache)
        if isa(prob.problem_type, TwoPointBVProblem)
            eval_bc_residual!(S)
        else
            general_eval_bc_residual!(S)
        end
        flatten_vector!(resid, S.residual)
        reorder!(resid)
        return nothing
    end

    jac_wrapper = BVPJacobianWrapper(loss!)

    flatten_vector!(vec_y, S.y)
    nlprob = _construct_nonlinear_problem_with_jacobian(jac_wrapper, S, vec_y, prob.p)
    opt = solve(nlprob, alg.nlsolve; kwargs...)
    nest_vector!(S.y, opt.u)

    return DiffEqBase.build_solution(prob, alg, x, S.y; opt.retcode)
end

#=
function solve(prob::BVProblem, alg::GeneralMIRK; dt=0.0, kwargs...)
    n = Int(cld((prob.tspan[2]-prob.tspan[1]),dt))
    x = collect(linspace(prob.tspan..., n+1))
    S = BVPSystem(prob.f, prob.bc, x, length(prob.u0), alg_order(alg))
    S.y[1] = prob.u0
    tableau = constructMIRK(S)
    cache = alg_cache(alg, S)
    # Upper-level iteration
    vec_y = Array{eltype(S.y[1])}(S.M*S.N)              # Vector
    loss = function (minimizer, resid)
        nest_vector!(S.y, minimizer)
        Φ!(S, tableau, cache)
        general_eval_bc_residual!(S)
        flatten_vector!(resid, S.residual)
        nothing
    end
    flatten_vector!(vec_y, S.y)
    opt = alg.nlsolve(loss, vec_y)
    nest_vector!(S.y, opt[1])
    retcode = opt[2] ? ReturnCode.Success : ReturnCode.Failure
    DiffEqBase.build_solution(prob, alg, x, S.y, retcode = retcode)
end
=#
