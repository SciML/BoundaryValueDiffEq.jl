using BandedMatrices

# The Solve Function
function DiffEqBase.__solve(prob::BVProblem, alg::Shooting; kwargs...)
    bc = prob.bc
    u0 = deepcopy(prob.u0)
    # Form a root finding function.
    loss = function (resid, minimizer)
        uEltype = eltype(minimizer)
        tmp_prob = remake(prob,u0=minimizer)
        sol = solve(tmp_prob,alg.ode_alg;kwargs...)
        bc(resid,sol,sol.prob.p,sol.t)
        nothing
    end
    opt = alg.nlsolve(loss, u0)
    sol_prob = remake(prob,u0=opt[1])
    sol = solve(sol_prob, alg.ode_alg;kwargs...)
    if sol.retcode == opt[2]
        DiffEqBase.solution_new_retcode(sol,:Success)
    else
        DiffEqBase.solution_new_retcode(sol,:Failure)
    end
    sol
end

function DiffEqBase.__solve(prob::BVProblem, alg::Union{GeneralMIRK,MIRK}; dt=0.0, kwargs...)
    if dt<=0
        error("dt must be positive")
    end
    n = Int(cld((prob.tspan[2]-prob.tspan[1]),dt))
    x = collect(range(prob.tspan[1], stop=prob.tspan[2], length=n))
    S = BVPSystem(prob.f, prob.bc, prob.p, x, length(prob.u0), alg_order(alg))
    if isa(prob.u0, Vector{<:Number})
        copyto!.(S.y, (prob.u0,))
    elseif isa(prob.u0, Vector{<:AbstractArray})
        copyto!(S.y, prob.u0)
    else
        error("u0 must be a Vector or Vector of Arrays")
    end
    tableau = constructMIRK(S)
    cache = alg_cache(alg, S)
    # Upper-level iteration
    vec_y = Array{eltype(S.y[1])}(undef, S.M*S.N)              # Vector
    reorder! = function (resid)
        # reorder the Jacobian matrix such that it is banded
        tmp_last = resid[end]
        for i in (length(resid)-1):-1:1
            resid[i+1] = resid[i]
        end
        resid[1], resid[end] = resid[end], tmp_last
    end
    loss = function (resid, minimizer)
        nest_vector!(S.y, minimizer)
        Φ!(S, tableau, cache)
        isa(prob.problem_type, TwoPointBVProblem) ? eval_bc_residual!(S) : general_eval_bc_residual!(S)
        flatten_vector!(resid, S.residual)
        reorder!(resid)
        nothing
    end

    jac_wrapper = BVPJacobianWrapper(loss)

    flatten_vector!(vec_y, S.y)
    opt = isa(prob.problem_type, TwoPointBVProblem) ? alg.nlsolve(ConstructJacobian(jac_wrapper, vec_y), vec_y) : alg.nlsolve(ConstructJacobian(jac_wrapper, S, vec_y), vec_y) # Sparse matrix is broken
    nest_vector!(S.y, opt[1])

    retcode = opt[2] ? :Success : :Failure
    DiffEqBase.build_solution(prob, alg, x, S.y, retcode = retcode)
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
    retcode = opt[2] ? :Success : :Failure
    DiffEqBase.build_solution(prob, alg, x, S.y, retcode = retcode)
end
=#
