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

function construct_MIRK_loss_function(S::BVPSystem, prob::BVProblem, TU, cache, mesh)
    function loss!(resid, u, p)
        u_ = reshape(u, S.M, S.N)
        resid_ = reshape(resid, S.M, S.N)
        Φ!(resid_, S, TU, cache, u_, p, mesh)
        eval_bc_residual!(resid_, prob.problem_type, S, u_, p, mesh)
        # @show resid
        return resid
    end
    return loss!
end

function DiffEqBase.__solve(prob::BVProblem, alg::Union{GeneralMIRK, MIRK}; dt = 0.0,
    abstol = 1e-3, adaptive::Bool = true, kwargs...)
    dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    T = eltype(prob.u0)
    n = Int(cld((prob.tspan[2] - prob.tspan[1]), dt))
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
    mesh_dt = diff(mesh)

    # Initialization
    defect_threshold = T(0.1)
    info::ReturnCode.T = ReturnCode.Success
    defect_norm = 2 * abstol
    MxNsub = 3000
    S = BVPSystem(prob, mesh, alg)

    y = __initial_state_from_prob(prob, mesh)
    while info == ReturnCode.Success && defect_norm > abstol
        TU, ITU = constructMIRK(alg, S)
        cache = alg_cache(alg, S)

        loss! = construct_MIRK_loss_function(S, prob, TU, cache, mesh)
        jac_wrapper = BVPJacobianWrapper(loss!)

        nlprob = _construct_nonlinear_problem_with_jacobian(jac_wrapper, S, vec(y), prob.p)
        opt = solve(nlprob, alg.nlsolve; abstol, kwargs...)
        vec(y) .= opt.u

        info = opt.retcode

        !adaptive && break

        if info == ReturnCode.Success
            defect, defect_norm, k_interp = defect_estimate(S, cache, alg, ITU, y, prob.p,
                mesh, mesh_dt)
            # The defect is greater than 10%, the solution is not acceptable
            defect_norm > defect_threshold && (info = ReturnCode.Failure)
        end

        if info == ReturnCode.Success
            if defect_norm > abstol
                # We construct a new mesh to equidistribute the defect
                mesh, Nsub_star, info = mesh_selector(S, alg, defect, abstol, mesh,
                    mesh_dt)
                mesh_dt = diff(mesh)
                # println("New mesh size would be: ", Nsub_star)
                if info == ReturnCode.Success
                    y__ = similar(y, S.M, Nsub_star)
                    for (i, m) in enumerate(mesh)
                        y__[:, i] .= first(interp_eval(S, cache, alg, ITU, m, k_interp,
                            mesh, y, mesh_dt))
                    end
                    y = y__
                    S = BVPSystem(prob, mesh, alg)
                end
            end
        else
            #  We cannot obtain a solution for the current mesh
            if 2 * (S.N - 1) > MxNsub
                # New mesh would be too large
                info = ReturnCode.Failure
            else
                mesh = half_mesh(mesh)
                mesh_dt = diff(mesh)
                S = BVPSystem(prob, mesh, alg)
                y = similar(y, S.M, S.N)
                fill!(y, 0)
                info = ReturnCode.Success # Force a restart
                defect_norm = 2 * abstol
            end
        end
    end

    return DiffEqBase.build_solution(prob, alg, mesh, collect(eachcol(y)); retcode = info)
end
