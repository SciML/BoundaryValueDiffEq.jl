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
        sol.retcode == opt.retcode ? ReturnCode.Success : ReturnCode.Failure)
end

function DiffEqBase.__solve(prob::BVProblem, alg::AbstractMIRK; dt = 0.0, abstol = 1e-3,
    adaptive::Bool = true, kwargs...)
    dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    (T, n) = if first(prob.u0) isa AbstractArray
        eltype(first(prob.u0)), (length(prob.u0) - 1)
    else
        eltype(prob.u0), Int(cld((prob.tspan[2] - prob.tspan[1]), dt))
    end
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
    mesh_dt = diff(mesh)

    # Initialization
    defect_threshold = T(0.1)
    info::ReturnCode.T = ReturnCode.Success
    defect_norm = 2 * abstol
    MxNsub = 3000

    y = __initial_state_from_prob(prob, mesh)
    S = BVPSystem(prob, mesh, alg, y)
    TU, ITU = constructMIRK(alg, S)
    while info == ReturnCode.Success && defect_norm > abstol
        cache = alg_cache(alg, S, y)

        nlprob = construct_MIRK_nlproblem(S, prob, TU, cache, mesh, vec(y), alg.jac_alg)
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
                mesh_new, Nsub_star, info = mesh_selector(S, alg, defect, abstol, mesh,
                    mesh_dt)
                mesh_dt_new = diff(mesh_new)
                # println("New mesh size would be: ", Nsub_star)
                if info == ReturnCode.Success
                    y__ = similar(y, S.M, Nsub_star + 1)
                    for (i, m) in enumerate(mesh_new)
                        y__[:, i] .= first(interp_eval(S, cache, alg, ITU, m, k_interp,
                            mesh, y, mesh_dt))
                    end
                    y = y__
                    S = BVPSystem(prob, mesh_new, alg, y)
                    mesh = mesh_new
                    mesh_dt = mesh_dt_new
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
                S = BVPSystem(prob, mesh, alg, y)
                y = similar(y, S.M, S.N)
                fill!(y, 0)
                info = ReturnCode.Success # Force a restart
                defect_norm = 2 * abstol
            end
        end
    end

    return DiffEqBase.build_solution(prob, alg, mesh, collect(eachcol(y)); retcode = info)
end
