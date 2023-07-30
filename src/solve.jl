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
    abstol = 1e-3,
    kwargs...)
    dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    n = Int(cld((prob.tspan[2] - prob.tspan[1]), dt))
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
    # Initialization
    defect_threshold = 0.1
    info = ReturnCode.Success
    defect_norm = 10
    MxNsub = 3000
    S = BVPSystem(prob, mesh, alg)
    initial_guess = false
    while info == ReturnCode.Success && defect_norm > abstol
        TU, ITU = constructMIRK(S)
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
            Φ!(S, TU, cache)
            if isa(prob.problem_type, TwoPointBVProblem)
                eval_bc_residual!(S)
            else
                general_eval_bc_residual!(S)
            end
            flatten_vector!(resid, S.residual)
            reorder!(resid)
            return nothing
        end

        function loss_with_initial_guess!(resid, u, p)
            @set! S.p = p
            Φ!(S, TU, cache)
            if isa(prob.problem_type, TwoPointBVProblem)
                eval_bc_residual!(S)
            else
                general_eval_bc_residual!(S)
            end
            flatten_vector!(resid, S.residual)
            reorder!(resid)
            return nothing
        end

        jac_wrapper = initial_guess ? BVPJacobianWrapper(loss_with_initial_guess!) :
                      BVPJacobianWrapper(loss!)
        initial_guess = false

        flatten_vector!(vec_y, S.y)
        nlprob = _construct_nonlinear_problem_with_jacobian(jac_wrapper, S, vec_y, prob.p)
        opt = solve(nlprob, alg.nlsolve; kwargs...)
        nest_vector!(S.y, opt.u)

        info = opt.retcode

        if info == ReturnCode.Success
            defect, defect_norm, k_interp = defect_estimate(S, cache, alg, ITU)
            # The defect is greater than 10%, the solution is not acceptable
            if defect_norm > defect_threshold
                info = ReturnCode.Failure
            end
        end

        if info == ReturnCode.Success
            if defect_norm > abstol
                # We construct a new mesh to equidistribute the defect
                mesh_new, Nsub_star, info = mesh_selector(S, alg, defect, abstol)
                #println("New mesh size would be: ", Nsub_star)
                if info == ReturnCode.Success
                    z, z_prime = zeros(S.M), zeros(S.M)
                    new_Y = [zeros(Float64, S.M) for i in 1:(Nsub_star + 1)]
                    for i in 0:Nsub_star
                        z, z_prime = interp_eval(S,
                            cache,
                            alg,
                            ITU,
                            mesh_new[i + 1],
                            k_interp)
                        new_Y[i + 1] = z
                    end
                    S.x = copy(mesh_new)
                    S.N = copy(Nsub_star) + 1
                    S.y = copy(new_Y)
                    initial_guess = true
                    S.residual = vector_alloc(eltype(S.x), S.M, S.N)
                end
            end
        else
            #  We cannot obtain a solution for the current mesh
            if 2 * (S.N - 1) > MxNsub
                # New mesh would be too large
                info = ReturnCode.Failure
            else
                mesh_new = half_mesh(S.x)
                S.x = copy(mesh_new)
                S.N = length(mesh_new)
                S.y = vector_alloc(eltype(S.x), S.M, S.N)
                S.residual = vector_alloc(eltype(S.x), S.M, S.N)
                info = ReturnCode.Success # Force a restart
                defect_norm = 2 * abstol
            end
        end
    end

    return DiffEqBase.build_solution(prob, alg, S.x, S.y; info)
end
