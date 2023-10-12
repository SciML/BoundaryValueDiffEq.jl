function __solve(prob::BVProblem, _alg::MultipleShooting; odesolve_kwargs = (;),
    nlsolve_kwargs = (;), ensemblealg = EnsembleThreads(), verbose = true, kwargs...)
    @unpack f, tspan = prob

    ig, T, N, Nig, u0 = __extract_problem_details(prob; dt = 0.1)
    has_initial_guess = known(ig)

    bcresid_prototype, resid_size = __get_bcresid_prototype(prob, u0)
    iip, bc, u0, u0_size = isinplace(prob), prob.f.bc, deepcopy(u0), size(u0)

    __alg = concretize_jacobian_algorithm(_alg, prob)
    alg = if has_initial_guess && Nig != __alg.nshoots + 1
        verbose &&
            @warn "Initial guess length != `nshoots + 1`! Adapting to `nshoots = $(Nig - 1)`"
        update_nshoots(__alg, Nig - 1)
    else
        __alg
    end
    nshoots = alg.nshoots

    if prob.problem_type isa TwoPointBVProblem
        resida_len = length(bcresid_prototype.x[1])
        residb_len = length(bcresid_prototype.x[2])
    end

    # We will use colored AD for this part!
    @views function solve_internal_odes!(resid_nodes, us, p, cur_nshoots, nodes)
        ts_ = Vector{Vector{typeof(first(tspan))}}(undef, cur_nshoots)
        us_ = Vector{Vector{typeof(us)}}(undef, cur_nshoots)

        function prob_func(probᵢ, i, repeat)
            return remake(probᵢ; u0 = reshape(us[((i - 1) * N + 1):(i * N)], u0_size),
                tspan = (nodes[i], nodes[i + 1]))
        end

        function reduction(u, data, I)
            for i in I
                u.us[i] = data[i].u
                u.ts[i] = data[i].t
                u.resid[((i - 1) * N + 1):(i * N)] .= vec(us[(i * N + 1):((i + 1) * N)]) .-
                                                      vec(data[i].u[end])
            end
            return (u, false)
        end

        odeprob = ODEProblem{iip}(f, reshape(us[1:N], u0_size), tspan, p)

        ensemble_prob = EnsembleProblem(odeprob; prob_func, reduction, safetycopy = false,
            u_init = (; us = us_, ts = ts_, resid = resid_nodes))
        ensemble_sol = __solve(ensemble_prob, alg.ode_alg, ensemblealg; odesolve_kwargs...,
            verbose, kwargs..., save_end = true, save_everystep = false,
            trajectories = cur_nshoots)

        return reduce(vcat, ensemble_sol.u.us), reduce(vcat, ensemble_sol.u.ts)
    end

    compute_bc_residual! = if prob.problem_type isa TwoPointBVProblem
        @views function compute_bc_residual_tp!(resid_bc, us::ArrayPartition, p,
            cur_nshoots, nodes, resid_nodes::Union{Nothing, MaybeDiffCache} = nothing)
            ua, ub0 = us.x
            # Just Recompute the last ODE Solution
            lastodeprob = ODEProblem{iip}(f, reshape(ub0, u0_size),
                (nodes[end - 1], nodes[end]), p)
            sol_ode_last = __solve(lastodeprob, alg.ode_alg; odesolve_kwargs..., verbose,
                kwargs..., save_everystep = false, saveat = (), save_end = true)
            ub = vec(sol_ode_last.u[end])

            resid_bc_a, resid_bc_b = if resid_bc isa ArrayPartition
                resid_bc.x
            else
                resid_bc[1:resida_len], resid_bc[(resida_len + 1):end]
            end

            if iip
                bc[1](resid_bc_a, ua, p)
                bc[2](resid_bc_b, ub, p)
            else
                resid_bc_a .= bc[1](ua, p)
                resid_bc_b .= bc[2](ub, p)
            end

            return resid_bc
        end
    else
        @views function compute_bc_residual_mp!(resid_bc, us, p, cur_nshoots, nodes,
            resid_nodes::Union{Nothing, MaybeDiffCache} = nothing)
            if resid_nodes === nothing
                _resid_nodes = similar(us, cur_nshoots * N)  # This might be Dual based on `us`
            else
                _resid_nodes = get_tmp(resid_nodes, us)
            end

            # NOTE: We need to recompute this to correctly propagate the dual numbers / gradients
            _us, _ts = solve_internal_odes!(_resid_nodes, us, p, cur_nshoots, nodes)

            # Boundary conditions
            # Builds an ODESolution object to keep the framework for bc(,,) consistent
            odeprob = ODEProblem{iip}(f, reshape(us[1:N], u0_size), tspan, p)
            total_solution = SciMLBase.build_solution(odeprob, alg.ode_alg, _ts, _us)

            if iip
                eval_bc_residual!(resid_bc, prob.problem_type, bc, total_solution, p)
            else
                resid_bc .= eval_bc_residual(prob.problem_type, bc, total_solution, p)
            end

            return resid_bc
        end
    end

    @views function loss!(resid::ArrayPartition, us, p, cur_nshoots, nodes)
        resid_bc, resid_nodes = resid.x[1], resid.x[2]

        _us, _ts = solve_internal_odes!(resid_nodes, us, p, cur_nshoots, nodes)

        # Boundary conditions
        # Builds an ODESolution object to keep the framework for bc(,,) consistent
        odeprob = ODEProblem{iip}(f, reshape(us[1:N], u0_size), tspan, p)
        total_solution = SciMLBase.build_solution(odeprob, alg.ode_alg, _ts, _us)

        if iip
            eval_bc_residual!(resid_bc, prob.problem_type, bc, total_solution, p)
        else
            resid_bc .= eval_bc_residual(prob.problem_type, bc, total_solution, p)
        end

        return resid
    end

    jac! = if prob.problem_type isa TwoPointBVProblem
        @views function jac_tp!(J::AbstractMatrix, us, p, resid_bc,
            resid_nodes::MaybeDiffCache, ode_jac_cache, bc_jac_cache::Tuple, ode_fn, bc_fn,
            cur_nshoot, nodes)
            # This is mostly a safety measure
            fill!(J, 0)

            J_bc = J[1:N, :]
            J_c = J[(N + 1):end, :]

            sparse_jacobian!(J_c, alg.jac_alg.nonbc_diffmode, ode_jac_cache, ode_fn,
                resid_nodes.du, us)

            # For BC
            bc_jac_cache′, J_bc′ = bc_jac_cache
            sparse_jacobian!(J_bc′, alg.jac_alg.bc_diffmode, bc_jac_cache′, bc_fn,
                resid_bc, ArrayPartition(us[1:N], us[(end - N + 1):end]))
            resida, residb = resid_bc.x
            J_bc[1:length(resida), 1:N] .= J_bc′[1:length(resida), 1:N]
            idxᵢ = (length(resida) + 1):(length(resida) + length(residb))
            J_bc[idxᵢ, (end - 2N + 1):(end - N)] .= J_bc′[idxᵢ, (end - N + 1):end]

            return nothing
        end
    else
        @views function jac_mp!(J::AbstractMatrix, us, p, resid_bc,
            resid_nodes::MaybeDiffCache, ode_jac_cache, bc_jac_cache, ode_fn, bc_fn,
            cur_nshoot, nodes)
            # This is mostly a safety measure
            fill!(J, 0)

            J_bc = J[1:N, :]
            J_c = J[(N + 1):end, :]

            sparse_jacobian!(J_c, alg.jac_alg.nonbc_diffmode, ode_jac_cache, ode_fn,
                resid_nodes.du, us)

            # For BC
            sparse_jacobian!(J_bc, alg.jac_alg.bc_diffmode, bc_jac_cache, bc_fn, resid_bc,
                us)

            return nothing
        end
    end

    # This gets all the nshoots except the final SingleShooting case
    all_nshoots = get_all_nshoots(alg.grid_coarsening, nshoots)
    u_at_nodes, nodes = similar(u0, 0), typeof(first(tspan))[]

    for (i, cur_nshoot) in enumerate(all_nshoots)
        if i == 1
            nodes, u_at_nodes = multiple_shooting_initialize(prob, alg, ig, nshoots;
                odesolve_kwargs, verbose, kwargs...)
        else
            nodes, u_at_nodes = multiple_shooting_initialize(u_at_nodes, prob, alg, nodes,
                cur_nshoot, all_nshoots[i - 1]::Int, ig; odesolve_kwargs, verbose,
                kwargs...)
        end

        resid_prototype = ArrayPartition(bcresid_prototype,
            similar(u_at_nodes, cur_nshoot * N))
        resid_nodes = __maybe_allocate_diffcache(resid_prototype.x[2],
            pickchunksize((cur_nshoot + 1) * N), alg.jac_alg.bc_diffmode)

        if alg.jac_alg.nonbc_diffmode isa AbstractSparseADType ||
           alg.jac_alg.bc_diffmode isa AbstractSparseADType
            J_full, J_c, J_bc = __generate_sparse_jacobian_prototype(alg, prob.problem_type,
                bcresid_prototype, u0, N, cur_nshoot)
        end

        ode_fn = (du, u) -> solve_internal_odes!(du, u, prob.p, cur_nshoot, nodes)
        sd_ode = alg.jac_alg.nonbc_diffmode isa AbstractSparseADType ?
                 PrecomputedJacobianColorvec(J_c) : NoSparsityDetection()
        ode_jac_cache = sparse_jacobian_cache(alg.jac_alg.nonbc_diffmode, sd_ode,
            ode_fn, similar(u_at_nodes, cur_nshoot * N), u_at_nodes)

        bc_fn = (du, u) -> compute_bc_residual!(du, u, prob.p, cur_nshoot, nodes,
            resid_nodes)
        if prob.problem_type isa TwoPointBVProblem
            sd_bc = alg.jac_alg.bc_diffmode isa AbstractSparseADType ?
                    PrecomputedJacobianColorvec(J_bc) : NoSparsityDetection()
            bc_jac_cache_partial = sparse_jacobian_cache(alg.jac_alg.bc_diffmode, sd_bc,
                bc_fn, similar(bcresid_prototype),
                ArrayPartition(@view(u_at_nodes[1:N]),
                    @view(u_at_nodes[(end - N + 1):end])))

            bc_jac_cache = (bc_jac_cache_partial, init_jacobian(bc_jac_cache_partial))

            jac_prototype = if @isdefined(J_full)
                J_full
            else
                __zeros_like(u_at_nodes, length(resid_prototype), length(u_at_nodes))
            end
        else
            sd_bc = alg.jac_alg.bc_diffmode isa AbstractSparseADType ?
                    SymbolicsSparsityDetection() : NoSparsityDetection()
            bc_jac_cache = sparse_jacobian_cache(alg.jac_alg.bc_diffmode,
                sd_bc, bc_fn, similar(bcresid_prototype), u_at_nodes)

            jac_prototype = vcat(init_jacobian(bc_jac_cache), init_jacobian(ode_jac_cache))
        end

        jac_fn = (J, us, p) -> jac!(J, us, p, similar(bcresid_prototype), resid_nodes,
            ode_jac_cache, bc_jac_cache, ode_fn, bc_fn, cur_nshoot, nodes)

        loss_function! = NonlinearFunction{true}((args...) -> loss!(args..., cur_nshoot,
                nodes); resid_prototype, jac = jac_fn, jac_prototype)
        nlprob = NonlinearProblem(loss_function!, u_at_nodes, prob.p)
        sol_nlsolve = __solve(nlprob, alg.nlsolve; nlsolve_kwargs..., verbose, kwargs...)
        # u_at_nodes = sol_nlsolve.u
    end

    single_shooting_prob = remake(prob; u0 = reshape(u_at_nodes[1:N], u0_size))
    return __solve(single_shooting_prob, Shooting(alg.ode_alg; alg.nlsolve);
        odesolve_kwargs, nlsolve_kwargs, verbose, kwargs...)
end

@views function multiple_shooting_initialize(prob, alg::MultipleShooting, ::True,
    nshoots; odesolve_kwargs = (;), verbose = true, kwargs...)
    @unpack f, u0, tspan, p = prob
    @unpack ode_alg = alg

    nodes = range(tspan[1], tspan[2]; length = nshoots + 1)
    N = length(first(u0))

    u_at_nodes = similar(first(u0), (nshoots + 1) * N)
    recursive_flatten!(u_at_nodes, u0)
    return nodes, u_at_nodes
end

@views function multiple_shooting_initialize(prob, alg::MultipleShooting, ::False,
    nshoots; odesolve_kwargs = (;), verbose = true, kwargs...)
    @unpack f, u0, tspan, p = prob
    @unpack ode_alg = alg

    nodes = range(tspan[1], tspan[2]; length = nshoots + 1)
    N = length(u0)

    # Ensures type stability in case the parameters are dual numbers
    if !(typeof(p) <: SciMLBase.NullParameters)
        if !isconcretetype(eltype(p)) && verbose
            @warn "Type inference will fail if eltype(p) is not a concrete type"
        end
        u_at_nodes = similar(u0, promote_type(eltype(u0), eltype(p)), (nshoots + 1) * N)
    else
        u_at_nodes = similar(u0, (nshoots + 1) * N)
    end

    # Assumes no initial guess for now
    start_prob = ODEProblem{isinplace(prob)}(f, u0, tspan, p)
    sol = __solve(start_prob, ode_alg; odesolve_kwargs..., verbose, kwargs...,
        saveat = nodes)

    if SciMLBase.successful_retcode(sol)
        u_at_nodes[1:N] .= vec(sol.u[1])
        for i in 2:(nshoots + 1)
            u_at_nodes[(N + (i - 2) * N) .+ (1:N)] .= vec(sol.u[i])
        end
    else
        @warn "Initialization using odesolve failed. Initializing using 0s. It is \
               recommended to provide an `initial_guess` function in this case."
        fill!(u_at_nodes, 0)
    end

    return nodes, u_at_nodes
end

@views function multiple_shooting_initialize(u_at_nodes_prev, prob, alg, prev_nodes,
    nshoots, old_nshoots, ig; odesolve_kwargs = (;), kwargs...)
    @unpack f, u0, tspan, p = prob
    nodes = range(tspan[1], tspan[2]; length = nshoots + 1)
    N = known(ig) ? length(first(u0)) : length(u0)

    u_at_nodes = similar(u_at_nodes_prev, N + nshoots * N)
    u_at_nodes[1:N] .= u_at_nodes_prev[1:N]
    u_at_nodes[(end - N + 1):end] .= u_at_nodes_prev[(end - N + 1):end]

    skipsize = old_nshoots / nshoots
    for i in 2:nshoots
        pos = skipsize * (i - 1) + 1
        idxs = (N + (i - 2) * N) .+ (1:N)
        if isinteger(pos)
            # If the current node is also a node of the finer grid
            ind = trunc(Int, pos)
            idxs_prev = (N + (ind - 2) * N .+ (1:N))
            u_at_nodes[idxs] .= u_at_nodes_prev[idxs_prev]
        else
            # If the current node is not a node of the finer grid simulate from closest
            # previous node and take result from simulation
            fpos = floor(Int, pos)
            r = pos - fpos

            t0 = prev_nodes[fpos]
            tf = prev_nodes[fpos + 1]
            tstop = t0 + r * (tf - t0)

            idxs_prev = (N + (fpos - 2) * N .+ (1:N))
            ustart = u_at_nodes_prev[idxs_prev]

            odeprob = ODEProblem(f, ustart, (t0, tstop), p)
            odesol = __solve(odeprob, alg.ode_alg; odesolve_kwargs..., kwargs...,
                saveat = (), save_end = true)

            u_at_nodes[idxs] .= odesol.u[end]
        end
    end

    return nodes, u_at_nodes
end

@inline function get_all_nshoots(grid_coarsening, nshoots)
    if grid_coarsening isa Bool
        !grid_coarsening && return [nshoots]
        update_fn = Base.Fix2(÷, 2)
    elseif grid_coarsening isa Function
        update_fn = grid_coarsening
    else
        grid_coarsening[1] == nshoots && return grid_coarsening
        return vcat(nshoots, grid_coarsening)
    end
    nshoots_vec = Int[nshoots]
    next = update_fn(nshoots)
    while next > 1
        push!(nshoots_vec, next)
        next = update_fn(last(nshoots_vec))
    end
    @assert !(1 in nshoots_vec)
    return nshoots_vec
end
