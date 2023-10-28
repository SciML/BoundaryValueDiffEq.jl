function __solve(prob::BVProblem, _alg::MultipleShooting; odesolve_kwargs = (;),
    nlsolve_kwargs = (;), ensemblealg = EnsembleThreads(), verbose = true, kwargs...)
    @unpack f, tspan = prob

    ig, T, N, Nig, u0 = __extract_problem_details(prob; dt = 0.1)
    has_initial_guess = _unwrap_val(ig)

    bcresid_prototype, resid_size = __get_bcresid_prototype(prob, u0)
    iip, bc, u0, u0_size = isinplace(prob), prob.f.bc, deepcopy(u0), size(u0)

    __alg = concretize_jacobian_algorithm(_alg, prob)
    alg = if has_initial_guess && Nig != __alg.nshoots
        verbose &&
            @warn "Initial guess length != `nshoots + 1`! Adapting to `nshoots = $(Nig)`"
        update_nshoots(__alg, Nig)
    else
        __alg
    end
    nshoots = alg.nshoots

    if prob.problem_type isa TwoPointBVProblem
        resida_len = prod(resid_size[1])
        residb_len = prod(resid_size[2])
    end

    internal_ode_kwargs = (; verbose, kwargs..., odesolve_kwargs..., save_end = true)
    solve_internal_odes! = (resid_nodes, us, p, cur_nshoot, nodes) -> __multiple_shooting_solve_internal_odes!(resid_nodes,
        us, p, Val(iip), f, cur_nshoot, nodes, tspan, u0_size, N, alg, ensemblealg,
        internal_ode_kwargs)

    # This gets all the nshoots except the final SingleShooting case
    all_nshoots = __get_all_nshoots(alg.grid_coarsening, nshoots)
    u_at_nodes, nodes = similar(u0, 0), typeof(first(tspan))[]

    for (i, cur_nshoot) in enumerate(all_nshoots)
        if i == 1
            u_at_nodes = __multiple_shooting_initialize!(nodes, prob, alg, ig, nshoots;
                kwargs..., verbose, odesolve_kwargs...)
        else
            u_at_nodes = __multiple_shooting_initialize!(nodes, u_at_nodes, prob, alg,
                cur_nshoot, all_nshoots[i - 1], ig; kwargs..., verbose, odesolve_kwargs...)
        end

        if __any_sparse_ad(alg.jac_alg)
            J_proto = __generate_sparse_jacobian_prototype(alg, prob.problem_type,
                bcresid_prototype, u0, N, cur_nshoot)
        end

        if prob.problem_type isa TwoPointBVProblem
            resid_prototype = vcat(bcresid_prototype[1],
                similar(u_at_nodes, cur_nshoot * N), bcresid_prototype[2])

            __resid_nodes = resid_prototype[(resida_len + 1):(resida_len + cur_nshoot * N)]
            resid_nodes = __maybe_allocate_diffcache(__resid_nodes,
                pickchunksize((cur_nshoot + 1) * N), alg.jac_alg.diffmode)

            loss_fn = (du, u, p) -> __multiple_shooting_2point_loss!(du, u, p, cur_nshoot,
                nodes, Val(iip), solve_internal_odes!, resida_len, residb_len, N, bc[1],
                bc[2])
            loss_fnₚ = (du, u) -> loss_fn(du, u, prob.p)

            sd_bvp = alg.jac_alg.diffmode isa AbstractSparseADType ?
                     __sparsity_detection_alg(J_proto) : NoSparsityDetection()

            resid_prototype_cached = similar(resid_prototype)
            jac_cache = sparse_jacobian_cache(alg.jac_alg.diffmode, sd_bvp, loss_fnₚ,
                resid_prototype_cached, u_at_nodes)
            jac_prototype = init_jacobian(jac_cache)

            jac_fn = (J, u, p) -> __multiple_shooting_2point_jacobian!(J, u, p, jac_cache,
                loss_fnₚ, resid_prototype_cached, alg)
        else
            resid_prototype = vcat(bcresid_prototype, similar(u_at_nodes, cur_nshoot * N))

            __resid_nodes = resid_prototype[(end - cur_nshoot * N + 1):end]
            resid_nodes = __maybe_allocate_diffcache(__resid_nodes,
                pickchunksize((cur_nshoot + 1) * N), alg.jac_alg.bc_diffmode)

            loss_fn = (du, u, p) -> __multiple_shooting_mpoint_loss!(du, u, p, cur_nshoot,
                nodes, Val(iip), solve_internal_odes!, prod(resid_size), N, f, bc, u0_size,
                tspan, alg.ode_alg)

            ode_fn = (du, u) -> solve_internal_odes!(du, u, prob.p, cur_nshoot, nodes)
            sd_ode = alg.jac_alg.nonbc_diffmode isa AbstractSparseADType ?
                     __sparsity_detection_alg(J_proto) : NoSparsityDetection()
            ode_jac_cache = sparse_jacobian_cache(alg.jac_alg.nonbc_diffmode, sd_ode,
                ode_fn, similar(u_at_nodes, cur_nshoot * N), u_at_nodes)

            bc_fn = (du, u) -> __multiple_shooting_mpoint_loss_bc!(du, u, prob.p,
                cur_nshoot, nodes, Val(iip), solve_internal_odes!, N, f, bc, u0_size, tspan,
                alg.ode_alg)
            sd_bc = alg.jac_alg.bc_diffmode isa AbstractSparseADType ?
                    SymbolicsSparsityDetection() : NoSparsityDetection()
            bc_jac_cache = sparse_jacobian_cache(alg.jac_alg.bc_diffmode,
                sd_bc, bc_fn, similar(bcresid_prototype), u_at_nodes)

            jac_prototype = vcat(init_jacobian(bc_jac_cache), init_jacobian(ode_jac_cache))

            jac_fn = (J, u, p) -> __multiple_shooting_mpoint_jacobian!(J, u, p,
                similar(bcresid_prototype), resid_nodes, ode_jac_cache, bc_jac_cache,
                ode_fn, bc_fn, alg, N)
        end
        loss_function! = NonlinearFunction{true}(loss_fn; resid_prototype, jac = jac_fn,
            jac_prototype)

        # NOTE: u_at_nodes is updated inplace
        nlprob = NonlinearProblem(loss_function!, u_at_nodes, prob.p)
        __solve(nlprob, alg.nlsolve; verbose, kwargs..., nlsolve_kwargs..., alias_u0 = true)
    end

    single_shooting_prob = remake(prob; u0 = reshape(u_at_nodes[1:N], u0_size))
    return __solve(single_shooting_prob, Shooting(alg.ode_alg; alg.nlsolve);
        odesolve_kwargs, nlsolve_kwargs, verbose, kwargs...)
end

function __multiple_shooting_solve_internal_odes!(resid_nodes, us, p, ::Val{iip}, f,
    cur_nshoots::Int, nodes, tspan, u0_size, N, alg::MultipleShooting,
    ensemblealg, kwargs) where {iip}
    ts_ = Vector{Vector{typeof(first(tspan))}}(undef, cur_nshoots)
    us_ = Vector{Vector{typeof(us)}}(undef, cur_nshoots)

    function prob_func(probᵢ, i, _)
        return remake(probᵢ; u0 = reshape(@view(us[((i - 1) * N + 1):(i * N)]), u0_size),
            tspan = (nodes[i], nodes[i + 1]))
    end

    function reduction(u, data, I)
        for i in I
            u.us[i] = data[i].u
            u.ts[i] = data[i].t
            u.resid[((i - 1) * N + 1):(i * N)] .= vec(@view(us[(i * N + 1):((i + 1) * N)])) .-
                                                  vec(data[i].u[end])
        end
        return (u, false)
    end

    odeprob = ODEProblem{iip}(f, reshape(@view(us[1:N]), u0_size), tspan, p)

    ensemble_prob = EnsembleProblem(odeprob; prob_func, reduction, safetycopy = false,
        u_init = (; us = us_, ts = ts_, resid = resid_nodes))
    ensemble_sol = __solve(ensemble_prob, alg.ode_alg, ensemblealg; kwargs...,
        trajectories = cur_nshoots)

    return reduce(vcat, ensemble_sol.u.us), reduce(vcat, ensemble_sol.u.ts)
end

function __multiple_shooting_2point_jacobian!(J, us, p, jac_cache, loss_fn, resid,
    alg::MultipleShooting)
    sparse_jacobian!(J, alg.jac_alg.diffmode, jac_cache, loss_fn, resid, us)
    return nothing
end

function __multiple_shooting_mpoint_jacobian!(J, us, p, resid_bc, resid_nodes,
    ode_jac_cache, bc_jac_cache, ode_fn, bc_fn, alg::MultipleShooting, N::Int)
    J_bc = @view(J[1:N, :])
    J_c = @view(J[(N + 1):end, :])

    sparse_jacobian!(J_c, alg.jac_alg.nonbc_diffmode, ode_jac_cache, ode_fn,
        resid_nodes.du, us)
    sparse_jacobian!(J_bc, alg.jac_alg.bc_diffmode, bc_jac_cache, bc_fn, resid_bc, us)

    return nothing
end

@views function __multiple_shooting_2point_loss!(resid, us, p, cur_nshoots::Int, nodes,
    ::Val{iip}, solve_internal_odes!, resida_len, residb_len, N, bca, bcb) where {iip}
    resid_ = resid[(resida_len + 1):(end - residb_len)]
    solve_internal_odes!(resid_, us, p, cur_nshoots, nodes)

    resid_bc_a = resid[1:resida_len]
    resid_bc_b = resid[(end - residb_len + 1):end]

    ua = us[1:N]
    ub = us[(end - N + 1):end]

    if iip
        bca(resid_bc_a, ua, p)
        bcb(resid_bc_b, ub, p)
    else
        resid_bc_a .= bca(ua, p)
        resid_bc_b .= bcb(ub, p)
    end

    return nothing
end

@views function __multiple_shooting_mpoint_loss_bc!(resid_bc, us, p, cur_nshoots::Int,
    nodes,
    ::Val{iip}, solve_internal_odes!, N, f, bc, u0_size, tspan, ode_alg) where {iip}
    _resid_nodes = similar(us, cur_nshoots * N)

    # NOTE: We need to recompute this to correctly propagate the dual numbers / gradients
    _us, _ts = solve_internal_odes!(_resid_nodes, us, p, cur_nshoots, nodes)

    odeprob = ODEProblem{iip}(f, reshape(us[1:N], u0_size), tspan, p)
    total_solution = SciMLBase.build_solution(odeprob, ode_alg, _ts, _us)

    if iip
        eval_bc_residual!(resid_bc, StandardBVProblem(), bc, total_solution, p)
    else
        resid_bc .= eval_bc_residual(StandardBVProblem(), bc, total_solution, p)
    end

    return nothing
end

@views function __multiple_shooting_mpoint_loss!(resid, us, p, cur_nshoots::Int, nodes,
    ::Val{iip}, solve_internal_odes!, resid_len, N, f, bc, u0_size, tspan,
    ode_alg) where {iip}
    resid_bc = resid[1:resid_len]
    resid_nodes = resid[(resid_len + 1):end]

    _us, _ts = solve_internal_odes!(resid_nodes, us, p, cur_nshoots, nodes)

    odeprob = ODEProblem{iip}(f, reshape(us[1:N], u0_size), tspan, p)
    total_solution = SciMLBase.build_solution(odeprob, ode_alg, _ts, _us)

    if iip
        eval_bc_residual!(resid_bc, StandardBVProblem(), bc, total_solution, p)
    else
        resid_bc .= eval_bc_residual(StandardBVProblem(), bc, total_solution, p)
    end

    return nothing
end

# Problem has initial guess
@views function __multiple_shooting_initialize!(nodes, prob, alg, ::Val{true}, nshoots;
    kwargs...)
    @unpack u0, tspan = prob

    resize!(nodes, nshoots + 1)
    nodes .= range(tspan[1], tspan[2]; length = nshoots + 1)

    N = length(first(u0))
    u_at_nodes = similar(first(u0), (nshoots + 1) * N)
    recursive_flatten!(u_at_nodes, u0)

    return u_at_nodes
end

# No initial guess
@views function __multiple_shooting_initialize!(nodes, prob, alg::MultipleShooting,
    ::Val{false}, nshoots; verbose, kwargs...)
    @unpack f, u0, tspan, p = prob
    @unpack ode_alg = alg

    resize!(nodes, nshoots + 1)
    nodes .= range(tspan[1], tspan[2]; length = nshoots + 1)
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
    sol = __solve(start_prob, ode_alg; verbose, kwargs..., saveat = nodes)

    if SciMLBase.successful_retcode(sol)
        u_at_nodes[1:N] .= vec(sol.u[1])
        for i in 2:(nshoots + 1)
            u_at_nodes[(N + (i - 2) * N) .+ (1:N)] .= vec(sol.u[i])
        end
    else
        @warn "Initialization using odesolve failed. Initializing using 0s. It is \
               recommended to provide an `initial_guess` in this case."
        fill!(u_at_nodes, 0)
    end

    return u_at_nodes
end

# Grid coarsening
@views function __multiple_shooting_initialize!(nodes, u_at_nodes_prev, prob, alg,
    nshoots, old_nshoots, ig; kwargs...)
    @unpack f, u0, tspan, p = prob
    prev_nodes = copy(nodes)

    resize!(nodes, nshoots + 1)
    nodes .= range(tspan[1], tspan[2]; length = nshoots + 1)
    N = _unwrap_val(ig) ? length(first(u0)) : length(u0)

    u_at_nodes = similar(_unwrap_val(ig) ? first(u0) : u0, N + nshoots * N)
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
            odesol = __solve(odeprob, alg.ode_alg; kwargs..., saveat = (), save_end = true)

            u_at_nodes[idxs] .= odesol.u[end]
        end
    end

    return u_at_nodes
end

@inline function __get_all_nshoots(g::Bool, nshoots)
    return g ? __get_all_nshoots(Base.Fix2(÷, 2)) : [nshoots]
end
@inline function __get_all_nshoots(g, nshoots)
    first(g) == nshoots && return g
    return vcat(nshoots, g)
end
@inline function __get_all_nshoots(update_fn::G, nshoots) where {G <: Function}
    nshoots_vec = Int[nshoots]
    next = update_fn(nshoots)
    while next > 1
        push!(nshoots_vec, next)
        next = update_fn(last(nshoots_vec))
    end
    @assert !(1 in nshoots_vec)
    return nshoots_vec
end
