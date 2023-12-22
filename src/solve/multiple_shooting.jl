function __solve(prob::BVProblem, _alg::MultipleShooting; odesolve_kwargs = (;),
        nlsolve_kwargs = (;), ensemblealg = EnsembleThreads(), verbose = true, kwargs...)
    (; f, tspan) = prob

    if !(ensemblealg isa EnsembleSerial) && !(ensemblealg isa EnsembleThreads)
        throw(ArgumentError("Currently MultipleShooting only supports `EnsembleSerial` and \
                             `EnsembleThreads`!"))
    end

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
        M = resida_len + residb_len
    else
        M = length(bcresid_prototype)
    end

    internal_ode_kwargs = (; verbose, kwargs..., odesolve_kwargs..., save_end = true)

    solve_internal_odes! = @closure (resid_nodes, us, p, cur_nshoot, nodes, odecache) -> __multiple_shooting_solve_internal_odes!(resid_nodes, us, cur_nshoot,
        odecache, nodes, u0_size, N, ensemblealg, tspan)

    # This gets all the nshoots except the final SingleShooting case
    all_nshoots = __get_all_nshoots(alg.grid_coarsening, nshoots)
    u_at_nodes, nodes = similar(u0, 0), typeof(first(tspan))[]

    ode_cache_loss_fn = __multiple_shooting_init_odecache(ensemblealg, prob,
        alg.ode_alg, u0, maximum(all_nshoots); internal_ode_kwargs...)

    for (i, cur_nshoot) in enumerate(all_nshoots)
        if i == 1
            u_at_nodes = __multiple_shooting_initialize!(nodes, prob, alg, ig, nshoots,
                ode_cache_loss_fn; kwargs..., verbose, odesolve_kwargs...)
        else
            u_at_nodes = __multiple_shooting_initialize!(nodes, u_at_nodes, prob, alg,
                cur_nshoot, all_nshoots[i - 1], ig, ode_cache_loss_fn, u0; kwargs...,
                verbose, odesolve_kwargs...)
        end

        if prob.problem_type isa TwoPointBVProblem
            __solve_nlproblem!(prob.problem_type, alg, bcresid_prototype, u_at_nodes, nodes,
                cur_nshoot, M, N, resida_len, residb_len, solve_internal_odes!, bc[1],
                bc[2], prob, u0, ode_cache_loss_fn, ensemblealg, internal_ode_kwargs;
                verbose, kwargs..., nlsolve_kwargs...)
        else
            __solve_nlproblem!(prob.problem_type, alg, bcresid_prototype, u_at_nodes, nodes,
                cur_nshoot, M, N, prod(resid_size), solve_internal_odes!, bc, prob, f,
                u0_size, u0, ode_cache_loss_fn, ensemblealg, internal_ode_kwargs; verbose,
                kwargs..., nlsolve_kwargs...)
        end
    end

    if prob.problem_type isa TwoPointBVProblem
        diffmode_shooting = __get_non_sparse_ad(alg.jac_alg.diffmode)
    else
        diffmode_shooting = __get_non_sparse_ad(alg.jac_alg.bc_diffmode)
    end
    shooting_alg = Shooting(alg.ode_alg, alg.nlsolve,
        BVPJacobianAlgorithm(diffmode_shooting))

    single_shooting_prob = remake(prob; u0 = reshape(u_at_nodes[1:N], u0_size))
    return __solve(single_shooting_prob, shooting_alg; odesolve_kwargs, nlsolve_kwargs,
        verbose, kwargs...)
end

# TODO: We can save even more memory by hoisting the preallocated caches for the ODEs
# TODO: out of the `__solve_nlproblem!` function and into the `__solve` function.
# TODO: But we can do it another day. Currently the gains here are quite high to justify
# TODO: waiting.

function __solve_nlproblem!(::TwoPointBVProblem, alg::MultipleShooting, bcresid_prototype,
        u_at_nodes, nodes, cur_nshoot::Int, M::Int, N::Int, resida_len::Int,
        residb_len::Int, solve_internal_odes!::S, bca::B1, bcb::B2, prob, u0,
        ode_cache_loss_fn, ensemblealg, internal_ode_kwargs; kwargs...) where {B1, B2, S}
    if __any_sparse_ad(alg.jac_alg)
        J_proto = __generate_sparse_jacobian_prototype(alg, prob.problem_type,
            bcresid_prototype, u0, N, cur_nshoot)
    end

    resid_prototype = vcat(bcresid_prototype[1],
        similar(u_at_nodes, cur_nshoot * N), bcresid_prototype[2])

    loss_fn = @closure (du, u, p) -> __multiple_shooting_2point_loss!(du, u, p, cur_nshoot,
        nodes, prob, solve_internal_odes!, resida_len, residb_len, N, bca, bcb,
        ode_cache_loss_fn)

    sd_bvp = alg.jac_alg.diffmode isa AbstractSparseADType ?
             __sparsity_detection_alg(J_proto) : NoSparsityDetection()

    resid_prototype_cached = similar(resid_prototype)
    jac_cache = sparse_jacobian_cache(alg.jac_alg.diffmode, sd_bvp, nothing,
        resid_prototype_cached, u_at_nodes)
    jac_prototype = init_jacobian(jac_cache)

    ode_cache_jac_fn = __multiple_shooting_init_jacobian_odecache(ensemblealg, prob,
        jac_cache, alg.jac_alg.diffmode, alg.ode_alg, cur_nshoot, u0;
        internal_ode_kwargs...)

    loss_fnₚ = @closure (du, u) -> __multiple_shooting_2point_loss!(du, u, prob.p,
        cur_nshoot, nodes, prob, solve_internal_odes!, resida_len, residb_len, N, bca, bcb,
        ode_cache_jac_fn)

    jac_fn = @closure (J, u, p) -> __multiple_shooting_2point_jacobian!(J, u, p, jac_cache,
        loss_fnₚ, resid_prototype_cached, alg)

    loss_function! = __unsafe_nonlinearfunction{true}(loss_fn; resid_prototype,
        jac = jac_fn, jac_prototype)

    # NOTE: u_at_nodes is updated inplace
    nlprob = __internal_nlsolve_problem(prob, M, N, loss_function!, u_at_nodes, prob.p)
    __solve(nlprob, alg.nlsolve; kwargs..., alias_u0 = true)

    return nothing
end

function __solve_nlproblem!(::StandardBVProblem, alg::MultipleShooting, bcresid_prototype,
        u_at_nodes, nodes, cur_nshoot::Int, M::Int, N::Int, resid_len::Int,
        solve_internal_odes!::S, bc::BC, prob, f::F, u0_size, u0, ode_cache_loss_fn,
        ensemblealg, internal_ode_kwargs; kwargs...) where {BC, F, S}
    if __any_sparse_ad(alg.jac_alg)
        J_proto = __generate_sparse_jacobian_prototype(alg, prob.problem_type,
            bcresid_prototype, u0, N, cur_nshoot)
    end
    resid_prototype = vcat(bcresid_prototype, similar(u_at_nodes, cur_nshoot * N))

    __resid_nodes = resid_prototype[(end - cur_nshoot * N + 1):end]
    resid_nodes = __maybe_allocate_diffcache(__resid_nodes,
        pickchunksize((cur_nshoot + 1) * N), alg.jac_alg.bc_diffmode)

    loss_fn = @closure (du, u, p) -> __multiple_shooting_mpoint_loss!(du, u, p, cur_nshoot,
        nodes, prob, solve_internal_odes!, resid_len, N, f, bc, u0_size, prob.tspan,
        alg.ode_alg, u0, ode_cache_loss_fn)

    # ODE Part
    sd_ode = alg.jac_alg.nonbc_diffmode isa AbstractSparseADType ?
             __sparsity_detection_alg(J_proto) : NoSparsityDetection()
    ode_jac_cache = sparse_jacobian_cache(alg.jac_alg.nonbc_diffmode, sd_ode,
        nothing, similar(u_at_nodes, cur_nshoot * N), u_at_nodes)
    ode_cache_ode_jac_fn = __multiple_shooting_init_jacobian_odecache(ensemblealg, prob,
        ode_jac_cache, alg.jac_alg.nonbc_diffmode, alg.ode_alg, cur_nshoot, u0;
        internal_ode_kwargs...)

    # BC Part
    sd_bc = alg.jac_alg.bc_diffmode isa AbstractSparseADType ?
            SymbolicsSparsityDetection() : NoSparsityDetection()
    bc_jac_cache = sparse_jacobian_cache(alg.jac_alg.bc_diffmode,
        sd_bc, nothing, similar(bcresid_prototype), u_at_nodes)
    ode_cache_bc_jac_fn = __multiple_shooting_init_jacobian_odecache(ensemblealg, prob,
        bc_jac_cache, alg.jac_alg.bc_diffmode, alg.ode_alg, cur_nshoot, u0;
        internal_ode_kwargs...)

    jac_prototype = vcat(init_jacobian(bc_jac_cache), init_jacobian(ode_jac_cache))

    # Define the functions now
    ode_fn = @closure (du, u) -> solve_internal_odes!(du, u, prob.p, cur_nshoot, nodes,
        ode_cache_ode_jac_fn)
    bc_fn = @closure (du, u) -> __multiple_shooting_mpoint_loss_bc!(du, u, prob.p,
        cur_nshoot, nodes, prob, solve_internal_odes!, N, f, bc, u0_size, prob.tspan,
        alg.ode_alg, u0, ode_cache_bc_jac_fn)

    jac_fn = @closure (J, u, p) -> __multiple_shooting_mpoint_jacobian!(J, u, p,
        similar(bcresid_prototype), resid_nodes, ode_jac_cache, bc_jac_cache,
        ode_fn, bc_fn, alg, N, M)

    loss_function! = __unsafe_nonlinearfunction{true}(loss_fn; resid_prototype,
        jac_prototype, jac = jac_fn)

    # NOTE: u_at_nodes is updated inplace
    nlprob = __internal_nlsolve_problem(prob, M, N, loss_function!, u_at_nodes, prob.p)
    __solve(nlprob, alg.nlsolve; kwargs..., alias_u0 = true)

    return nothing
end

function __multiple_shooting_init_odecache(::EnsembleSerial, prob, alg, u0, nshoots;
        kwargs...)
    odeprob = ODEProblem{isinplace(prob)}(prob.f, u0, prob.tspan, prob.p)
    return SciMLBase.__init(odeprob, alg; kwargs...)
end

function __multiple_shooting_init_odecache(::EnsembleThreads, prob, alg, u0, nshoots;
        kwargs...)
    odeprob = ODEProblem{isinplace(prob)}(prob.f, u0, prob.tspan, prob.p)
    return [SciMLBase.__init(odeprob, alg; kwargs...)
            for _ in 1:min(Threads.nthreads(), nshoots)]
end

function __multiple_shooting_init_jacobian_odecache(ensemblealg, prob, jac_cache, ad, alg,
        nshoots, u; kwargs...)
    return __multiple_shooting_init_odecache(ensemblealg, prob, alg, u, nshoots;
        kwargs...)
end

function __multiple_shooting_init_jacobian_odecache(ensemblealg, prob, jac_cache,
        ::Union{AutoForwardDiff, AutoSparseForwardDiff}, alg, nshoots, u; kwargs...)
    cache = jac_cache.cache
    if cache isa ForwardDiff.JacobianConfig
        xduals = reshape(cache.duals[2][1:length(u)], size(u))
    else
        xduals = reshape(cache.t[1:length(u)], size(u))
    end
    fill!(xduals, 0)
    return __multiple_shooting_init_odecache(ensemblealg, prob, alg, xduals, nshoots;
        kwargs...)
end

# Not using `EnsembleProblem` since it is hard to initialize the cache and stuff
function __multiple_shooting_solve_internal_odes!(resid_nodes, us, cur_nshoots::Int,
        odecache, nodes, u0_size, N::Int, ::EnsembleSerial, tspan)
    ts_ = Vector{Vector{typeof(first(tspan))}}(undef, cur_nshoots)
    us_ = Vector{Vector{typeof(us)}}(undef, cur_nshoots)

    for i in 1:cur_nshoots
        SciMLBase.reinit!(odecache, reshape(@view(us[((i - 1) * N + 1):(i * N)]), u0_size);
            t0 = nodes[i], tf = nodes[i + 1])
        sol = solve!(odecache)
        us_[i] = deepcopy(sol.u)
        ts_[i] = deepcopy(sol.t)
        resid_nodes[((i - 1) * N + 1):(i * N)] .= @view(us[(i * N + 1):((i + 1) * N)]) .-
                                                  vec(sol.u[end])
    end

    return reduce(vcat, us_), reduce(vcat, ts_)
end

function __multiple_shooting_solve_internal_odes!(resid_nodes, us, cur_nshoots::Int,
        odecache::Vector, nodes, u0_size, N::Int, ::EnsembleThreads, tspan)
    ts_ = Vector{Vector{typeof(first(tspan))}}(undef, cur_nshoots)
    us_ = Vector{Vector{typeof(us)}}(undef, cur_nshoots)

    n_splits = min(cur_nshoots, Threads.nthreads())
    n_per_chunk, n_remaining = divrem(cur_nshoots, n_splits)
    data_partition = map(1:n_splits) do i
        first = 1 + (i - 1) * n_per_chunk + ifelse(i ≤ n_remaining, i - 1, n_remaining)
        last = (first - 1) + n_per_chunk + ifelse(i <= n_remaining, 1, 0)
        return first:1:last
    end

    Threads.@threads for idx in 1:length(data_partition)
        cache = odecache[idx]
        for i in data_partition[idx]
            SciMLBase.reinit!(cache, reshape(@view(us[((i - 1) * N + 1):(i * N)]), u0_size);
                t0 = nodes[i], tf = nodes[i + 1])
            sol = solve!(cache)
            us_[i] = deepcopy(sol.u)
            ts_[i] = deepcopy(sol.t)
            resid_nodes[((i - 1) * N + 1):(i * N)] .= @view(us[(i * N + 1):((i + 1) * N)]) .-
                                                      vec(sol.u[end])
        end
    end

    return reduce(vcat, us_), reduce(vcat, ts_)
end

function __multiple_shooting_2point_jacobian!(J, us, p, jac_cache, loss_fn::F, resid,
        alg::MultipleShooting) where {F}
    sparse_jacobian!(J, alg.jac_alg.diffmode, jac_cache, loss_fn, resid, us)
    return nothing
end

function __multiple_shooting_mpoint_jacobian!(J, us, p, resid_bc, resid_nodes,
        ode_jac_cache, bc_jac_cache, ode_fn::F1, bc_fn::F2, alg::MultipleShooting,
        N::Int, M::Int) where {F1, F2}
    J_bc = @view(J[1:M, :])
    J_c = @view(J[(M + 1):end, :])

    sparse_jacobian!(J_c, alg.jac_alg.nonbc_diffmode, ode_jac_cache, ode_fn,
        resid_nodes.du, us)
    sparse_jacobian!(J_bc, alg.jac_alg.bc_diffmode, bc_jac_cache, bc_fn, resid_bc, us)

    return nothing
end

@views function __multiple_shooting_2point_loss!(resid, us, p, cur_nshoots::Int, nodes,
        prob, solve_internal_odes!::S, resida_len, residb_len, N, bca::BCA, bcb::BCB,
        ode_cache) where {S, BCA, BCB}
    resid_ = resid[(resida_len + 1):(end - residb_len)]
    solve_internal_odes!(resid_, us, p, cur_nshoots, nodes, ode_cache)

    resid_bc_a = resid[1:resida_len]
    resid_bc_b = resid[(end - residb_len + 1):end]

    ua = us[1:N]
    ub = us[(end - N + 1):end]

    if isinplace(prob)
        bca(resid_bc_a, ua, p)
        bcb(resid_bc_b, ub, p)
    else
        resid_bc_a .= bca(ua, p)
        resid_bc_b .= bcb(ub, p)
    end

    return nothing
end

@views function __multiple_shooting_mpoint_loss_bc!(resid_bc, us, p, cur_nshoots::Int,
        nodes, prob, solve_internal_odes!::S, N, f::F, bc::BC, u0_size, tspan,
        ode_alg, u0, ode_cache) where {S, F, BC}
    iip = isinplace(prob)
    _resid_nodes = similar(us, cur_nshoots * N)

    # NOTE: We need to recompute this to correctly propagate the dual numbers / gradients
    _us, _ts = solve_internal_odes!(_resid_nodes, us, p, cur_nshoots, nodes, ode_cache)

    odeprob = ODEProblem{iip}(f, u0, tspan, p)
    total_solution = SciMLBase.build_solution(odeprob, ode_alg, _ts, _us)

    if iip
        eval_bc_residual!(resid_bc, StandardBVProblem(), bc, total_solution, p)
    else
        resid_bc .= eval_bc_residual(StandardBVProblem(), bc, total_solution, p)
    end

    return nothing
end

@views function __multiple_shooting_mpoint_loss!(resid, us, p, cur_nshoots::Int, nodes,
        prob, solve_internal_odes!::S, resid_len, N, f::F, bc::BC, u0_size, tspan,
        ode_alg, u0, ode_cache) where {S, F, BC}
    iip = isinplace(prob)
    resid_bc = resid[1:resid_len]
    resid_nodes = resid[(resid_len + 1):end]

    _us, _ts = solve_internal_odes!(resid_nodes, us, p, cur_nshoots, nodes, ode_cache)

    odeprob = ODEProblem{iip}(f, u0, tspan, p)
    total_solution = SciMLBase.build_solution(odeprob, ode_alg, _ts, _us)

    if iip
        eval_bc_residual!(resid_bc, StandardBVProblem(), bc, total_solution, p)
    else
        resid_bc .= eval_bc_residual(StandardBVProblem(), bc, total_solution, p)
    end

    return nothing
end

# Problem has initial guess
@views function __multiple_shooting_initialize!(nodes, prob, alg, ::Val{true}, nshoots::Int,
        odecache; kwargs...)
    @unpack u0, tspan = prob

    resize!(nodes, nshoots + 1)
    nodes .= range(tspan[1], tspan[2]; length = nshoots + 1)

    # NOTE: We don't check `u0 isa Function` since `u0` in-principle can be a callable
    #       struct
    u0_ = u0 isa VectorOfArray ? u0 : [__initial_guess(u0, prob.p, t) for t in nodes]

    N = length(first(u0_))
    u_at_nodes = similar(first(u0_), (nshoots + 1) * N)
    recursive_flatten!(u_at_nodes, u0_)

    return u_at_nodes
end

# No initial guess
@views function __multiple_shooting_initialize!(nodes, prob, alg::MultipleShooting,
        ::Val{false}, nshoots::Int, odecache_; verbose, kwargs...)
    @unpack f, u0, tspan, p = prob
    @unpack ode_alg = alg

    resize!(nodes, nshoots + 1)
    nodes .= range(tspan[1], tspan[2]; length = nshoots + 1)
    N = length(u0)

    # Ensures type stability in case the parameters are dual numbers
    if !(p isa SciMLBase.NullParameters)
        if !isconcretetype(eltype(p)) && verbose
            @warn "Type inference will fail if eltype(p) is not a concrete type"
        end
        u_at_nodes = similar(u0, promote_type(eltype(u0), eltype(p)), (nshoots + 1) * N)
    else
        u_at_nodes = similar(u0, (nshoots + 1) * N)
    end

    # Assumes no initial guess for now
    odecache = odecache_ isa Vector ? first(odecache_) : odecache_
    SciMLBase.reinit!(odecache, u0; t0 = tspan[1], tf = tspan[2])
    sol = solve!(odecache)

    if SciMLBase.successful_retcode(sol)
        for i in 1:length(nodes)
            u_at_nodes[(i - 1) * N .+ (1:N)] .= vec(sol(nodes[i]))
        end
    else
        @warn "Initialization using odesolve failed. Initializing using 0s. It is \
               recommended to provide an initial guess function via \
               `u0 = <function>(p, t)` in this case."
        fill!(u_at_nodes, 0)
    end

    return u_at_nodes
end

# Grid coarsening
@views function __multiple_shooting_initialize!(nodes, u_at_nodes_prev, prob, alg,
        nshoots, old_nshoots, ig, odecache_, u0; kwargs...)
    @unpack f, tspan, p = prob
    prev_nodes = copy(nodes)
    odecache = odecache_ isa Vector ? first(odecache_) : odecache_

    resize!(nodes, nshoots + 1)
    nodes .= range(tspan[1], tspan[2]; length = nshoots + 1)
    N = length(u0)

    u_at_nodes = similar(u0, N + nshoots * N)
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
            # TODO: Batch this computation and do it for all points between two nodes
            # TODO: Though it is unlikely that this will be a bottleneck
            # If the current node is not a node of the finer grid simulate from closest
            # previous node and take result from simulation
            fpos = floor(Int, pos)
            r = pos - fpos

            t0 = prev_nodes[fpos]
            tf = prev_nodes[fpos + 1]
            tstop = t0 + r * (tf - t0)

            idxs_prev = (N + (fpos - 2) * N .+ (1:N))
            ustart = u_at_nodes_prev[idxs_prev]

            SciMLBase.reinit!(odecache, ustart; t0, tf = tstop)
            odesol = solve!(odecache)

            u_at_nodes[idxs] .= odesol.u[end]
        end
    end

    return u_at_nodes
end

@inline function __get_all_nshoots(g::Bool, nshoots)
    return g ? __get_all_nshoots(Base.Fix2(÷, 2), nshoots) : [nshoots]
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
