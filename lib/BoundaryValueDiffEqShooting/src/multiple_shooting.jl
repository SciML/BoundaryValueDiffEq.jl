function SciMLBase.__solve(
        prob::BVProblem, _alg::MultipleShooting; abstol = 1.0e-6, odesolve_kwargs = (;),
        nlsolve_kwargs = (; abstol), optimize_kwargs = (; abstol),
        ensemblealg = EnsembleThreads(), verbose = true, kwargs...
    )
    if _alg.device_steps !== nothing || !(_alg.platform isa CPU)
        return __multiple_shooting_device_solve(
            prob, _alg; abstol, odesolve_kwargs, nlsolve_kwargs,
            optimize_kwargs, ensemblealg, verbose, kwargs...
        )
    end
    verbose_spec = _process_verbose_param(verbose)

    (; f, tspan) = prob

    if !(ensemblealg isa EnsembleSerial) && !(ensemblealg isa EnsembleThreads)
        throw(ArgumentError("Currently MultipleShooting only supports `EnsembleSerial` and \
                             `EnsembleThreads`!"))
    end

    ig, T, N, Nig, u0 = __extract_problem_details(prob; dt = 0.1 * oneunit(first(tspan)))
    has_initial_guess = _unwrap_val(ig)

    @assert u0 isa AbstractVector "Non-Vector Inputs for Multiple-Shooting hasn't been implemented yet!"

    bcresid_prototype, resid_size = __get_bcresid_prototype(prob, u0)
    iip, bc, u0, u0_size = isinplace(prob), prob.f.bc, deepcopy(u0), size(u0)
    @assert (iip || isnothing(_alg.optimize)) "Out-of-place constraints don't allow optimization solvers "

    __alg = concretize_jacobian_algorithm(_alg, prob)
    alg = if has_initial_guess && Nig != __alg.nshoots
        @SciMLMessage(
            "Initial guess length != `nshoots + 1`! Adapting to `nshoots = $(Nig)`",
            verbose_spec,
            :multiple_shooting_initial_guess
        )
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

    internal_ode_kwargs = (; kwargs..., odesolve_kwargs..., save_end = true)

    # This gets all the nshoots except the final SingleShooting case
    all_nshoots = __get_all_nshoots(alg.grid_coarsening, nshoots)
    u_at_nodes, nodes = similar(u0, 0), typeof(first(tspan))[]

    # Lazily builds an ODE integrator whose state element type matches `us`, for
    # AD evaluations that pass tagged states into the loss functions
    odecache_for_states = GeneralLazyBufferCache(
        @closure us -> __multiple_shooting_init_odecache(
            ensemblealg, prob, alg.ode_alg,
            copy(reshape(@view(us[1:N]), u0_size)), maximum(all_nshoots);
            internal_ode_kwargs...
        )
    )
    solve_internal_odes! = @closure (
        resid_nodes,
        us,
        p,
        cur_nshoot,
        nodes,
        odecache,
    ) -> __multiple_shooting_solve_internal_odes!(
        resid_nodes, us, cur_nshoot,
        __multiple_shooting_matching_odecache(odecache, odecache_for_states, us),
        nodes, u0_size, N, ensemblealg, tspan, alg.platform
    )

    ode_cache_loss_fn = __multiple_shooting_init_odecache(
        ensemblealg, prob, alg.ode_alg, u0, maximum(all_nshoots); internal_ode_kwargs...
    )

    for (i, cur_nshoot) in enumerate(all_nshoots)
        if i == 1
            u_at_nodes = __multiple_shooting_initialize!(
                nodes, prob, alg, ig, nshoots, ode_cache_loss_fn;
                kwargs..., verbose_spec, odesolve_kwargs...
            )
        else
            u_at_nodes = __multiple_shooting_initialize!(
                nodes, u_at_nodes, prob, alg, cur_nshoot, all_nshoots[i - 1], ig,
                ode_cache_loss_fn, u0; kwargs..., verbose_spec, odesolve_kwargs...
            )
        end

        if prob.problem_type isa TwoPointBVProblem
            __solve_nlproblem!(
                prob.problem_type, alg, bcresid_prototype, u_at_nodes, nodes,
                cur_nshoot, M, N, resida_len, residb_len, solve_internal_odes!, bc[1],
                bc[2], prob, u0, ode_cache_loss_fn, ensemblealg, internal_ode_kwargs;
                verbose, kwargs..., nlsolve_kwargs, optimize_kwargs...
            )
        else
            __solve_nlproblem!(
                prob.problem_type, alg, bcresid_prototype, u_at_nodes, nodes,
                cur_nshoot, M, N, prod(resid_size), solve_internal_odes!, bc, prob,
                f, u0_size, u0, ode_cache_loss_fn, ensemblealg, internal_ode_kwargs;
                verbose, kwargs..., nlsolve_kwargs, optimize_kwargs...
            )
        end
    end

    if prob.problem_type isa TwoPointBVProblem
        diffmode_shooting = __get_non_sparse_ad(alg.jac_alg.diffmode)
    else
        diffmode_shooting = __get_non_sparse_ad(alg.jac_alg.bc_diffmode)
    end

    shooting_alg = Shooting(alg.ode_alg, alg.nlsolve, alg.optimize, BVPJacobianAlgorithm(diffmode_shooting))

    single_shooting_prob = remake(prob; u0 = reshape(u_at_nodes[1:N], u0_size))
    return __solve(
        single_shooting_prob, shooting_alg; odesolve_kwargs,
        nlsolve_kwargs, optimize_kwargs, verbose, kwargs...
    )
end

# TODO: We can save even more memory by hoisting the preallocated caches for the ODEs
# TODO: out of the `__solve_nlproblem!` function and into the `__solve` function.
# TODO: But we can do it another day. Currently the gains here are quite high to justify
# TODO: waiting.

function __solve_nlproblem!(
        ::TwoPointBVProblem, alg::MultipleShooting, bcresid_prototype, u_at_nodes,
        nodes, cur_nshoot::Int, M::Int, N::Int, resida_len::Int, residb_len::Int,
        solve_internal_odes!::S, bca::B1, bcb::B2, prob, u0, ode_cache_loss_fn,
        ensemblealg, internal_ode_kwargs; kwargs...
    ) where {B1, B2, S}
    resid_prototype = vcat(bcresid_prototype[1], similar(u_at_nodes, cur_nshoot * N), bcresid_prototype[2])

    loss_fn = @closure (
        du,
        u,
        p,
    ) -> __multiple_shooting_2point_loss!(
        du, u, p, cur_nshoot, nodes, prob, solve_internal_odes!,
        resida_len, residb_len, N, bca, bcb, ode_cache_loss_fn
    )

    diffmode = if alg.jac_alg.diffmode isa AutoSparse
        sparse_jacobian_prototype = __generate_sparse_jacobian_prototype(
            alg, prob.problem_type, bcresid_prototype, u0, N, cur_nshoot
        )
        AutoSparse(
            get_dense_ad(alg.jac_alg.diffmode),
            sparsity_detector = ADTypes.KnownJacobianSparsityDetector(sparse_jacobian_prototype),
            coloring_algorithm = __default_coloring_algorithm(alg.jac_alg.diffmode)
        )
    else
        alg.jac_alg.diffmode
    end

    resid_prototype_cached = similar(resid_prototype)
    jac_cache = DI.prepare_jacobian(
        nothing, resid_prototype_cached, diffmode, u_at_nodes; strict = Val(false)
    )

    ode_cache_jac_fn = __multiple_shooting_init_jacobian_odecache(
        ensemblealg, prob, jac_cache, diffmode, alg.ode_alg,
        cur_nshoot, u0; internal_ode_kwargs...
    )

    loss_fnₚ = @closure (
        du,
        u,
    ) -> __multiple_shooting_2point_loss!(
        du, u, prob.p, cur_nshoot, nodes, prob, solve_internal_odes!,
        resida_len, residb_len, N, bca, bcb, ode_cache_jac_fn
    )
    jac_prototype = DI.jacobian(loss_fnₚ, resid_prototype, jac_cache, diffmode, u_at_nodes)

    jac_fn = @closure (
        J,
        u,
        p,
    ) -> __multiple_shooting_2point_jacobian!(
        J, u, p, jac_cache, loss_fnₚ, diffmode, resid_prototype_cached, alg
    )

    loss_function! = NonlinearFunction{true}(
        loss_fn; jac = jac_fn, resid_prototype,
        jac_prototype
    )

    # NOTE: u_at_nodes is updated inplace
    nlprob = __construct_internal_problem(
        prob, prob.problem_type, alg, loss_fn, jac_fn, jac_prototype,
        resid_prototype, u_at_nodes, prob.p, M, length(nodes), nothing, nothing
    )

    nlsolve_alg = __concrete_solve_algorithm(nlprob, alg.nlsolve, alg.optimize)
    # Extract verbose, nlsolve_kwargs, optimize_kwargs from merged kwargs
    verbose_val = get(kwargs, :verbose, true)
    verbose_spec = _process_verbose_param(verbose_val)
    nlsolve_kw = get(kwargs, :nlsolve_kwargs, (;))
    optimize_kw = get(kwargs, :optimize_kwargs, (;))

    # Construct kwargs with nonlinear verbosity
    concrete_kw = __concrete_kwargs(alg.nlsolve, alg.optimize, nlsolve_kw, optimize_kw, verbose_spec)

    # Filter out verbose, nlsolve_kwargs, optimize_kwargs from kwargs since they're handled
    other_kw = filter(kwargs) do (k, v)
        k ∉ (:verbose, :nlsolve_kwargs, :optimize_kwargs)
    end

    __internal_solve(nlprob, nlsolve_alg; concrete_kw..., other_kw...)

    return nothing
end

function __solve_nlproblem!(
        ::StandardBVProblem, alg::MultipleShooting, bcresid_prototype,
        u_at_nodes, nodes, cur_nshoot::Int, M::Int, N::Int, resid_len::Int,
        solve_internal_odes!::S, bc::BC, prob, f::F, u0_size, u0, ode_cache_loss_fn,
        ensemblealg, internal_ode_kwargs; kwargs...
    ) where {BC, F, S}
    resid_prototype = vcat(bcresid_prototype, similar(u_at_nodes, cur_nshoot * N))

    __resid_nodes = resid_prototype[(end - cur_nshoot * N + 1):end]
    resid_nodes = __maybe_allocate_diffcache(
        __resid_nodes, pickchunksize((cur_nshoot + 1) * N), alg.jac_alg.bc_diffmode
    )

    loss_fn = @closure (
        du,
        u,
        p,
    ) -> __multiple_shooting_mpoint_loss!(
        du, u, p, cur_nshoot, nodes, prob, solve_internal_odes!, resid_len,
        N, f, bc, u0_size, prob.tspan, alg.ode_alg, u0, ode_cache_loss_fn
    )

    # ODE Part
    nonbc_diffmode = if alg.jac_alg.nonbc_diffmode isa AutoSparse
        sparse_jacobian_prototype = __generate_sparse_jacobian_prototype(
            alg, prob.problem_type, bcresid_prototype, u0, N, cur_nshoot
        )
        AutoSparse(
            get_dense_ad(alg.jac_alg.nonbc_diffmode),
            sparsity_detector = ADTypes.KnownJacobianSparsityDetector(sparse_jacobian_prototype),
            coloring_algorithm = __default_coloring_algorithm(alg.jac_alg.nonbc_diffmode)
        )
    else
        alg.jac_alg.nonbc_diffmode
    end
    ode_jac_cache = DI.prepare_jacobian(
        nothing, similar(u_at_nodes, cur_nshoot * N),
        nonbc_diffmode, u_at_nodes; strict = Val(false)
    )
    ode_cache_ode_jac_fn = __multiple_shooting_init_jacobian_odecache(
        ensemblealg, prob, ode_jac_cache, nonbc_diffmode,
        alg.ode_alg, cur_nshoot, u0; internal_ode_kwargs...
    )

    # BC Part
    (; bc_diffmode) = alg.jac_alg
    bc_diffmode = if bc_diffmode isa AutoSparse
        get_dense_ad(alg.jac_alg.bc_diffmode)
    else
        bc_diffmode
    end
    bc_jac_cache = DI.prepare_jacobian(
        nothing, similar(bcresid_prototype), bc_diffmode, u_at_nodes; strict = Val(false)
    )
    ode_cache_bc_jac_fn = __multiple_shooting_init_jacobian_odecache(
        ensemblealg, prob, bc_jac_cache, bc_diffmode,
        alg.ode_alg, cur_nshoot, u0; internal_ode_kwargs...
    )

    # Define the functions now
    ode_fn = @closure (
        du,
        u,
    ) -> solve_internal_odes!(du, u, prob.p, cur_nshoot, nodes, ode_cache_ode_jac_fn)
    bc_fn = @closure (
        du,
        u,
    ) -> __multiple_shooting_mpoint_loss_bc!(
        du, u, prob.p, cur_nshoot, nodes, prob, solve_internal_odes!, N,
        f, bc, u0_size, prob.tspan, alg.ode_alg, u0, ode_cache_bc_jac_fn
    )

    jac_prototype_ode = DI.jacobian(
        ode_fn, similar(u_at_nodes, cur_nshoot * N),
        ode_jac_cache, nonbc_diffmode, u_at_nodes
    )
    jac_prototype_bc = DI.jacobian(
        bc_fn, similar(bcresid_prototype), bc_jac_cache, bc_diffmode, u_at_nodes
    )
    jac_prototype = vcat(sparse(jac_prototype_ode), jac_prototype_bc)

    jac_fn = @closure (
        J,
        u,
        p,
    ) -> __multiple_shooting_mpoint_jacobian!(
        J, u, p, similar(bcresid_prototype), resid_nodes, ode_jac_cache, bc_jac_cache,
        ode_fn, bc_fn, nonbc_diffmode, bc_diffmode, N, M, __cache_trait(alg.jac_alg)
    )

    # NOTE: u_at_nodes is updated inplace
    nlprob = __construct_internal_problem(
        prob, prob.problem_type, alg, loss_fn, jac_fn, jac_prototype,
        resid_prototype, u_at_nodes, prob.p, M, length(nodes), nothing, nothing
    )
    nlsolve_alg = __concrete_solve_algorithm(nlprob, alg.nlsolve, alg.optimize)

    # Extract verbose, nlsolve_kwargs, optimize_kwargs from merged kwargs
    verbose_val = get(kwargs, :verbose, true)
    verbose_spec = _process_verbose_param(verbose_val)
    nlsolve_kw = get(kwargs, :nlsolve_kwargs, (;))
    optimize_kw = get(kwargs, :optimize_kwargs, (;))

    # Construct kwargs with nonlinear verbosity
    concrete_kw = __concrete_kwargs(alg.nlsolve, alg.optimize, nlsolve_kw, optimize_kw, verbose_spec)

    # Filter out verbose, nlsolve_kwargs, optimize_kwargs from kwargs since they're handled
    other_kw = filter(kwargs) do (k, v)
        k ∉ (:verbose, :nlsolve_kwargs, :optimize_kwargs)
    end

    __solve(nlprob, nlsolve_alg; concrete_kw..., other_kw...)

    return nothing
end

function __multiple_shooting_init_odecache(
        ::EnsembleSerial, prob, alg, u0, nshoots; kwargs...
    )
    odeprob = ODEProblem{isinplace(prob)}(prob.f, u0, prob.tspan, prob.p)
    return SciMLBase.__init(odeprob, alg; kwargs...)
end

function __multiple_shooting_init_odecache(
        ::EnsembleThreads, prob, alg, u0, nshoots; kwargs...
    )
    odeprob = ODEProblem{isinplace(prob)}(prob.f, u0, prob.tspan, prob.p)
    return [SciMLBase.__init(odeprob, alg; kwargs...) for _ in 1:nshoots]
end

function __multiple_shooting_init_jacobian_odecache(
        ensemblealg, prob, jac_cache, diffmode, alg, nshoots, u; kwargs...
    )
    return __multiple_shooting_init_jacobian_odecache(
        ensemblealg, prob, jac_cache, __cache_trait(diffmode),
        diffmode, alg, nshoots, u; kwargs...
    )
end

function __multiple_shooting_init_jacobian_odecache(
        ensemblealg, prob, jac_cache, ::NoDiffCacheNeeded,
        diffmode, alg, nshoots, u; kwargs...
    )
    return __multiple_shooting_init_odecache(ensemblealg, prob, alg, u, nshoots; kwargs...)
end

function __multiple_shooting_init_jacobian_odecache(
        ensemblealg, prob, jac_cache, ::DiffCacheNeeded,
        diffmode, alg, nshoots, u; kwargs...
    )
    T_dual = eltype(overloaded_input_type(jac_cache))
    xduals = zeros(T_dual, size(u))
    return __multiple_shooting_init_odecache(
        ensemblealg, prob, alg, xduals, nshoots; kwargs...
    )
end

function __multiple_shooting_matching_odecache(odecache, odecache_for_states, us)
    cache = odecache isa Vector ? first(odecache) : odecache
    return eltype(cache.u) === eltype(us) ? odecache : odecache_for_states[us]
end

# Not using `EnsembleProblem` since it is hard to initialize the cache and stuff
function __multiple_shooting_solve_internal_odes!(
        resid_nodes, us, cur_nshoots::Int, odecache,
        nodes, u0_size, N::Int, ::EnsembleSerial, tspan, platform
    )
    ts_ = Vector{Vector{typeof(first(tspan))}}(undef, cur_nshoots)
    us_ = Vector{Vector{typeof(us)}}(undef, cur_nshoots)

    for i in 1:cur_nshoots
        SciMLBase.reinit!(
            odecache, reshape(@view(us[((i - 1) * N + 1):(i * N)]), u0_size);
            t0 = nodes[i], tf = nodes[i + 1]
        )
        sol = solve!(odecache)
        us_[i] = deepcopy(sol.u)
        ts_[i] = deepcopy(sol.t)
        resid_nodes[((i - 1) * N + 1):(i * N)] .= @view(us[(i * N + 1):((i + 1) * N)]) .-
            vec(sol.u[end])
    end

    return reduce(vcat, us_), reduce(vcat, ts_)
end

# Each work item owns `odecaches[i]` and writes only to `us_[i]`, `ts_[i]` and the
# `resid_nodes` block of interval `i`.
@kernel function __ms_solve_internal_odes_kernel!(
        resid_nodes, us_, ts_, us, odecaches, nodes, u0_size, N
    )
    i = @index(Global, Linear)
    cache = odecaches[i]
    SciMLBase.reinit!(
        cache, reshape(view(us, ((i - 1) * N + 1):(i * N)), u0_size);
        t0 = nodes[i], tf = nodes[i + 1]
    )
    sol = solve!(cache)
    us_[i] = deepcopy(sol.u)
    ts_[i] = deepcopy(sol.t)
    resid_nodes[((i - 1) * N + 1):(i * N)] .= view(us, (i * N + 1):((i + 1) * N)) .-
        vec(sol.u[end])
end

function __multiple_shooting_solve_internal_odes!(
        resid_nodes, us, cur_nshoots::Int, odecache::Vector,
        nodes, u0_size, N::Int, ::EnsembleThreads, tspan, platform
    )
    ts_ = Vector{Vector{typeof(first(tspan))}}(undef, cur_nshoots)
    us_ = Vector{Vector{typeof(us)}}(undef, cur_nshoots)

    kernel! = __ms_solve_internal_odes_kernel!(platform)
    kernel!(
        resid_nodes, us_, ts_, us, odecache, nodes, u0_size, N;
        ndrange = cur_nshoots
    )
    synchronize(platform)

    return reduce(vcat, us_), reduce(vcat, ts_)
end

function __multiple_shooting_2point_jacobian!(
        J, us, p, jac_cache, loss_fn::F, diffmode, resid, alg::MultipleShooting
    ) where {F}
    DI.jacobian!(loss_fn, resid, J, jac_cache, diffmode, us)
    return nothing
end

function __multiple_shooting_mpoint_jacobian!(
        J, us, p, resid_bc, resid_nodes, ode_jac_cache, bc_jac_cache, ode_fn::F1, bc_fn::F2,
        nonbc_diffmode, bc_diffmode, N::Int, M::Int, ::DiffCacheNeeded
    ) where {F1, F2}
    J_bc = @view(J[1:M, :])
    J_c = @view(J[(M + 1):end, :])

    DI.jacobian!(ode_fn, resid_nodes.du, J_c, ode_jac_cache, nonbc_diffmode, us)
    DI.jacobian!(bc_fn, resid_bc, J_bc, bc_jac_cache, bc_diffmode, us)

    return nothing
end
function __multiple_shooting_mpoint_jacobian!(
        J, us, p, resid_bc, resid_nodes, ode_jac_cache, bc_jac_cache, ode_fn::F1, bc_fn::F2,
        nonbc_diffmode, bc_diffmode, N::Int, M::Int, ::NoDiffCacheNeeded
    ) where {F1, F2}
    J_bc = @view(J[1:M, :])
    J_c = @view(J[(M + 1):end, :])

    DI.jacobian!(ode_fn, resid_nodes, J_c, ode_jac_cache, nonbc_diffmode, us)
    DI.jacobian!(bc_fn, resid_bc, J_bc, bc_jac_cache, bc_diffmode, us)

    return nothing
end

@views function __multiple_shooting_2point_loss!(
        resid, us, p, cur_nshoots::Int, nodes, prob, solve_internal_odes!::S,
        resida_len, residb_len, N, bca::BCA, bcb::BCB, ode_cache
    ) where {S, BCA, BCB}
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

@views function __multiple_shooting_mpoint_loss_bc!(
        resid_bc, us, p, cur_nshoots::Int, nodes, prob, solve_internal_odes!::S,
        N, f::F, bc::BC, u0_size, tspan, ode_alg, u0, ode_cache
    ) where {S, F, BC}
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

@views function __multiple_shooting_mpoint_loss!(
        resid, us, p, cur_nshoots::Int, nodes, prob, solve_internal_odes!::S, resid_len,
        N, f::F, bc::BC, u0_size, tspan, ode_alg, u0, ode_cache
    ) where {S, F, BC}
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
@views function __multiple_shooting_initialize!(
        nodes, prob, alg, ::Val{true}, nshoots::Int, odecache; kwargs...
    )
    (; u0, tspan, p) = prob

    resize!(nodes, nshoots + 1)
    nodes .= range(tspan[1], tspan[2]; length = nshoots + 1)
    u0_ = __extract_u0(u0, p, tspan[1])
    guess = __initial_guess_on_mesh(u0, nodes, p)

    N = length(u0_)
    u_at_nodes = similar(u0_, (nshoots + 1) * N)
    recursive_flatten!(u_at_nodes, guess.u)

    return u_at_nodes
end

# No initial guess
@views function __multiple_shooting_initialize!(
        nodes, prob, alg::MultipleShooting, ::Val{false},
        nshoots::Int, odecache_; verbose_spec, kwargs...
    )
    (; f, u0, tspan, p) = prob
    (; ode_alg) = alg

    resize!(nodes, nshoots + 1)
    nodes .= range(tspan[1], tspan[2]; length = nshoots + 1)
    N = length(u0)

    # Ensures type stability in case the parameters are dual numbers
    if !(p isa SciMLBase.NullParameters)
        if !isconcretetype(eltype(p))
            @SciMLMessage(
                "Type inference will fail if eltype(p) is not a concrete type",
                verbose_spec,
                :type_inference
            )
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
        for i in eachindex(nodes)
            u_at_nodes[(i - 1) * N .+ (1:N)] .= vec(sol(nodes[i]))
        end
    else
        @SciMLMessage(
            "Initialization using odesolve failed. Initializing using 0s. It is recommended to provide an initial guess function via `u0 = <function>(p, t)` in this case.",
            verbose_spec,
            :initialization
        )
        fill!(u_at_nodes, 0)
    end

    return u_at_nodes
end

# Grid coarsening
@views function __multiple_shooting_initialize!(
        nodes, u_at_nodes_prev, prob, alg, nshoots,
        old_nshoots, ig, odecache_, u0; kwargs...
    )
    (; f, tspan, p) = prob
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

function __shooting_copy(platform, x::AbstractArray)
    out = KernelAbstractions.allocate(platform, eltype(x), size(x))
    copyto!(out, x)
    return out
end
__shooting_parameter(platform, p::AbstractArray) = isbits(p) ? p : __shooting_copy(platform, p)
__shooting_parameter(platform, p::Union{Tuple, NamedTuple}) = map(x -> __shooting_parameter(platform, x), p)
function __shooting_parameter(platform, p)
    isbits(p) || throw(ArgumentError("Device shooting parameters must be isbits, numeric arrays, or tuples of these."))
    return p
end

@inline function __shooting_eval!(out, f::F, args, ::Val{true}) where {F}
    f(out, args...)
    return nothing
end
@inline function __shooting_eval!(out, f::F, args, ::Val{false}) where {F}
    value = f(args...)
    # Empty static boundary residuals have no valid getindex method on GPU.
    isempty(value) && return nothing
    @inbounds for j in eachindex(out)
        out[j] = value[j]
    end
    return nothing
end

# The optional DiffEqGPU extension specializes these hooks for GPUODEAlgorithm.
# OrdinaryDiffEq integrators are only constructed and executed on CPU.
function __shooting_validate_ode(ode_alg, platform)
    platform isa CPU || throw(
        ArgumentError(
            "GPU MultipleShooting requires a DiffEqGPU kernel algorithm. Load DiffEqGPU and use e.g. GPUTsit5() instead of Tsit5()."
        )
    )
    ode_alg isa SciMLBase.AbstractODEAlgorithm || throw(
        ArgumentError(
            "Fixed-step CPU MultipleShooting requires an ODE algorithm, e.g. OrdinaryDiffEqTsit5.Tsit5()."
        )
    )
    return nothing
end

function __shooting_odecache(ode_alg, u, cache, ::Type{T}) where {T}
    __shooting_validate_ode(ode_alg, cache.platform)
    (; f, p, mesh, n, steps, intervals, iip) = cache
    return map(1:intervals) do i
        u0 = T.(u[((i - 1) * n + 1):(i * n)])
        tspan = (mesh[i], mesh[i + 1])
        prob = ODEProblem{_unwrap_val(iip)}(f, u0, tspan, p)
        SciMLBase.init(
            prob, ode_alg; adaptive = false, dt = (tspan[2] - tspan[1]) / steps,
            save_everystep = false, save_start = false, save_end = false, dense = false
        )
    end
end

@kernel function __shooting_cpu_integrate_kernel!(r, u, integrators, mesh, steps, n, na)
    i = @index(Global, Linear)
    integrator = integrators[i]
    SciMLBase.reinit!(
        integrator, view(u, ((i - 1) * n + 1):(i * n));
        t0 = mesh[i], tf = mesh[i + 1], reset_dt = false
    )
    SciMLBase.set_proposed_dt!(integrator, (mesh[i + 1] - mesh[i]) / steps)
    sol = solve!(integrator)
    SciMLBase.successful_retcode(sol) || error("Shooting interval $i failed: $(sol.retcode)")
    @inbounds for j in 1:n
        r[na + (i - 1) * n + j] = integrator.u[j] - u[i * n + j]
    end
end

function __shooting_integrate!(r, u, ode_alg, cache, integrators)
    __shooting_cpu_integrate_kernel!(cache.platform)(
        r, u, integrators, cache.mesh, cache.steps, cache.n, cache.na;
        ndrange = cache.intervals
    )
    return nothing
end

@kernel function __shooting_derivative_kernel!(d, u, f::F, p, mesh, iip) where {F}
    node = @index(Global, Linear)
    n = size(d, 1)
    __shooting_eval!(view(d, :, node), f, (view(u, ((node - 1) * n + 1):(node * n)), p, mesh[node]), iip)
end

# An allocation-free solution view for boundary functions and public interpolation.
struct ShootingDeviceEvalSol{U, D, T}
    state::U
    d::D
    t::T
    n::Int
end
@inline Base.getproperty(s::ShootingDeviceEvalSol, name::Symbol) = name === :u ? s : getfield(s, name)
Base.length(s::ShootingDeviceEvalSol) = length(s.t)
Base.firstindex(::ShootingDeviceEvalSol) = 1
Base.lastindex(s::ShootingDeviceEvalSol) = length(s)
Base.@propagate_inbounds Base.getindex(s::ShootingDeviceEvalSol, i::Int) = view(s.state, ((i - 1) * s.n + 1):(i * s.n))
Base.@propagate_inbounds Base.getindex(s::ShootingDeviceEvalSol, j::Int, i::Int) = s.state[(i - 1) * s.n + j]
Base.@propagate_inbounds Base.getindex(s::ShootingDeviceEvalSol, ::Colon, i::Int) = s[i]

struct ShootingDeviceValue{T, S, W} <: AbstractVector{T}
    sol::S
    interval::Int
    weight::W
end
Base.size(v::ShootingDeviceValue) = (v.sol.n,)
Base.IndexStyle(::Type{<:ShootingDeviceValue}) = IndexLinear()
@inline function __shooting_interp(s, i, w, j, ::Val{D}) where {D}
    @inbounds begin
        h = s.t[i + 1] - s.t[i]
        a, b = s.state[(i - 1) * s.n + j], s.state[i * s.n + j]
        da, db = s.d[j, i], s.d[j, i + 1]
        if D == 0
            # Endpoint branches retain exact endpoint sparsity during tracing.
            iszero(w) && return a
            isone(w) && return b
            return (2w^3 - 3w^2 + 1) * a + (w^3 - 2w^2 + w) * h * da +
                (-2w^3 + 3w^2) * b + (w^3 - w^2) * h * db
        end
        return (6w^2 - 6w) / h * a + (3w^2 - 4w + 1) * da +
            (-6w^2 + 6w) / h * b + (3w^2 - 2w) * db
    end
end
Base.@propagate_inbounds Base.getindex(v::ShootingDeviceValue, j::Int) = __shooting_interp(v.sol, v.interval, v.weight, j, Val(0))
@inline function (s::ShootingDeviceEvalSol)(t::Number)
    lo, hi = 1, length(s.t)
    @inbounds while lo + 1 < hi
        mid = (lo + hi) ÷ 2
        if s.t[mid] <= t
            lo = mid
        else
            hi = mid
        end
    end
    @inbounds w = (t - s.t[lo]) / (s.t[lo + 1] - s.t[lo])
    return ShootingDeviceValue{eltype(s.state), typeof(s), typeof(w)}(s, lo, w)
end

@kernel function __shooting_boundary_kernel!(r, u, d, bc::B, p, mesh, n, na, nb, iip, ::Val{true}) where {B}
    side = @index(Global, Linear)
    if side == 1
        __shooting_eval!(view(r, 1:na), bc[1], (view(u, 1:n), p), iip)
    else
        __shooting_eval!(view(r, (length(r) - nb + 1):length(r)), bc[2], (view(u, (length(u) - n + 1):length(u)), p), iip)
    end
end
@kernel function __shooting_boundary_kernel!(r, u, d, bc::B, p, mesh, n, na, nb, iip, ::Val{false}) where {B}
    __shooting_eval!(view(r, 1:na), bc, (ShootingDeviceEvalSol(u, d, mesh, n), p, mesh), iip)
end

function __shooting_buffers(u, cache, nr, ::Type{T}) where {T}
    return (;
        input = similar(u, T, length(u)), residual = similar(u, T, nr),
        odecache = __shooting_odecache(cache.ode_alg, u, cache, T),
        derivative = similar(u, T, cache.n, cache.intervals + 1),
    )
end

function __shooting_residual!(r, u, cache, work, selection = :all)
    (; platform, n, intervals, na, nb, f, bc, p, mesh, ode_alg, iip, twopoint) = cache
    if selection !== :boundary
        __shooting_integrate!(r, u, ode_alg, cache, work.odecache)
    end
    if !twopoint && selection !== :continuity
        __shooting_derivative_kernel!(platform)(work.derivative, u, f, p, mesh, iip; ndrange = intervals + 1)
    end
    synchronize(platform)
    if selection !== :continuity
        __shooting_boundary_kernel!(platform)(r, u, work.derivative, bc, p, mesh, n, na, nb, iip, Val(twopoint); ndrange = twopoint ? 2 : 1)
    end
    synchronize(platform)
    return r
end

struct ShootingDeviceInterpolation{S, P} <: SciMLBase.AbstractDiffEqInterpolation
    sol::S
    platform::P
end
SciMLBase.interp_summary(::ShootingDeviceInterpolation) = "Multiple shooting device cubic Hermite interpolation"
@kernel function __shooting_interpolate_kernel!(out, u, d, mesh, n, t, idxs, deriv)
    j = @index(Global, Linear)
    row = idxs === nothing ? j : (idxs isa Integer ? idxs : idxs[j])
    value = ShootingDeviceEvalSol(u, d, mesh, n)(t)
    @inbounds out[j] = __shooting_interp(value.sol, value.interval, value.weight, row, deriv)
end
function (interp::ShootingDeviceInterpolation)(t::Number, idxs, ::Type{Val{D}}, p, continuity::Symbol = :left) where {D}
    D in (0, 1) || throw(ArgumentError("Device shooting interpolation supports derivative orders 0 and 1."))
    s = interp.sol
    ids = idxs === nothing ? (1:s.n) : (idxs isa Integer ? (idxs,) : idxs)
    all(j -> j isa Integer && 1 <= j <= s.n, ids) || throw(BoundsError(1:s.n, idxs))
    out = similar(s.state, length(ids))
    device_idxs = idxs isa AbstractArray ? __shooting_copy(interp.platform, idxs) : idxs
    __shooting_interpolate_kernel!(interp.platform)(out, s.state, s.d, s.t, s.n, t, device_idxs, Val(D); ndrange = length(out))
    synchronize(interp.platform)
    return idxs isa Integer ? sum(out) : out
end
# Supported index forms keep this call distinct from the batch-time signature.
function (interp::ShootingDeviceInterpolation)(out::AbstractArray, t::Number, idxs::Union{Nothing, Integer, AbstractArray, Tuple}, deriv::Type{Val{D}}, p, continuity::Symbol = :left) where {D}
    copyto!(out, interp(t, idxs, deriv, p, continuity))
    return out
end
function (interp::ShootingDeviceInterpolation)(ts::AbstractVector, idxs, deriv::Type{Val{D}}, p, continuity::Symbol = :left) where {D}
    times = collect(ts)
    return DiffEqArray([interp(t, idxs, deriv, p, continuity) for t in times], times)
end

function __shooting_validate(prob, alg, odesolve_kwargs, kwargs)
    alg.device_steps !== nothing || throw(ArgumentError("GPU MultipleShooting requires device_steps (fixed ODE steps per interval)."))
    __shooting_validate_ode(alg.ode_alg, alg.platform)
    alg.grid_coarsening === false || throw(ArgumentError("Device MultipleShooting requires grid_coarsening=false."))
    alg.optimize === nothing || throw(ArgumentError("Device MultipleShooting supports nonlinear solvers, not optimize."))
    isempty(odesolve_kwargs) || throw(ArgumentError("Device MultipleShooting uses device_steps; odesolve_kwargs are not supported."))
    isempty(kwargs) || throw(ArgumentError("Unsupported device shooting solve keywords: $(keys(kwargs)). Use nlsolve_kwargs for nonlinear options."))
    prob.tspan[2] > prob.tspan[1] || throw(ArgumentError("Device shooting requires an increasing tspan."))
    prob.f.mass_matrix == LinearAlgebra.I || throw(ArgumentError("Device shooting requires the identity mass matrix."))
    for name in (:lb, :ub, :lcons, :ucons)
        getproperty(prob, name) === nothing || throw(ArgumentError("Device shooting does not support bounds or optimization constraints."))
    end
    get(prob.kwargs, :tune_parameters, false) && throw(ArgumentError("Device shooting does not support tune_parameters."))
    return nothing
end

function __shooting_device_setup(prob, alg)
    __shooting_validate_ode(alg.ode_alg, alg.platform)
    host_mesh = collect(range(prob.tspan...; length = alg.nshoots + 1))
    state = __extract_u0(prob.u0, prob.p, first(host_mesh))
    state isa AbstractVector || throw(ArgumentError("Device shooting requires vector states."))
    T = eltype(state)
    T <: Union{Float32, Float64} || throw(ArgumentError("Device shooting states must use Float32 or Float64."))
    n, intervals = length(state), alg.nshoots
    n > 0 || throw(ArgumentError("Device shooting needs a nonempty state."))
    # Initialization is allowed on the host. Iteration storage remains resident.
    host_u = if prob.u0 isa AbstractVector{<:Number}
        repeat(Array(prob.u0), intervals + 1)
    else
        guess = __initial_guess_on_mesh(prob.u0, host_mesh, __shooting_host(prob.p))
        length(guess.u) == intervals + 1 && all(x -> length(x) == n, guess.u) ||
            throw(DimensionMismatch("Initial guess must contain nshoots+1 equally sized states."))
        # Materialize the vector of states so reduce uses Base's single-allocation
        # vcat specialization. A generator instead repeatedly copies the prefix.
        reduce(vcat, map(Array, guess.u))
    end
    length(host_u) == n * (intervals + 1) || throw(DimensionMismatch("Initial guess dimensions do not match the shooting mesh."))
    u = __shooting_copy(alg.platform, host_u)
    prototype = prob.f.bcresid_prototype
    twopoint = prob.problem_type isa TwoPointBVProblem
    if twopoint
        prototype === nothing && throw(ArgumentError("Device TwoPointBVProblem requires bcresid_prototype=(left,right)."))
        na, nb = map(length, prototype.x)
    else
        na, nb = prototype === nothing ? (n, 0) : (length(prototype), 0)
    end
    nr = na + n * intervals + nb
    cache = (;
        platform = alg.platform, n, intervals, steps = alg.device_steps, na, nb,
        f = prob.f.f, bc = prob.f.bc, p = __shooting_parameter(alg.platform, prob.p),
        mesh = __shooting_copy(alg.platform, host_mesh), ode_alg = alg.ode_alg,
        iip = Val(isinplace(prob)), twopoint,
    )
    work = __shooting_buffers(u, cache, nr, T)
    plan = __shooting_jacobian_plan(prob, alg, u, host_mesh, cache, work)
    return (; u, host_mesh, cache, work, plan)
end

# The CUDSS extension specializes this hook for shooting's segment layout.
__shooting_default_linsolve(u, cache, plan) = __default_linsolve(u)
__shooting_default_linsolve(u::Array, cache, plan) = __default_sparse_linsolve(plan.matrix)

function __multiple_shooting_device_solve(prob, alg; abstol, odesolve_kwargs, nlsolve_kwargs, optimize_kwargs, ensemblealg, verbose, kwargs...)
    __shooting_validate(prob, alg, odesolve_kwargs, kwargs)
    (; u, host_mesh, cache, work, plan) = __shooting_device_setup(prob, alg)
    residual! = (r, x, p) -> __shooting_residual!(r, x, cache, work)
    jacobian! = (J, x, p) -> __shooting_jacobian!(J, x, cache, plan)

    product = copy(plan.matrix)
    function jvp!(out, v, x, p)
        __shooting_jacobian!(product, x, cache, plan)
        LinearAlgebra.mul!(out, product, v)
        return nothing
    end
    function vjp!(out, v, x, p)
        __shooting_jacobian!(product, x, cache, plan)
        LinearAlgebra.mul!(out, adjoint(product), v)
        return nothing
    end
    nf = NonlinearFunction{true}(
        residual!; jac = jacobian!, jvp = jvp!, vjp = vjp!,
        jac_prototype = plan.matrix, resid_prototype = work.residual
    )
    nlprob = BoundaryValueDiffEqCore.__internal_nlsolve_problem(prob, work.residual, u, nf, u, cache.p)
    if nlprob isa SciMLBase.NonlinearProblem && length(work.residual) != length(u)
        throw(DimensionMismatch("Square shooting requires as many boundary residuals as states; use nlls=Val(true)."))
    end
    linsolve = if alg.nlsolve !== nothing
        nothing
    elseif nlprob isa SciMLBase.NonlinearLeastSquaresProblem
        alg.device_linsolve === nothing ||
            throw(ArgumentError("device_linsolve requires a square nonlinear problem; configure nlsolve for least squares."))
        nothing
    else
        alg.device_linsolve === nothing ?
            __shooting_default_linsolve(u, cache, plan) :
            alg.device_linsolve
    end
    concrete_jac = nlprob isa SciMLBase.NonlinearLeastSquaresProblem || linsolve !== nothing ? true : nothing
    solver = __concrete_device_solve_algorithm(
        nlprob, alg.nlsolve, alg.optimize; linsolve, concrete_jac
    )
    nlsol = __internal_solve(nlprob, solver; abstol, nlsolve_kwargs...)
    y = copy(nlsol.u)
    d = similar(work.derivative)
    __shooting_derivative_kernel!(alg.platform)(d, y, cache.f, cache.p, cache.mesh, cache.iip; ndrange = cache.intervals + 1)
    synchronize(alg.platform)
    interp = ShootingDeviceInterpolation(ShootingDeviceEvalSol(y, d, cache.mesh, cache.n), alg.platform)
    states = [view(y, ((i - 1) * cache.n + 1):(i * cache.n)) for i in eachindex(host_mesh)]
    odesol = SciMLBase.build_solution(prob, alg, host_mesh, states; interp, retcode = nlsol.retcode)
    return __build_solution(prob, odesol, nlsol)
end
