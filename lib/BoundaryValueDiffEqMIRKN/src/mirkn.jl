@concrete struct MIRKNCache{iip, T} <: AbstractBoundaryValueDiffEqCache
    order::Int                 # The order of MIRKN method
    stage::Int                 # The state of MIRKN method
    M::Int                     # The number of equations
    in_size
    f
    bc
    prob                       # SecondOrderBVProblem
    problem_type               # StandardSecondOrderBVProblem
    p                          # Parameters
    alg                        # MIRKN methods
    TU                         # MIRKN Tableau
    bcresid_prototype
    mesh                       # Discrete mesh
    mesh_dt
    host_mesh                  # Host mesh metadata for packed storage; nothing on CPU
    k_discrete                 # Stage information associated with the discrete Runge-Kutta-Nyström method
    y
    y₀
    residual
    jac_prototype              # Flat dense Jacobian buffer; nothing for sparse/CPU
    jacobian_cache             # Sparse matrix and plan; nothing for dense/CPU
    # One scratch pair per mesh interval, so backend work items do not alias
    collocation_cache
    device_cache
    work_buffers               # Owning packed scratch vectors, retained across resizing
    resid_size
    nlsolve_kwargs
    optimize_kwargs
    kwargs
    verbose
end

Base.eltype(::MIRKNCache{iip, T}) where {iip, T} = T

function SciMLBase.__init(
        prob::SecondOrderBVProblem, alg::AbstractMIRKN;
        dt = 0.0, adaptive = false, abstol = 1.0e-6,
        controller = NoErrorControl(), nlsolve_kwargs = (; abstol),
        optimize_kwargs = (; abstol), verbose = DEFAULT_VERBOSE, kwargs...
    )
    u0 = __device_initial_state(prob.u0, prob.p, first(prob.tspan))
    if !(__device_initial_backend(u0) isa CPU)
        return __init_mirkn_device(
            prob, alg, u0; dt, adaptive, abstol, controller,
            nlsolve_kwargs, optimize_kwargs, verbose, kwargs...
        )
    end
    verbose_spec = _process_verbose_param(verbose)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    iip = isinplace(prob)
    @assert (iip || isnothing(alg.optimize)) "Out-of-place constraints don't allow optimization solvers "
    t₀, t₁ = prob.tspan
    ig, T, M, Nig, u0 = __extract_problem_details(prob; dt, check_positive_dt = true)
    mesh = __extract_mesh(prob.u0, t₀, t₁, Nig)
    mesh_dt = diff(mesh)

    TU = constructMIRKN(alg, T)

    # Don't flatten this here, since we need to expand it later if needed
    y₀ = __initial_guess_on_mesh(prob, prob.u0, Nig, prob.p)
    chunksize = pickchunksize(M * (2 * Nig - 2))
    __alloc = @closure x -> __maybe_allocate_diffcache(vec(zero(x)), chunksize, alg.jac_alg)

    y = __alloc.(copy.(y₀.u))
    collocation_cache = [(__alloc(zero(u0)), __alloc(zero(u0))) for _ in 1:Nig]
    stage = alg_stage(alg)
    bcresid_prototype = zero(vcat(u0, u0))
    k_discrete = [
        __maybe_allocate_diffcache(safe_similar(u0, M, stage), chunksize, alg.jac_alg)
            for _ in 1:Nig
    ]

    residual = if iip
        __alloc.(copy.(@view(y₀.u[1:end])))
    else
        nothing
    end

    resid_size = size(bcresid_prototype)
    f,
        bc = if u0 isa AbstractVector
        prob.f, prob.f.bc
    elseif iip
        vecf! = @closure (ddu, du, u, p, t) -> __vec_f!(ddu, du, u, p, t, prob.f, size(u0))
        vecbc! = if !(prob.problem_type isa TwoPointSecondOrderBVProblem)
            @closure (
                r, du, u, p,
                t,
            ) -> __vec_so_bc!(r, du, u, p, t, prob.f.bc, resid_size, size(u0))
        else
            (
                @closure(
                    (
                        r,
                        du,
                        u,
                        p,
                    ) -> __vec_so_bc!(
                        r, du, u, p, first(prob.f.bc), resid_size[1], size(u0)
                    )
                ),
                @closure(
                    (
                        r,
                        du,
                        u,
                        p,
                    ) -> __vec_so_bc!(r, du, u, p, last(prob.f.bc), resid_size[2], size(u0))
                ),
            )
        end
        vecf!, vecbc!
    else
        vecf = @closure (du, u, p, t) -> __vec_f(du, u, p, t, prob.f, size(u0))
        vecbc = if !(prob.problem_type isa TwoPointSecondOrderBVProblem)
            @closure (du, u, p, t) -> __vec_so_bc(du, u, p, t, prob.f.bc, size(u0))
        else
            (
                @closure((du, u, p) -> __vec_so_bc(du, u, p, first(prob.f.bc), size(u0))),
                @closure((du, u, p) -> __vec_so_bc(du, u, p, last(prob.f.bc), size(u0))),
            )
        end
        vecf, vecbc
    end

    # Initial guess objects (`ODESolution`, `VectorOfArray`, functions, ...) must be
    # stripped down to the extracted `u0` vector here: under RecursiveArrayTools v4
    # `AbstractVectorOfArray <: AbstractArray`, so a plain `isa AbstractArray` check
    # would embed e.g. an entire previous solution's type in the cache and force
    # recompilation of all downstream code against it (issue #500).
    prob_ = if !(prob.u0 isa AbstractArray) || prob.u0 isa AbstractVectorOfArray
        remake(prob; u0)
    else
        prob
    end

    return MIRKNCache{iip, T}(
        alg_order(alg), stage, M, size(u0), f, bc, prob_, prob.problem_type,
        prob.p, alg, TU, bcresid_prototype, mesh, mesh_dt, nothing, k_discrete,
        y, y₀, residual, nothing, nothing, collocation_cache,
        __mirkn_device_cache(alg.platform, prob, alg, u0, TU), nothing, resid_size, nlsolve_kwargs,
        optimize_kwargs, (; abstol, dt, adaptive, controller, kwargs...), verbose_spec
    )
end

function SciMLBase.solve!(cache::MIRKNCache{iip, T}) where {iip, T}
    cache.host_mesh === nothing || return __solve_mirkn_device!(cache)
    info::ReturnCode.T = ReturnCode.Success
    N = length(cache.mesh)

    sol_nlprob, info = __perform_mirkn_iteration(cache)

    solu = ArrayPartition.(cache.y₀.u[1:N], cache.y₀.u[(N + 1):end])
    odesol = SciMLBase.build_solution(
        cache.prob, cache.alg, cache.mesh, solu; retcode = info
    )
    return __build_solution(cache.prob, odesol, sol_nlprob)
end

function __perform_mirkn_iteration(cache::MIRKNCache)
    nlprob = __construct_nlproblem(cache, copy(vec(cache.y₀)), copy(cache.y₀))
    solve_alg = __concrete_solve_algorithm(nlprob, cache.alg.nlsolve, cache.alg.optimize)
    kwargs = __concrete_kwargs(
        cache.alg.nlsolve, cache.alg.optimize, cache.nlsolve_kwargs, cache.optimize_kwargs,
        cache.verbose
    )
    sol_nlprob = __internal_solve(nlprob, solve_alg; kwargs...)
    recursive_unflatten!(cache.y₀, sol_nlprob.u)

    return sol_nlprob, sol_nlprob.retcode
end

# Constructing the Nonlinear Problem
function __construct_nlproblem(cache::MIRKNCache{iip}, y::AbstractVector, y₀::AbstractVectorOfArray) where {iip}
    pt = cache.problem_type
    L = length(cache.mesh)

    eval_sol = EvalSol(__restructure_sol(y₀.u[1:L], cache.in_size), cache.mesh, cache)
    eval_dsol = EvalSol(__restructure_sol(y₀.u[(L + 1):end], cache.in_size), cache.mesh, cache)

    loss_bc = if iip
        @closure (
            du, u,
            p,
        ) -> __mirkn_loss_bc!(du, u, p, pt, cache.bc, cache.y, cache.mesh, cache)
    else
        @closure (u, p) -> __mirkn_loss_bc(u, p, pt, cache.bc, cache.y, cache.mesh, cache)
    end

    loss_collocation = if iip
        @closure (
            du,
            u,
            p,
        ) -> __mirkn_loss_collocation!(
            du, u, p, cache.y, cache.mesh, cache.residual, cache
        )
    else
        @closure (
            u,
            p,
        ) -> __mirkn_loss_collocation(u, p, cache.y, cache.mesh, cache.residual, cache)
    end

    loss = if iip
        @closure (
            du,
            u,
            p,
        ) -> __mirkn_loss!(
            du, u, p, cache.y, pt, cache.bc, cache.residual,
            cache.mesh, cache, eval_sol, eval_dsol
        )
    else
        @closure (
            u,
            p,
        ) -> __mirkn_loss(
            u, p, cache.y, pt, cache.bc, cache.mesh, cache, eval_sol, eval_dsol
        )
    end

    return __construct_nlproblem(cache, y, loss_bc, loss_collocation, loss, pt)
end

function __construct_nlproblem(
        cache::MIRKNCache{iip}, y, loss_bc::BC, loss_collocation::C,
        loss::LF, ::StandardSecondOrderBVProblem
    ) where {iip, BC, C, LF}
    (; jac_alg) = cache.alg
    N = length(cache.mesh)

    resid_bc = cache.bcresid_prototype
    L = length(resid_bc)
    resid_collocation = safe_similar(y, cache.M * (2 * N - 2))

    bc_diffmode = if jac_alg.bc_diffmode isa AutoSparse
        AutoSparse(
            get_dense_ad(jac_alg.bc_diffmode);
            sparsity_detector = __default_sparsity_detector(jac_alg.bc_diffmode),
            coloring_algorithm = __default_coloring_algorithm(jac_alg.bc_diffmode)
        )
    else
        jac_alg.bc_diffmode
    end

    cache_bc = if iip
        DI.prepare_jacobian(
            loss_bc, resid_bc, bc_diffmode, y, Constant(cache.p); strict = Val(false)
        )
    else
        DI.prepare_jacobian(
            loss_bc, bc_diffmode, y, Constant(cache.p); strict = Val(false)
        )
    end

    nonbc_diffmode = if jac_alg.nonbc_diffmode isa AutoSparse
        AutoSparse(
            get_dense_ad(jac_alg.nonbc_diffmode);
            sparsity_detector = __default_sparsity_detector(jac_alg.nonbc_diffmode),
            coloring_algorithm = __default_coloring_algorithm(jac_alg.nonbc_diffmode)
        )
    else
        jac_alg.nonbc_diffmode
    end

    cache_collocation = if iip
        DI.prepare_jacobian(
            loss_collocation, resid_collocation, nonbc_diffmode, y, Constant(cache.p);
            strict = Val(false)
        )
    else
        DI.prepare_jacobian(
            loss_collocation, nonbc_diffmode, y, Constant(cache.p); strict = Val(false)
        )
    end

    J_bc = if iip
        DI.jacobian(loss_bc, resid_bc, cache_bc, bc_diffmode, y, Constant(cache.p))
    else
        DI.jacobian(loss_bc, cache_bc, bc_diffmode, y, Constant(cache.p))
    end
    J_c = if iip
        DI.jacobian(
            loss_collocation, resid_collocation, cache_collocation,
            nonbc_diffmode, y, Constant(cache.p)
        )
    else
        DI.jacobian(
            loss_collocation, cache_collocation, nonbc_diffmode, y, Constant(cache.p)
        )
    end

    jac_prototype = vcat(J_bc, J_c)

    jac = if iip
        @closure (
            J,
            u,
            p,
        ) -> __mirkn_mpoint_jacobian!(
            J, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc, cache_collocation,
            loss_bc, loss_collocation, resid_bc, resid_collocation, L, cache.p
        )
    else
        @closure (
            u,
            p,
        ) -> __mirkn_mpoint_jacobian(
            jac_prototype, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc,
            cache_collocation, loss_bc, loss_collocation, L, cache.p
        )
    end
    resid_prototype = vcat(resid_bc, resid_collocation)
    return __construct_internal_problem(
        cache.prob, cache.problem_type, cache.alg, loss, jac,
        jac_prototype, resid_prototype, y, cache.p, cache.M, 2 * N
    )
end

function __construct_nlproblem(
        cache::MIRKNCache{iip}, y, loss_bc::BC, loss_collocation::C,
        loss::LF, ::TwoPointSecondOrderBVProblem
    ) where {iip, BC, C, LF}
    (; nlsolve, jac_alg) = cache.alg
    N = length(cache.mesh)

    resid = vcat(
        @view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
        safe_similar(y, cache.M * 2 * (N - 1)),
        @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end])
    )

    diffmode = if jac_alg.diffmode isa AutoSparse
        AutoSparse(
            get_dense_ad(jac_alg.diffmode);
            sparsity_detector = __default_sparsity_detector(jac_alg.diffmode),
            coloring_algorithm = __default_coloring_algorithm(jac_alg.diffmode)
        )
    else
        jac_alg.diffmode
    end

    diffcache = if iip
        DI.prepare_jacobian(
            loss, resid, diffmode, y, Constant(cache.p); strict = Val(false)
        )
    else
        DI.prepare_jacobian(
            loss, diffmode, y, Constant(cache.p); strict = Val(false)
        )
    end

    jac_prototype = if iip
        DI.jacobian(loss, resid, diffcache, diffmode, y, Constant(cache.p))
    else
        DI.jacobian(loss, diffcache, diffmode, y, Constant(cache.p))
    end

    jac = if iip
        @closure (
            J, u, p,
        ) -> __mirkn_2point_jacobian!(J, u, diffmode, diffcache, loss, resid, p)
    else
        @closure (
            u, p,
        ) -> __mirkn_2point_jacobian(u, jac_prototype, diffmode, diffcache, loss, p)
    end

    resid_prototype = copy(resid)
    return __construct_internal_problem(
        cache.prob, cache.problem_type, cache.alg, loss, jac,
        jac_prototype, resid_prototype, y, cache.p, cache.M, 2 * N
    )
end

function __mirkn_2point_jacobian!(J, x, diffmode, diffcache, loss_fn::L, resid, p) where {L}
    DI.jacobian!(loss_fn, resid, J, diffcache, diffmode, x, Constant(p))
    return J
end

function __mirkn_2point_jacobian(x, J, diffmode, diffcache, loss_fn::L, p) where {L}
    DI.jacobian!(loss_fn, J, diffcache, diffmode, x, Constant(p))
    return J
end

function __mirkn_mpoint_jacobian!(
        J, _, x, bc_diffmode, nonbc_diffmode, bc_diffcache, nonbc_diffcache, loss_bc::BC,
        loss_collocation::C, resid_bc, resid_collocation, L::Int, p
    ) where {BC, C}
    DI.jacobian!(
        loss_bc, resid_bc, @view(J[1:L, :]), bc_diffcache, bc_diffmode, x, Constant(p)
    )
    DI.jacobian!(
        loss_collocation, resid_collocation, @view(J[(L + 1):end, :]),
        nonbc_diffcache, nonbc_diffmode, x, Constant(p)
    )
    return nothing
end

function __mirkn_mpoint_jacobian(
        J, _, x, bc_diffmode, nonbc_diffmode, bc_diffcache, nonbc_diffcache,
        loss_bc::BC, loss_collocation::C, L::Int, p
    ) where {BC, C}
    DI.jacobian!(loss_bc, @view(J[1:L, :]), bc_diffcache, bc_diffmode, x, Constant(p))
    DI.jacobian!(
        loss_collocation, @view(J[(L + 1):end, :]),
        nonbc_diffcache, nonbc_diffmode, x, Constant(p)
    )
    return J
end

@views function __mirkn_loss!(
        resid, u, p, y, pt::StandardSecondOrderBVProblem, bc::BC,
        residual, mesh, cache::MIRKNCache, EvalSol, EvalDSol
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    Φ!(resids[3:end], cache, y_, u, p)
    EvalSol.u[1:end] .= __restructure_sol(y_[1:length(cache.mesh)], cache.in_size)
    EvalSol.cache.k_discrete[1:end] .= cache.k_discrete
    EvalDSol.u[1:end] .= __restructure_sol(y_[(length(cache.mesh) + 1):end], cache.in_size)
    EvalDSol.cache.k_discrete[1:end] .= cache.k_discrete
    eval_bc_residual!(resids[1:2], pt, bc, EvalSol, EvalDSol, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirkn_loss(
        u, p, y, pt::StandardSecondOrderBVProblem, bc::BC,
        mesh, cache::MIRKNCache, EvalSol, EvalDSol
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    resid_co = Φ(cache, y_, u, p)
    EvalSol.u[1:end] .= __restructure_sol(y_[1:length(cache.mesh)], cache.in_size)
    EvalSol.cache.k_discrete[1:end] .= cache.k_discrete
    EvalDSol.u[1:end] .= __restructure_sol(y_[(length(cache.mesh) + 1):end], cache.in_size)
    EvalDSol.cache.k_discrete[1:end] .= cache.k_discrete
    resid_bc = eval_bc_residual(pt, bc, EvalSol, EvalDSol, p, mesh)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end

@views function __mirkn_loss_bc!(
        resid, u, p, pt, bc!::BC, y, mesh, cache::MIRKNCache
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    soly_ = EvalSol(__restructure_sol(y_[1:length(cache.mesh)], cache.in_size), mesh, cache)
    dsoly_ = EvalSol(__restructure_sol(y_[(length(cache.mesh) + 1):end], cache.in_size), cache.mesh, cache)
    eval_bc_residual!(resid, pt, bc!, soly_, dsoly_, p, mesh)
    return nothing
end

@views function __mirkn_loss_bc(u, p, pt, bc!::BC, y, mesh, cache::MIRKNCache) where {BC}
    y_ = recursive_unflatten!(y, u)
    soly_ = EvalSol(__restructure_sol(y_[1:length(cache.mesh)], cache.in_size), mesh, cache)
    dsoly_ = EvalSol(__restructure_sol(y_[(length(cache.mesh) + 1):end], cache.in_size), cache.mesh, cache)
    return eval_bc_residual(pt, bc!, soly_, dsoly_, p, mesh)
end

@views function __mirkn_loss_collocation!(resid, u, p, y, mesh, residual, cache)
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual[3:end]]
    Φ!(resids, cache, y_, u, p)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirkn_loss_collocation(u, p, y, mesh, residual, cache)
    y_ = recursive_unflatten!(y, u)
    resids = Φ(cache, y_, u, p)
    return mapreduce(vec, vcat, resids)
end

@views function __mirkn_loss!(
        resid, u, p, y, pt::TwoPointSecondOrderBVProblem, bc!::BC,
        residual, mesh, cache::MIRKNCache, _, _
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    Φ!(resids[3:end], cache, y_, u, p)
    eval_bc_residual!(resids[1:2], pt, bc!, y_, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirkn_loss(
        u, p, y, pt::TwoPointSecondOrderBVProblem,
        bc!::BC, mesh, cache::MIRKNCache, _, _
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    resid_co = Φ(cache, y_, u, p)
    resid_bc = eval_bc_residual(pt, bc!, y_, p, mesh)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end

# Resident storage interleaves position and velocity at each node:
# y[1:M, i] = u_i, y[M+1:2M, i] = du_i. Consequently each interval has
# one contiguous 2M residual block and only two adjacent unknown blocks.
@inline __mirkn_states(cache::MIRKNCache) = __reshape_buffer(cache.y, 2cache.M, length(cache.host_mesh))
@inline __mirkn_stages(cache::MIRKNCache) = __reshape_buffer(cache.k_discrete, cache.M, cache.TU.s, length(cache.host_mesh) - 1)
@inline __mirkn_collocation(cache::MIRKNCache) = __reshape_buffer(cache.collocation_cache, 2cache.M, length(cache.host_mesh) - 1)
@inline __mirkn_jacobian(cache::MIRKNCache) = cache.jacobian_cache === nothing ?
    __reshape_buffer(cache.jac_prototype, length(cache.residual), length(cache.y)) : cache.jacobian_cache[nothing].matrix
@inline __mirkn_jacobian_plan(cache::MIRKNCache) = cache.jacobian_cache === nothing ?
    nothing : cache.jacobian_cache[nothing].plan
BoundaryValueDiffEqCore.__bvp_device_residual_prototype(cache::MIRKNCache) = cache.residual
BoundaryValueDiffEqCore.__bvp_device_jacobian_plan(cache::MIRKNCache) = __mirkn_jacobian_plan(cache)

# MIRKN still solves fixed meshes. Keep buffer resizing separate from structural
# discovery so repeated solves retain the owning arrays and rebuild only sparse
# metadata that can depend on mutable problem parameters.
function __mirkn_resize_buffers!(cache::MIRKNCache)
    empty!(cache.device_cache)
    N = length(cache.host_mesh) - 1
    M = cache.M
    resize!(cache.y, 2M * (N + 1))
    resize!(cache.mesh, N + 1)
    resize!(cache.mesh_dt, N)
    copyto!(cache.mesh, cache.host_mesh)
    copyto!(cache.mesh_dt, diff(cache.host_mesh))
    resize!(cache.k_discrete, M * cache.TU.s * N)
    resize!(cache.collocation_cache, 2M * N)
    nbc = cache.problem_type isa TwoPointSecondOrderBVProblem ? sum(prod, cache.resid_size) : prod(first(cache.resid_size))
    resize!(cache.residual, 2M * N + nbc)
    if cache.jacobian_cache === nothing
        resize!(cache.jac_prototype, length(cache.residual) * length(cache.y))
    end
    return cache
end

function __init_mirkn_device(
        prob, alg, u0; dt, abstol, adaptive, controller, nlsolve_kwargs,
        optimize_kwargs, verbose, kwargs...
    )
    adaptive && throw(ArgumentError("MIRKN supports fixed meshes; use adaptive = false."))
    controller isa NoErrorControl || throw(ArgumentError("MIRKN supports NoErrorControl only."))
    if alg.optimize !== nothing || prob.f.inequality !== nothing || prob.f.equality !== nothing ||
            prob.lb !== nothing || prob.ub !== nothing
        throw(ArgumentError("MIRKN optimization and constraints require a CPU initial guess with platform selecting GPU collocation."))
    end
    get(prob.kwargs, :tune_parameters, false) && throw(ArgumentError("MIRKN does not support parameter tuning."))
    eltype(u0) <: Union{Float32, Float64} || throw(ArgumentError("Device MIRKN requires Float32 or Float64 states."))
    platform = KernelAbstractions.get_backend(u0)
    typeof(alg.platform) === typeof(platform) && (platform = alg.platform)
    @set! alg.platform = platform
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    twopoint = prob.problem_type isa TwoPointSecondOrderBVProblem
    modes = twopoint ? (alg.jac_alg.diffmode,) : (alg.jac_alg.bc_diffmode, alg.jac_alg.nonbc_diffmode)
    foreach(__device_validate_ad, modes)
    _, T, M, N, _ = __extract_problem_details(prob; dt, check_positive_dt = true)
    host_mesh = collect(__extract_mesh(prob.u0, prob.tspan..., N))
    to_device(x) = __device_parameter(platform, x)
    mesh, mesh_dt = to_device(host_mesh), to_device(diff(host_mesh))
    y_buffer = similar(u0, T, 2M * (N + 1))
    y = __reshape_buffer(y_buffer, 2M, N + 1)
    __mirkn_device_initial_guess!(view(y, 1:M, :), prob.u0, prob.p, host_mesh, u0)
    copyto!(view(y, (M + 1):2M, :), view(y, 1:M, :))
    host_TU = constructMIRKN(alg, T)
    TU = MIRKNTableau(
        host_TU.s, map(
            to_device, (
                host_TU.c, host_TU.v, host_TU.w, host_TU.b,
                host_TU.x, host_TU.vp, host_TU.bp, host_TU.xp,
            )
        )...
    )
    bc_sizes = __device_bc_sizes(prob, u0)
    nbc = twopoint ? sum(prod, bc_sizes) : prod(first(bc_sizes))
    residual = similar(y, T, 2M * N + nbc)
    k = similar(y_buffer, T, M * TU.s * N)
    tmp = similar(y_buffer, T, 2M * N)
    jacobian = __mirkn_prepare_device_jacobian(prob, alg, y, host_mesh, bc_sizes, prob.p, size(u0))
    return MIRKNCache{isinplace(prob), T}(
        alg_order(alg), TU.s, M, size(u0), prob.f.f, prob.f.bc, prob, prob.problem_type,
        to_device(prob.p), alg, TU, nothing, mesh, mesh_dt, host_mesh, k, y_buffer, nothing, residual,
        jacobian.plan === nothing ? copy(vec(jacobian.matrix)) : nothing,
        jacobian.plan === nothing ? nothing : Dict(nothing => jacobian),
        tmp, Dict{DataType, Any}(), Dict{DataType, Any}(), bc_sizes,
        __concrete_kwargs(alg.nlsolve, nothing, nlsolve_kwargs, optimize_kwargs, _process_verbose_param(verbose)),
        optimize_kwargs, (; abstol, dt, adaptive, controller, kwargs...), _process_verbose_param(verbose)
    )
end

function __mirkn_device_buffers(cache::MIRKNCache, ::Type{T}) where {T}
    return get!(cache.device_cache, T) do
        if T === eltype(__mirkn_states(cache))
            (; k = __mirkn_stages(cache), tmp = __mirkn_collocation(cache))
        else
            buffers = get!(cache.work_buffers, T) do
                (; k = similar(cache.k_discrete, T), tmp = similar(cache.collocation_cache, T))
            end
            resize!(buffers.k, length(cache.k_discrete))
            resize!(buffers.tmp, length(cache.collocation_cache))
            (;
                k = __reshape_buffer(buffers.k, size(__mirkn_stages(cache))),
                tmp = __reshape_buffer(buffers.tmp, size(__mirkn_collocation(cache))),
            )
        end
    end
end

# Structural discovery and coloring run on the host before solves and cache reuse. Only
# index metadata and sparse storage are transferred to the device; neither the
# current state nor a numerically evaluated Jacobian is copied to the host.

# Constructing the Nonlinear Problem
function __construct_nlproblem(cache::MIRKNCache, u0::AbstractVector)
    loss! = (r, u, p) -> __device_residual!(r, u, cache)
    jac! = (J, u, p) -> __device_jacobian!(J, u, cache)
    nf = SciMLBase.NonlinearFunction{true}(
        loss!; jac = jac!, __device_jacobian_products(cache)...,
        resid_prototype = cache.residual, jac_prototype = __mirkn_jacobian(cache)
    )
    return __mirkn_device_nlproblem(cache.prob, nf, u0, cache.p, cache.residual)
end

# Core's second-order constructor always builds a square NonlinearProblem.
# Classify resident problems by the declared nlls setting and actual dimensions.
function __mirkn_device_nlproblem(
        ::SecondOrderBVProblem{U, T, I, N}, nf, u0, p, residual
    ) where {U, T, I, N}
    least_squares = (N === Val{true} || N === true) || (N === Nothing && length(residual) != length(u0))
    if least_squares
        return SciMLBase.NonlinearLeastSquaresProblem(nf, u0, p)
    end
    length(residual) == length(u0) || throw(DimensionMismatch("A square MIRKN problem requires 2M boundary residuals; use nlls = Val(true) for least squares."))
    return SciMLBase.NonlinearProblem(nf, u0, p)
end

function __solve_mirkn_device!(cache::MIRKNCache)
    __mirkn_resize_buffers!(cache)
    # Mutable parameters can change BC dependencies, so retrace/recolor on reuse.
    __device_copy_parameter!(cache.p, cache.prob.p)
    if cache.jacobian_cache !== nothing
        cache.jacobian_cache[nothing] = __mirkn_prepare_device_jacobian(
            cache.prob, cache.alg, __mirkn_states(cache),
            cache.host_mesh, cache.resid_size, cache.prob.p, cache.in_size
        )
    end
    nlprob = __construct_nlproblem(cache, copy(vec(__mirkn_states(cache))))
    sol_nlprob = __internal_solve(nlprob, __concrete_solve_algorithm(nlprob, cache); cache.nlsolve_kwargs...)
    copyto!(vec(__mirkn_states(cache)), sol_nlprob.u)
    # Own output storage so a subsequent solve! cannot modify an existing solution.
    y = copy(__mirkn_states(cache))
    M = cache.M
    solu = [
        ArrayPartition(
                reshape(copy(view(y, 1:M, i)), cache.in_size),
                reshape(copy(view(y, (M + 1):2M, i)), cache.in_size)
            ) for i in axes(y, 2)
    ]
    interp = MIRKNDeviceInterpolation(y, copy(cache.mesh), cache.in_size, cache.alg.platform)
    odesol = SciMLBase.build_solution(
        cache.prob, cache.alg, copy(cache.host_mesh), solu;
        interp, retcode = sol_nlprob.retcode
    )
    return __build_solution(cache.prob, odesol, sol_nlprob)
end

# Initial Guess and Backend

function __mirkn_device_initial_guess!(y, guess, p, mesh, u0)
    states = if guess isa AbstractVector{<:AbstractArray}
        guess
    elseif guess isa Union{AbstractVectorOfArray, SciMLBase.ODESolution}
        guess.u
    else
        nothing
    end
    for i in eachindex(mesh)
        state = states !== nothing ? states[i] :
            (
                guess isa Function ?
                (i == 1 ? u0 : __device_initial_state(guess, p, mesh[i])) : u0
            )
        size(state) == size(u0) || throw(DimensionMismatch("Initial guess state sizes differ."))
        typeof(KernelAbstractions.get_backend(state)) ===
            typeof(KernelAbstractions.get_backend(y)) || throw(
            ArgumentError("All initial guess states must be on the same backend.")
        )
        copyto!(view(y, :, i), vec(state))
    end
    return y
end

BoundaryValueDiffEqCore.__bvp_device_unknowns(cache::MIRKNCache) = __mirkn_states(cache)

function __mirkn_device_boundary_pattern(
        prob, alg, y, host_mesh, bc_sizes, p, in_size
    )
    M, nodes = size(y)
    nbc, nunknowns = prod(bc_sizes[1]), length(y)
    return __device_boundary_pattern(alg.jac_alg.bc_diffmode, eltype(y), nbc, nunknowns) do
        host_p = __device_host_parameter(p)
        iip = Val(isinplace(prob))
        function boundary!(residual, x)
            states = reshape(x, M, nodes)
            sol = MIRKNDeviceEvalSol(states, host_mesh, in_size, 0)
            dsol = MIRKNDeviceEvalSol(states, host_mesh, in_size, M ÷ 2)
            __device_eval!(
                __device_reshape(residual, bc_sizes[1]), prob.f.bc,
                (dsol, sol, host_p, host_mesh), iip
            )
            return nothing
        end
        return boundary!
    end
end

function __generate_sparse_jacobian_prototype(
        prob::SecondOrderBVProblem, alg::AbstractMIRKN, y::AbstractMatrix, host_mesh,
        bc_sizes, p, in_size = (size(y, 1) ÷ 2,)
    )
    boundary = () -> __mirkn_device_boundary_pattern(prob, alg, y, host_mesh, bc_sizes, p, in_size)
    return __device_sparse_structure(prob.problem_type, alg.jac_alg, y, bc_sizes, 0, boundary)
end

function __mirkn_prepare_device_jacobian(prob, alg, y, host_mesh, bc_sizes, p, in_size)
    return __prepare_device_jacobian(y, prob.problem_type, bc_sizes) do
        __generate_sparse_jacobian_prototype(prob, alg, y, host_mesh, bc_sizes, p, in_size)
    end
end
