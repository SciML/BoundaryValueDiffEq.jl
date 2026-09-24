@concrete struct MIRKCache{iip, T, use_both, diffcache, tune_parameters, Y} <:
    AbstractBoundaryValueDiffEqCache
    order::Int                 # The order of MIRK method
    stage::Int                 # The state of MIRK method
    M::Int                     # The number of equations
    in_size
    f
    mass_matrix
    algebraic_indices
    bc
    prob                       # BVProblem
    problem_type               # StandardBVProblem
    p                          # Parameters
    alg                        # MIRK methods
    TU                         # MIRK Tableau
    ITU                        # MIRK Interpolation Tableau
    f_prototype
    bcresid_prototype
    mesh                       # Discrete mesh
    mesh_dt                    # Step size
    host_mesh                  # Host mesh metadata for packed storage; nothing on CPU
    k_discrete                 # Stage information associated with the discrete Runge-Kutta method
    k_interp                   # Stage information associated with the discrete Runge-Kutta method
    y::Y
    y₀
    y₀_flat                    # Flat Vector{T} mirror of y₀ used as nlprob u0 to keep
    # LinearSolve / NonlinearSolveBase happy (they require a
    # concrete `Vector{T}`, not the `Base.ReshapedArray` that
    # `vec(::VectorOfArray)` returns under RAT v4).
    residual
    jac_prototype              # Flat dense device Jacobian buffer; nothing for sparse/CPU
    jacobian_cache             # Typed sparse matrix/plan dictionary; nothing for dense/CPU
    # CPU scratch caches outside collocation keep their original size.
    fᵢ_cache
    fᵢ₂_cache
    # One scratch cache per mesh interval, so backend work items do not alias
    collocation_cache
    device_cache               # Offload buffers or packed AD buffers; nothing on CPU
    errors
    new_stages
    resid_size
    singular_term
    nlsolve_kwargs
    optimize_kwargs
    kwargs
    verbose
    # Element-type-adaptive buffer for the boundary-condition EvalSol. The residual can
    # be differentiated directly (e.g. a line-search directional derivative), so the
    # solution handed to the BCs may carry ForwardDiff.Duals; the lazy cache allocates a
    # matching-eltype buffer on demand and reuses it across calls.
    eval_sol_cache
end

Base.eltype(::MIRKCache{iip, T, use_both}) where {iip, T, use_both} = T

# Shaped arrays are ephemeral views of the owning vectors. Never retain one
# across resize!: GPU reshape objects may still refer to the old allocation.
@inline __mirk_states(cache::MIRKCache) = __reshape_buffer(cache.y, cache.M, length(cache.mesh))
@inline __mirk_stages(cache::MIRKCache) =
    __reshape_buffer(cache.k_discrete, cache.M, cache.stage, length(cache.mesh_dt))
@inline __mirk_interp_stages(cache::MIRKCache) =
    __reshape_buffer(cache.k_interp, cache.M, cache.ITU.s_star - cache.stage, length(cache.mesh_dt))
@inline __mirk_collocation(cache::MIRKCache) =
    __reshape_buffer(cache.collocation_cache, cache.M, length(cache.mesh_dt))
@inline __mirk_rhs_tmp(cache::MIRKCache) =
    __reshape_buffer(cache.fᵢ₂_cache, cache.M, length(cache.mesh_dt))
@inline __mirk_jacobian(cache::MIRKCache) = cache.jacobian_cache === nothing ?
    __reshape_buffer(cache.jac_prototype, length(cache.residual), length(cache.y)) :
    cache.jacobian_cache[nothing].matrix
@inline __mirk_jacobian_plan(cache::MIRKCache) = cache.jacobian_cache === nothing ?
    nothing : cache.jacobian_cache[nothing].plan

BoundaryValueDiffEqCore.__bvp_device_residual_prototype(cache::MIRKCache) = cache.residual
BoundaryValueDiffEqCore.__bvp_device_jacobian_plan(cache::MIRKCache) = __mirk_jacobian_plan(cache)

function SciMLBase.__init(
        prob::BVProblem, alg::AbstractMIRK; dt = 0.0, abstol = 1.0e-6, adaptive = true,
        controller = DefectControl(), nlsolve_kwargs = (; abstol),
        optimize_kwargs = (; abstol), verbose = DEFAULT_VERBOSE, kwargs...
    )
    initial_state = __device_initial_state(prob.u0, prob.p, first(prob.tspan))
    if !(__device_initial_backend(initial_state) isa CPU)
        return __init_mirk_device(
            prob, alg, initial_state; dt, abstol, adaptive, controller,
            nlsolve_kwargs, optimize_kwargs, verbose, kwargs...
        )
    end
    verbose_spec = _process_verbose_param(verbose)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    iip = isinplace(prob)
    diffcache = __cache_trait(alg.jac_alg)
    @assert (iip || isnothing(alg.optimize)) "Out-of-place constraints don't allow optimization solvers "

    tune_parameters = haskey(prob.kwargs, :tune_parameters)
    if tune_parameters
        prob.p isa SciMLBase.NullParameters &&
            throw(ArgumentError("`tune_parameters` is true but `prob.p` is not set."))
    end

    constraint = (!isnothing(prob.f.inequality)) ||
        (!isnothing(prob.f.equality)) ||
        (!isnothing(prob.lb)) ||
        (!isnothing(prob.ub))

    t₀, t₁ = prob.tspan
    ig, T,
        N,
        Nig,
        u0 = __extract_problem_details(prob; dt, check_positive_dt = true, tune_parameters)
    __mirk_validate_device_problem(alg.platform, prob, alg, u0, tune_parameters)
    mesh = __extract_mesh(prob.u0, t₀, t₁, Nig)
    mesh_dt = diff(mesh)

    chunksize = pickchunksize(N * (Nig - 1))
    __alloc = @closure x -> __maybe_allocate_diffcache(vec(zero(x)), chunksize, alg.jac_alg)

    fᵢ_cache = __alloc(zero(u0))
    fᵢ₂_cache = vec(zero(u0))
    collocation_cache = [__alloc(zero(u0)) for _ in 1:Nig]

    # Don't flatten this here, since we need to expand it later if needed
    y₀ = __initial_guess_on_mesh(prob.u0, mesh, prob.p; tune_parameters)
    y₀_flat = collect(vec(y₀))

    y = __alloc.(copy.(y₀.u))
    TU, ITU = constructMIRK(alg, T)
    stage = alg_stage(alg)
    f_prototype = if isnothing(prob.f.f_prototype)
        constraint ? __vec(u0) : nothing
    else
        __vec(prob.f.f_prototype)
    end
    L_f_prototype = isnothing(f_prototype) ? N : length(f_prototype)

    k_discrete = if !constraint
        [
            __maybe_allocate_diffcache(safe_similar(u0, N, stage), chunksize, alg.jac_alg)
                for _ in 1:Nig
        ]
    else
        [
            __maybe_allocate_diffcache(safe_similar(u0, L_f_prototype, stage), chunksize, alg.jac_alg)
                for _ in 1:Nig
        ]
    end
    k_interp = if !constraint
        VectorOfArray([safe_similar(u0, N, ITU.s_star - stage) for _ in 1:Nig])
    else
        VectorOfArray([safe_similar(u0, L_f_prototype, ITU.s_star - stage) for _ in 1:Nig])
    end

    bcresid_prototype, resid₁_size = __get_bcresid_prototype(prob.problem_type, prob, u0)

    residual = if iip
        if !constraint
            if prob.problem_type isa TwoPointBVProblem
                vcat([__alloc(__vec(bcresid_prototype))], __alloc.(copy.(@view(y₀.u[2:end]))))
            else
                vcat([__alloc(bcresid_prototype)], __alloc.(copy.(@view(y₀.u[2:end]))))
            end
        else
            if prob.problem_type isa TwoPointBVProblem
                vcat(
                    [__alloc(__vec(bcresid_prototype))], __alloc.(
                        copy.(
                            [
                                f_prototype
                                    for _ in 1:Nig
                            ]
                        )
                    )
                )
            else
                vcat(
                    [__alloc(bcresid_prototype)], __alloc.(
                        copy.(
                            [
                                f_prototype
                                    for _ in 1:Nig
                            ]
                        )
                    )
                )
            end
        end
    else
        nothing
    end

    use_both = __use_both_error_control(controller)
    errors = if !constraint
        VectorOfArray(
            [
                safe_similar(u0, ifelse(adaptive, N, 0))
                    for _ in 1:ifelse(use_both, 2Nig, Nig)
            ]
        )
    else
        VectorOfArray(
            [
                safe_similar(u0, ifelse(adaptive, L_f_prototype, 0))
                    for _ in 1:ifelse(use_both, 2Nig, Nig)
            ]
        )
    end
    new_stages = if !constraint
        VectorOfArray([safe_similar(u0, N) for _ in 1:Nig])
    else
        VectorOfArray([safe_similar(u0, L_f_prototype) for _ in 1:Nig])
    end

    # Transform the functions to handle non-vector inputs
    bcresid_prototype = __vec(bcresid_prototype)
    f,
        bc = if u0 isa AbstractVector
        f_wrapped = prob.f
        bc_wrapped = prob.f.bc
        if tune_parameters && SciMLStructures.isscimlstructure(prob.p)
            tunable_part, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
            l_parameters = length(tunable_part)
            f_wrapped = @closure (
                du,
                u,
                p,
                t,
            ) -> begin
                @inbounds @views begin
                    _p = repack(u[(end - l_parameters + 1):end])
                    prob.f(du, u, _p, t)
                    fill!(du[(end - l_parameters + 1):end], zero(eltype(du)))
                end
                return nothing
            end
        elseif tune_parameters
            l_parameters = length(prob.p)
            f_wrapped = @closure (
                du,
                u,
                p,
                t,
            ) -> begin
                @inbounds @views begin
                    prob.f(du, u, u[(end - l_parameters + 1):end], t)
                    fill!(du[(end - l_parameters + 1):end], zero(eltype(du)))
                end
                return nothing
            end
        end
        f_wrapped, bc_wrapped
    elseif iip
        vecf! = @closure (du, u, p, t) -> __vec_f!(du, u, p, t, prob.f, size(u0))
        vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
            @closure (r, u, p, t) -> __vec_bc!(r, u, p, t, prob.f.bc, resid₁_size, size(u0))
        else
            (
                @closure(
                    (
                        r, u,
                        p,
                    ) -> __vec_bc!(r, u, p, first(prob.f.bc), resid₁_size[1], size(u0))
                ),
                @closure(
                    (
                        r, u, p,
                    ) -> __vec_bc!(r, u, p, last(prob.f.bc), resid₁_size[2], size(u0))
                ),
            )
        end
        vecf!, vecbc!
    else
        vecf = @closure (u, p, t) -> __vec_f(u, p, t, prob.f, size(u0))
        vecbc = if !(prob.problem_type isa TwoPointBVProblem)
            @closure (u, p, t) -> __vec_bc(u, p, t, prob.f.bc, size(u0))
        else
            (
                @closure((u, p) -> __vec_bc(u, p, first(prob.f.bc), size(u0))),
                @closure((u, p) -> __vec_bc(u, p, last(prob.f.bc), size(u0))),
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

    algebraic_indices = __get_algebraic_indices(prob.f.mass_matrix)
    __check_dae_adaptivity(algebraic_indices, adaptive)

    return MIRKCache{iip, T, use_both, typeof(diffcache), tune_parameters}(
        alg_order(alg), stage, N, size(u0), f, prob.f.mass_matrix, algebraic_indices, bc, prob_, prob.problem_type, prob.p, alg,
        TU, ITU, f_prototype, bcresid_prototype, mesh, mesh_dt, nothing, k_discrete, k_interp, y,
        y₀, y₀_flat, residual, nothing, nothing, fᵢ_cache, fᵢ₂_cache, collocation_cache,
        __mirk_device_cache(alg.platform, prob, alg, u0, TU, tune_parameters), errors,
        new_stages, resid₁_size, prob.singular_term, nlsolve_kwargs, optimize_kwargs,
        (; abstol, dt, adaptive, controller, tune_parameters, kwargs...), verbose_spec,
        LazyBufferCache()
    )
end


function __init_mirk_device(
        prob, alg, u0; dt, abstol, adaptive, controller, nlsolve_kwargs,
        optimize_kwargs, verbose, kwargs...
    )
    # Determine algebraic rows from host metadata before launching kernels.
    host_mass = __device_host_parameter(prob.f.mass_matrix)
    algebraic_indices = __get_algebraic_indices(host_mass)
    __check_dae_adaptivity(algebraic_indices, adaptive)
    platform = KernelAbstractions.get_backend(u0)
    tune_parameters = get(prob.kwargs, :tune_parameters, false)
    if tune_parameters
        isinplace(prob) && u0 isa AbstractVector && prob.p isa AbstractVector{<:Number} ||
            throw(ArgumentError("Resident MIRK parameter tuning requires an in-place RHS, vector states and numeric vector parameters."))
    end
    if alg.optimize !== nothing || prob.f.inequality !== nothing ||
            prob.f.equality !== nothing || prob.lb !== nothing || prob.ub !== nothing
        throw(ArgumentError("MIRK optimization and constraints require a CPU initial guess with `platform` selecting GPU collocation."))
    end
    # Keep explicit launch/compiler options for the matching device backend.
    typeof(alg.platform) === typeof(platform) && (platform = alg.platform)
    @set! alg.platform = platform
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    twopoint = prob.problem_type isa TwoPointBVProblem
    modes = twopoint ? (alg.jac_alg.diffmode,) :
        (alg.jac_alg.bc_diffmode, alg.jac_alg.nonbc_diffmode)
    foreach(__device_validate_ad, modes)

    _, T, M, N, _ = __extract_problem_details(prob; dt, check_positive_dt = true)
    nparameters = tune_parameters ? length(prob.p) : 0
    M += nparameters
    host_mesh = collect(__extract_mesh(prob.u0, prob.tspan..., N))
    host_dt = diff(host_mesh)

    to_device(x) = __device_parameter(platform, x)
    mesh, mesh_dt = to_device(host_mesh), to_device(host_dt)
    y_buffer = similar(u0, T, M * (N + 1))
    y = __reshape_buffer(y_buffer, M, N + 1)
    __mirk_device_initial_guess!(view(y, 1:(M - nparameters), :), prob.u0, prob.p, host_mesh, u0)
    if tune_parameters
        parameters = to_device(prob.p)
        for node in axes(y, 2)
            copyto!(view(y, (M - nparameters + 1):M, node), parameters)
        end
    end
    in_size = tune_parameters ? (M,) : size(u0)

    TU, ITU = constructMIRK(alg, T)
    TU = MIRKTableau(TU.s, to_device(TU.c), to_device(TU.v), to_device(TU.b), to_device(TU.x))
    ITU = MIRKInterpTableau(
        ITU.s_star, to_device(ITU.c_star), to_device(ITU.v_star), to_device(ITU.x_star),
        ITU.τ_star, ITU.p_star
    )

    bc_sizes = __device_bc_sizes(prob, tune_parameters ? view(y, :, 1) : u0)
    nbc = twopoint ? sum(prod, bc_sizes) : prod(first(bc_sizes))
    nresid = M * N + nbc
    k = similar(y_buffer, M * TU.s * N)
    ki = similar(y_buffer, M * (ITU.s_star - TU.s) * N)
    tmp = similar(y_buffer, M * N)
    residual = similar(y, T, (nresid,))
    jacobian = __mirk_prepare_device_jacobian(
        prob, alg, y, host_mesh, TU, ITU, bc_sizes, prob.p, in_size
    )
    jac_prototype = jacobian.plan === nothing ? copy(vec(jacobian.matrix)) : nothing
    jacobian_cache = jacobian.plan === nothing ? nothing : Dict(nothing => jacobian)
    singular_term = to_device(prob.singular_term)

    # Strip ODEFunction metadata before passing the RHS to a kernel.
    f = __device_function(prob.f.f)
    tune_parameters && (f = BVPTunableRHS(f, nparameters))
    verbose_spec = _process_verbose_param(verbose)
    return MIRKCache{isinplace(prob), T, false, NoDiffCacheNeeded, tune_parameters}(
        alg_order(alg), TU.s, M, in_size, f, to_device(host_mass), to_device(algebraic_indices),
        prob.f.bc, prob, prob.problem_type,
        to_device(prob.p), alg, TU, ITU, nothing, nothing,
        mesh, mesh_dt, host_mesh, k, ki, y_buffer,
        nothing, nothing, residual, jac_prototype, jacobian_cache,
        nothing, similar(tmp), tmp, Dict{DataType, Any}(),
        similar(y, T, (N,)),
        (; y = similar(y_buffer, 0), mesh = similar(mesh, 0)),
        bc_sizes, singular_term,
        __concrete_kwargs(alg.nlsolve, nothing, nlsolve_kwargs, optimize_kwargs, verbose_spec),
        optimize_kwargs, (; abstol, dt, adaptive, controller, tune_parameters, kwargs...),
        verbose_spec, nothing
    )
end

function __mirk_device_initial_guess!(y, guess, p, mesh, u0)
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

"""
    __expand_cache!(cache::MIRKCache)

After redistributing or halving the mesh, this function expands the required vectors to
match the length of the new mesh.
"""
function __expand_cache!(cache::MIRKCache{iip, T, use_both}) where {iip, T, use_both}
    Nₙ = length(cache.mesh)
    __resize!(cache.k_discrete, Nₙ - 1, cache.M)
    __resize!(cache.k_interp.u, Nₙ - 1, cache.M)
    __resize!(cache.y, Nₙ, cache.M)
    __resize!(cache.y₀.u, Nₙ, cache.M)
    resize!(cache.y₀_flat, Nₙ * cache.M)
    __resize!(cache.residual, Nₙ, cache.M)
    __resize!(cache.collocation_cache, Nₙ - 1, cache.M)
    __resize!(cache.errors.u, ifelse(use_both, 2 * (Nₙ - 1), (Nₙ - 1)), cache.M)
    __resize!(cache.new_stages.u, Nₙ - 1, cache.M)
    __mirk_reset_device_cache!(cache.device_cache)
    return cache
end

function __expand_cache!(
        cache::MIRKCache{iip, T, U, D, P, Y}
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    M, N = cache.M, length(cache.mesh_dt)
    # Cached primal and Dual views may still alias the old device allocation.
    empty!(cache.device_cache)
    resize!(cache.k_discrete, M * cache.stage * N)
    resize!(cache.k_interp, M * (cache.ITU.s_star - cache.stage) * N)
    resize!(cache.collocation_cache, M * N)
    resize!(cache.fᵢ₂_cache, M * N)
    nbc = cache.problem_type isa TwoPointBVProblem ? sum(prod, cache.resid_size) : prod(cache.resid_size[1])
    resize!(cache.residual, M * N + nbc)
    __expand_cache!(cache, Val(:jacobian))
    resize!(cache.errors, N)
    return cache
end

function SciMLBase.solve!(cache::MIRKCache)
    __mirk_reset_device_cache!(cache)
    (abstol, adaptive, controller, _), _ = __split_kwargs(; cache.kwargs...)

    # Keep the first iteration outside the loop to preserve the type of the
    # nonlinear solution stored in the final solution's `original` field.
    sol_nlprob, info, error_norm = __perform_mirk_iteration(cache, abstol, adaptive, controller)
    if adaptive
        while successful_retcode(info) && error_norm > abstol
            sol_nlprob, info, error_norm = __perform_mirk_iteration(cache, abstol, adaptive, controller)
        end
    end
    return __mirk_build_solution(cache, sol_nlprob, info)
end

# The same cache expansion is used by mesh refinement and repeated solves.
function __expand_cache!(cache::MIRKCache, ::Val{:jacobian})
    if cache.jacobian_cache === nothing
        resize!(cache.jac_prototype, length(cache.residual) * length(cache.y))
    else
        jacobian = __mirk_prepare_device_jacobian(
            cache.prob, cache.alg, __mirk_states(cache), cache.host_mesh, cache.TU, cache.ITU,
            cache.resid_size, cache.p, cache.in_size
        )
        cache.jacobian_cache[nothing] = jacobian
    end
    return cache
end

__mirk_reset_device_cache!(cache::MIRKCache) = nothing
function __mirk_reset_device_cache!(
        cache::MIRKCache{iip, T, U, D, P, Y}
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    return __mirk_reset_device_cache!(cache, __mirk_jacobian_plan(cache))
end
__mirk_reset_device_cache!(cache::MIRKCache, ::Nothing) = nothing
function __mirk_reset_device_cache!(cache::MIRKCache, ::SparseJacobianCache)
    # Parameters can change boundary evaluation times when solve! reuses a cache.
    __expand_cache!(cache, Val(:jacobian))
    return nothing
end

function __mirk_build_solution(
        cache::MIRKCache{iip, T, use_both, diffcache, tune_parameters}, sol_nlprob, info
    ) where {iip, T, use_both, diffcache, tune_parameters}
    prob = cache.prob
    length_u = cache.in_size

    # Parameter estimation, put the estimated parameters to sol.prob.p
    if tune_parameters && SciMLStructures.isscimlstructure(prob.p)
        tunable_part, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
        length_u = cache.M - length(tunable_part)
        new_p = repack(cache.y₀.u[1][(length_u + 1):end])
        prob = remake(prob; p = new_p)
        foreach(x -> resize!(x, length_u), cache.y₀.u)
        resize!(cache.fᵢ₂_cache, length_u)
    elseif tune_parameters
        length_u = cache.M - length(prob.p)
        prob = remake(prob; p = cache.y₀.u[1][(length_u + 1):end])
        foreach(x -> resize!(x, length_u), cache.y₀.u)
        resize!(cache.fᵢ₂_cache, length_u)
    end

    u = recursivecopy(cache.y₀)

    interpolation = __build_interpolation(cache, u.u)

    odesol = SciMLBase.build_solution(
        prob, cache.alg, cache.mesh, u.u; interp = interpolation, retcode = info
    )
    return __build_solution(prob, odesol, sol_nlprob)
end

function __mirk_build_solution(
        cache::MIRKCache{iip, T, U, D, P, Y}, nlsol, retcode
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    # Preserve the state shape and device storage in the SciML solution.
    y, k, ki = copy(__mirk_states(cache)), copy(__mirk_stages(cache)), copy(__mirk_interp_stages(cache))
    prob = cache.prob
    in_size = cache.in_size
    if P
        nstates = cache.M - length(cache.p)
        prob = remake(prob; p = copy(view(y, (nstates + 1):cache.M, 1)))
        y = copy(view(y, 1:nstates, :))
        k = copy(view(k, 1:nstates, :, :))
        ki = copy(view(ki, 1:nstates, :, :))
        in_size = (nstates,)
    end
    values = [reshape(copy(view(y, :, i)), in_size) for i in axes(y, 2)]
    interp = __build_interpolation(
        y, k, ki, copy(cache.mesh), copy(cache.mesh_dt),
        Val(nameof(typeof(cache.alg))), in_size, cache.alg.platform
    )
    odesol = SciMLBase.build_solution(
        prob, cache.alg, copy(cache.host_mesh), values; interp, retcode
    )
    return __build_solution(prob, odesol, nlsol)
end

function __perform_mirk_iteration(cache::MIRKCache, abstol, adaptive::Bool, controller::AbstractErrorControl)
    # Refresh the flat mirror from the structured guess (in-place; no fresh allocation
    # per outer iteration). NonlinearSolve / LinearSolve still see a `Vector{T}`.
    copyto!(cache.y₀_flat, vec(cache.y₀))
    nlprob = __construct_problem(cache, cache.y₀_flat, copy(cache.y₀))
    solve_alg = __concrete_solve_algorithm(nlprob, cache.alg.nlsolve, cache.alg.optimize)
    kwargs = __concrete_kwargs(
        cache.alg.nlsolve, cache.alg.optimize, cache.nlsolve_kwargs, cache.optimize_kwargs,
        cache.verbose
    )
    sol_nlprob = __internal_solve(nlprob, solve_alg; kwargs...)
    recursive_unflatten!(cache.y₀, sol_nlprob.u)

    error_norm = 2 * abstol

    # Early terminate if non-adaptive
    adaptive || return sol_nlprob, sol_nlprob.retcode, error_norm

    info::ReturnCode.T = sol_nlprob.retcode

    if info == ReturnCode.Success # Nonlinear Solve was successful
        error_norm,
            info = error_estimate!(
            cache, controller, cache.errors, sol_nlprob, solve_alg, abstol
        )
    end

    if info == ReturnCode.Success # Nonlinear Solve Successful and defect norm is acceptable
        if error_norm > abstol
            # We construct a new mesh to equidistribute the defect
            mesh, mesh_dt, _, info = mesh_selector!(cache, controller)
            if info == ReturnCode.Success
                (length(mesh) < length(cache.mesh)) &&
                    __resize!(cache.y₀.u, length(cache.mesh), cache.M)
                for (i, m) in enumerate(cache.mesh)
                    interp_eval!(cache.y₀.u[i], cache, m, mesh, mesh_dt)
                end
                __expand_cache!(cache)
            end
        end
    else # Something bad happened
        # We cannot obtain a solution for the current mesh
        if 2 * (length(cache.mesh) - 1) > cache.alg.max_num_subintervals
            # New mesh would be too large
            info = ReturnCode.Failure
        else
            half_mesh!(cache)
            __expand_cache!(cache)
            recursivefill!(cache.y₀, 0)
            info = ReturnCode.Success # Force a restart
        end
    end

    return sol_nlprob, info, error_norm
end

# Device storage uses the same solve loop, with its own residual/Jacobian
# construction and device error estimates.
function __perform_mirk_iteration(
        cache::MIRKCache{iip, T, U, D, P, Y}, abstol, adaptive::Bool, controller::AbstractErrorControl
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    nlprob = __construct_problem(cache, copy(cache.y))
    solve_alg = __concrete_solve_algorithm(nlprob, cache)
    kwargs = __concrete_kwargs(
        cache.alg.nlsolve, cache.alg.optimize, cache.nlsolve_kwargs, cache.optimize_kwargs,
        cache.verbose
    )
    sol_nlprob = __internal_solve(nlprob, solve_alg; kwargs...)
    copyto!(cache.y, sol_nlprob.u)
    __device_residual!(cache.residual, cache.y, cache)
    __mirk_device_interp_setup!(
        cache.alg.platform, __mirk_interp_stages(cache), __mirk_collocation(cache), __mirk_stages(cache),
        __mirk_states(cache), cache.f, cache.p,
        cache.mesh, cache.mesh_dt, cache.ITU, cache.in_size, Val(iip), cache.singular_term
    )
    info = sol_nlprob.retcode
    error_norm = zero(abstol)
    if adaptive && successful_retcode(info)
        error_norm, estimate_info = error_estimate!(
            cache, controller, cache.errors, sol_nlprob, cache.alg.nlsolve, abstol
        )
        if estimate_info == ReturnCode.Failure && isfinite(error_norm) && error_norm > abstol
            # A large defect requires bisection before attempting redistribution.
            if 2 * length(cache.mesh_dt) > cache.alg.max_num_subintervals
                info = ReturnCode.Failure
            else
                half_mesh!(cache)
            end
        elseif !successful_retcode(estimate_info)
            info = estimate_info
        elseif error_norm > abstol
            _, _, _, info = mesh_selector!(cache, controller)
        end
    end
    return sol_nlprob, info, error_norm
end

# Boundary Residuals
@kernel function __mirk_device_bc_kernel!(
        resid, bc, y, k, ki, p, mesh, mesh_dt, algid, in_size, bc_sizes, iip, twopoint, tune_parameters
    )
    @inbounds begin
        left = prod(bc_sizes[1])
        parameters = tune_parameters isa Val{true} ?
            view(y, (size(y, 1) - length(p) + 1):size(y, 1), 1) : p
        if twopoint isa Val{true}
            right = prod(bc_sizes[2])
            __device_eval!(
                __device_reshape(view(resid, 1:left), bc_sizes[1]), bc[1],
                (__device_reshape(view(y, :, 1), in_size), parameters), iip
            )
            __device_eval!(
                __device_reshape(
                    view(resid, (length(resid) - right + 1):length(resid)), bc_sizes[2]
                ),
                bc[2], (
                    __device_reshape(view(y, :, size(y, 2)), in_size),
                    tune_parameters isa Val{true} ? view(y, (size(y, 1) - length(p) + 1):size(y, 1), size(y, 2)) : p,
                ), iip
            )
        else
            sol = EvalSol(__build_interpolation(y, k, ki, mesh, mesh_dt, algid, in_size))
            __device_eval!(
                __device_reshape(view(resid, 1:left), bc_sizes[1]),
                bc, (sol, parameters, mesh), iip
            )
        end
    end
end

# Collocation and Boundary Residual Assembly

function __mirk_device_buffers(
        cache::MIRKCache{iip, T, U, D, P, Y}, ::Type{S}
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}, S}
    return get!(cache.device_cache, S) do
        if S === eltype(cache.y)
            (;
                k = __mirk_stages(cache), ki = __mirk_interp_stages(cache),
                tmp = __mirk_collocation(cache), residual = cache.residual,
            )
        else
            (;
                k = similar(__mirk_stages(cache), S), ki = similar(__mirk_interp_stages(cache), S),
                tmp = similar(__mirk_collocation(cache), S), residual = similar(cache.residual, S),
            )
        end
    end
end

function BoundaryValueDiffEqCore.__device_residual!(
        resid, u, cache::MIRKCache{iip, T, U, D, P, Y}, boundary = true
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    work = __mirk_device_buffers(cache, eltype(u))
    y = __reshape_buffer(u, cache.M, length(cache.mesh))
    M, nodes = size(y)
    left = prod(cache.resid_size[1])
    collocation = reshape(view(resid, (left + 1):(left + M * (nodes - 1))), M, nodes - 1)
    (; c, v, x, b) = cache.TU
    __mirk_packed_collocation_kernel!(cache.alg.platform)(
        collocation, work.tmp, work.k, cache.f, y, cache.p, cache.mesh,
        cache.mesh_dt, c, v, x, b, cache.singular_term, cache.in_size,
        cache.in_size, P ? length(cache.p) : 0, Val(iip), Val(false), Val(P),
        cache.mass_matrix, cache.algebraic_indices; ndrange = nodes - 1
    )
    synchronize(cache.alg.platform)
    if boundary
        if !(cache.problem_type isa TwoPointBVProblem)
            __mirk_device_interp_setup!(
                cache.alg.platform, work.ki, work.tmp, work.k, y, cache.f, cache.p,
                cache.mesh, cache.mesh_dt, cache.ITU, cache.in_size,
                Val(iip), cache.singular_term
            )
        end
        __mirk_device_bc_kernel!(cache.alg.platform)(
            resid, cache.bc, y, work.k, work.ki, cache.p, cache.mesh, cache.mesh_dt,
            Val(nameof(typeof(cache.alg))), cache.in_size, cache.resid_size,
            Val(iip), Val(cache.problem_type isa TwoPointBVProblem), Val(P); ndrange = 1
        )
        synchronize(cache.alg.platform)
    end
    return resid
end

# Constructing the Nonlinear Problem
function __construct_problem(
        cache::MIRKCache{iip, T, U, D, P, Y}, u0::AbstractVector
    ) where {iip, T, U, D, P, Y <: AbstractVector{<:Number}}
    loss! = (r, u, p) -> __device_residual!(r, u, cache)
    jac! = (J, u, p) -> __device_jacobian!(J, u, cache)
    nf = SciMLBase.NonlinearFunction{true}(
        loss!; jac = jac!,
        __device_jacobian_products(cache)...,
        resid_prototype = cache.residual, jac_prototype = __mirk_jacobian(cache)
    )
    return BoundaryValueDiffEqCore.__internal_nlsolve_problem(
        cache.prob, cache.residual, u0, nf, u0, cache.p
    )
end

function __construct_problem(cache::MIRKCache{iip}, y::AbstractVector, y₀::AbstractVectorOfArray) where {iip}
    constraint = (!isnothing(cache.prob.f.inequality)) ||
        (!isnothing(cache.prob.f.equality)) ||
        (!isnothing(cache.prob.lb)) ||
        (!isnothing(cache.prob.ub))
    return __construct_problem(cache, y, y₀, Val(constraint))
end

function __construct_problem(
        cache::MIRKCache{iip}, y::AbstractVector,
        y₀::AbstractVectorOfArray, constraint
    ) where {iip}
    pt = cache.problem_type
    (; jac_alg) = cache.alg

    eval_sol = EvalSol(__restructure_sol(y₀.u, cache.in_size), cache.mesh, cache)

    trait = __cache_trait(jac_alg)

    loss_bc = if iip
        @closure (
            du,
            u,
            p,
        ) -> __mirk_loss_bc!(du, u, p, pt, cache.bc, cache.y, cache.mesh, cache, trait)
    else
        @closure (
            u, p,
        ) -> __mirk_loss_bc(u, p, pt, cache.bc, cache.y, cache.mesh, cache, trait)
    end

    loss_collocation = if iip
        @closure (
            du,
            u,
            p,
        ) -> __mirk_loss_collocation!(
            du, u, p, cache.y, cache.mesh, cache.residual, cache, trait, constraint
        )
    else
        @closure (
            u,
            p,
        ) -> __mirk_loss_collocation(
            u, p, cache.y, cache.mesh, cache.residual, cache, trait
        )
    end

    loss = if iip
        @closure (
            du,
            u,
            p,
        ) -> __mirk_loss!(
            du, u, p, cache.y, pt, cache.bc, cache.residual,
            cache.mesh, cache, eval_sol, trait, constraint
        )
    else
        @closure (
            u,
            p,
        ) -> __mirk_loss(
            u, p, cache.y, pt, cache.bc, cache.mesh, cache, eval_sol, trait
        )
    end

    if !isnothing(cache.alg.optimize)
        loss = @closure (
            du,
            u,
            p,
        ) -> __mirk_loss!(
            du, u, p, cache.y, pt, cache.bc, cache.residual,
            cache.bcresid_prototype, cache.mesh, cache, eval_sol, trait, constraint
        )
    end

    return __construct_problem(cache, y, loss_bc, loss_collocation, loss, pt, constraint)
end

@views function __mirk_loss!(
        resid, u, p, y, pt::StandardBVProblem, bc!::BC, residual, mesh,
        cache, eval_sol, trait::DiffCacheNeeded, constraint
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    Φ!(resids[2:end], cache, y_, u, trait, constraint)
    eval_sol = update_eval_sol!(eval_sol, y_, cache)
    eval_bc_residual!(resids[1], pt, bc!, eval_sol, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirk_loss!(
        resid, u, p, y, pt::StandardBVProblem, bc!::BC, residual, mesh,
        cache, eval_sol, trait::NoDiffCacheNeeded, constraint
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    Φ!(residual[2:end], cache, y_, u, trait, constraint)
    eval_sol = update_eval_sol!(eval_sol, y_, cache)
    eval_bc_residual!(residual[1], pt, bc!, eval_sol, p, mesh)
    recursive_flatten!(resid, residual)
    return nothing
end

# loss function for optimization based solvers
@views function __mirk_loss!(
        resid, u, p, y, pt::StandardBVProblem, bc!::BC, residual,
        bcresid_prototype, mesh, cache, _, trait, constraint
    ) where {BC}
    bcresid = length(bcresid_prototype)
    __mirk_loss_bc!(resid[1:bcresid], u, p, pt, bc!, y, mesh, cache, trait)
    __mirk_loss_collocation!(
        resid[(bcresid + 1):end], u, p, y, mesh, residual, cache, trait, constraint
    )
    return nothing
end

@views function __mirk_loss!(
        resid, u, p, y, pt::TwoPointBVProblem, bc!::Tuple{BC1, BC2}, residual,
        mesh, cache, _, trait::DiffCacheNeeded, constraint
    ) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    Φ!(resids[2:end], cache, y_, u, trait, constraint)
    resida = resids[1][1:prod(cache.resid_size[1])]
    residb = resids[1][(prod(cache.resid_size[1]) + 1):end]
    eval_bc_residual!((resida, residb), pt, bc!, y_, p, mesh)
    recursive_flatten_twopoint!(resid, resids, cache.resid_size)
    return nothing
end

@views function __mirk_loss!(
        resid, u, p, y, pt::TwoPointBVProblem, bc!::Tuple{BC1, BC2}, residual,
        mesh, cache, _, trait::NoDiffCacheNeeded, constraint
    ) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    Φ!(residual[2:end], cache, y_, u, trait, constraint)
    resida = residual[1][1:prod(cache.resid_size[1])]
    residb = residual[1][(prod(cache.resid_size[1]) + 1):end]
    eval_bc_residual!((resida, residb), pt, bc!, y_, p, mesh)
    recursive_flatten_twopoint!(resid, residual, cache.resid_size)
    return nothing
end

# loss function for optimization based solvers
@views function __mirk_loss!(
        resid, u, p, y, pt::TwoPointBVProblem, bc!::Tuple{BC1, BC2}, residual,
        bcresid_prototype, mesh, cache, _, trait, constraint
    ) where {BC1, BC2}
    __mirk_loss!(resid, u, p, y, pt, bc!, residual, mesh, cache, nothing, trait, constraint)
    return nothing
end

@views function __mirk_loss(
        u, p, y, pt::StandardBVProblem, bc::BC, mesh, cache, eval_sol, trait
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    resid_co = Φ(cache, y_, u, trait)
    eval_sol = update_eval_sol!(eval_sol, y_, cache)
    resid_bc = eval_bc_residual(pt, bc, eval_sol, p, mesh)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end

@views function __mirk_loss(
        u, p, y, pt::TwoPointBVProblem, bc::Tuple{BC1, BC2},
        mesh, cache, _, trait
    ) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    resid_co = Φ(cache, y_, u, trait)
    resid_bca, resid_bcb = eval_bc_residual(pt, bc, y_, p, mesh)
    return vcat(resid_bca, mapreduce(vec, vcat, resid_co), resid_bcb)
end

@views function __mirk_loss_bc!(
        resid, u, p, pt, bc!::BC, y, mesh, cache::MIRKCache, trait
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    soly_ = EvalSol(__restructure_sol(y_, cache.in_size), mesh, cache)
    eval_bc_residual!(resid, pt, bc!, soly_, p, mesh)
    return nothing
end

@views function __mirk_loss_bc(
        u, p, pt, bc!::BC, y, mesh, cache::MIRKCache, trait
    ) where {BC}
    y_ = recursive_unflatten!(y, u)
    soly_ = EvalSol(__restructure_sol(y_, cache.in_size), mesh, cache)
    return eval_bc_residual(pt, bc!, soly_, p, mesh)
end

@views function __mirk_loss_collocation!(
        resid, u, p, y, mesh, residual, cache, trait::DiffCacheNeeded, constraint
    )
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual[2:end]]
    Φ!(resids, cache, y_, u, trait, constraint)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirk_loss_collocation!(
        resid, u, p, y, mesh, residual, cache, trait::NoDiffCacheNeeded, constraint
    )
    y_ = recursive_unflatten!(y, u)
    resids = [r for r in residual[2:end]]
    Φ!(resids, cache, y_, u, trait, constraint)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirk_loss_collocation(u, p, y, mesh, residual, cache, trait)
    y_ = recursive_unflatten!(y, u)
    resids = Φ(cache, y_, u, trait)
    return mapreduce(vec, vcat, resids)
end

function __construct_problem(
        cache::MIRKCache{iip, T, UB, DC, tune_parameters}, y, loss_bc::BC, loss_collocation::C, loss::LF,
        ::StandardBVProblem, constraint::Val{true}
    ) where {iip, T, UB, DC, tune_parameters, BC, C, LF}
    (; jac_alg) = cache.alg
    (; f_prototype, bcresid_prototype, prob) = cache
    (; bc_diffmode) = jac_alg
    N = length(cache.mesh)

    resid_bc = bcresid_prototype
    L = length(resid_bc)
    L_f_prototype = length(f_prototype)
    resid_collocation = safe_similar(y, L_f_prototype * (N - 1))

    cache_bc = if iip
        DI.prepare_jacobian(
            loss_bc, resid_bc, bc_diffmode, y, Constant(cache.p); strict = Val(false)
        )
    else
        DI.prepare_jacobian(
            loss_bc, bc_diffmode, y, Constant(cache.p); strict = Val(false)
        )
    end

    nonbc_diffmode = AutoSparse(
        get_dense_ad(jac_alg.nonbc_diffmode),
        sparsity_detector = __default_sparsity_detector(jac_alg.nonbc_diffmode),
        coloring_algorithm = __default_coloring_algorithm(jac_alg.nonbc_diffmode)
    )
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
        ) -> __mirk_mpoint_jacobian!(
            J, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc, cache_collocation,
            loss_bc, loss_collocation, resid_bc, resid_collocation, L, cache.p
        )
    else
        @closure (
            u,
            p,
        ) -> __mirk_mpoint_jacobian(
            jac_prototype, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc,
            cache_collocation, loss_bc, loss_collocation, L, cache.p
        )
    end

    cost_fun = __build_cost(
        prob.f.cost, cache, cache.mesh, cache.M;
        tune_parameters, cache.p
    )

    resid_prototype = vcat(resid_bc, resid_collocation)
    return __construct_internal_problem(
        prob, cache.problem_type, cache.alg, loss, jac, jac_prototype, resid_prototype,
        bcresid_prototype, f_prototype, y, cache.p, cache.M, N, cost_fun
    )
end

# Dispatch for problems with constraints
function __construct_problem(
        cache::MIRKCache{iip, T, UB, DC, tune_parameters}, y, loss_bc::BC, loss_collocation::C, loss::LF,
        ::StandardBVProblem, constraint::Val{false}
    ) where {iip, T, UB, DC, tune_parameters, BC, C, LF}
    (; jac_alg) = cache.alg
    (; f_prototype, bcresid_prototype, prob) = cache
    (; bc_diffmode) = jac_alg
    N = length(cache.mesh)

    resid_bc = bcresid_prototype
    L = length(resid_bc)
    resid_collocation = safe_similar(y, cache.M * (N - 1))
    resid_prototype = vcat(resid_bc, resid_collocation)

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
        if L < cache.M
            # For underdetermined problems we use sparse since we don't have banded qr
            J_full_band = nothing
            sparse_jacobian_prototype = __generate_sparse_jacobian_prototype(
                cache, cache.problem_type, y, y, cache.M, N
            )
        else
            J_full_band = BandedMatrix(
                Ones{eltype(y)}(L + cache.M * (N - 1), cache.M * N),
                (L + 1, cache.M + max(cache.M - L, 0))
            )
            sparse_jacobian_prototype = __generate_sparse_jacobian_prototype(
                cache, cache.problem_type, y, y, cache.M, N
            )
        end
        AutoSparse(
            get_dense_ad(jac_alg.nonbc_diffmode);
            sparsity_detector = ADTypes.KnownJacobianSparsityDetector(sparse_jacobian_prototype),
            coloring_algorithm = __default_coloring_algorithm(jac_alg.nonbc_diffmode)
        )
    else
        J_full_band = nothing
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

    if J_full_band === nothing
        jac_prototype = vcat(J_bc, J_c)
    else
        # Keep sparse AD/coloring, but store the small boundary block densely.
        # Almost-banded QR applies dense updates to this block and its factors;
        # a sparse container makes those updates and triangular solves expensive.
        # DI can decompress the sparse Jacobian directly into this dense block.
        jac_prototype = AlmostBandedMatrix{eltype(cache)}(J_full_band, Matrix(J_bc))
    end

    jac = if iip
        @closure (
            J,
            u,
            p,
        ) -> __mirk_mpoint_jacobian!(
            J, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc, cache_collocation,
            loss_bc, loss_collocation, resid_bc, resid_collocation, L, cache.p
        )
    else
        @closure (
            u,
            p,
        ) -> __mirk_mpoint_jacobian(
            jac_prototype, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc,
            cache_collocation, loss_bc, loss_collocation, L, cache.p
        )
    end

    cost_fun = __build_cost(
        prob.f.cost, cache, cache.mesh, cache.M;
        tune_parameters, cache.p
    )

    return __construct_internal_problem(
        prob, cache.problem_type, cache.alg, loss, jac, jac_prototype, resid_prototype,
        bcresid_prototype, f_prototype, y, cache.p, cache.M, N, cost_fun
    )
end

function __mirk_mpoint_jacobian!(
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

function __mirk_mpoint_jacobian!(
        J::AlmostBandedMatrix, J_c, x, bc_diffmode, nonbc_diffmode,
        bc_diffcache, nonbc_diffcache, loss_bc::BC, loss_collocation::C,
        resid_bc, resid_collocation, L::Int, p
    ) where {BC, C}
    J_bc = fillpart(J)
    DI.jacobian!(
        loss_collocation, resid_collocation, J_c,
        nonbc_diffcache, nonbc_diffmode, x, Constant(p)
    )
    DI.jacobian!(loss_bc, resid_bc, J_bc, bc_diffcache, bc_diffmode, x, Constant(p))
    exclusive_bandpart(J) .= J_c
    finish_part_setindex!(J)
    return nothing
end

function __mirk_mpoint_jacobian(
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

function __mirk_mpoint_jacobian(
        J::AlmostBandedMatrix, J_c, x, bc_diffmode, nonbc_diffmode, bc_diffcache,
        nonbc_diffcache, loss_bc::BC, loss_collocation::C, L::Int, p
    ) where {BC, C}
    J_bc = fillpart(J)
    DI.jacobian!(loss_bc, J_bc, bc_diffcache, bc_diffmode, x, Constant(p))
    DI.jacobian!(loss_collocation, J_c, nonbc_diffcache, nonbc_diffmode, x, Constant(p))
    exclusive_bandpart(J) .= J_c
    finish_part_setindex!(J)
    return J
end

function __construct_problem(
        cache::MIRKCache{iip, T, UB, DC, tune_parameters}, y, loss_bc::BC, loss_collocation::C, loss::LF,
        ::TwoPointBVProblem, constraint::Val{true}
    ) where {iip, T, UB, DC, tune_parameters, BC, C, LF}
    (; jac_alg) = cache.alg
    (; f_prototype, bcresid_prototype, prob) = cache
    N = length(cache.mesh)
    L_f_prototype = length(f_prototype)

    resid = vcat(
        @view(bcresid_prototype[1:prod(cache.resid_size[1])]),
        safe_similar(y, L_f_prototype * (N - 1)),
        @view(bcresid_prototype[(prod(cache.resid_size[1]) + 1):end])
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
        ) -> __mirk_2point_jacobian!(J, u, diffmode, diffcache, loss, resid, p)
    else
        @closure (
            u, p,
        ) -> __mirk_2point_jacobian(u, jac_prototype, diffmode, diffcache, loss, p)
    end

    cost_fun = __build_cost(
        prob.f.cost, cache, cache.mesh, cache.M;
        tune_parameters, cache.p
    )

    resid_prototype = copy(resid)
    return __construct_internal_problem(
        prob, cache.problem_type, cache.alg, loss, jac, jac_prototype, resid_prototype,
        bcresid_prototype, f_prototype, y, cache.p, cache.M, N, cost_fun
    )
end

function __construct_problem(
        cache::MIRKCache{iip, T, UB, DC, tune_parameters}, y, loss_bc::BC, loss_collocation::C, loss::LF,
        ::TwoPointBVProblem, constraint::Val{false}
    ) where {iip, T, UB, DC, tune_parameters, BC, C, LF}
    (; jac_alg) = cache.alg
    (; f_prototype, bcresid_prototype, prob) = cache
    N = length(cache.mesh)

    resid = vcat(
        @view(bcresid_prototype[1:prod(cache.resid_size[1])]),
        safe_similar(y, cache.M * (N - 1)),
        @view(bcresid_prototype[(prod(cache.resid_size[1]) + 1):end])
    )

    diffmode = if jac_alg.diffmode isa AutoSparse
        sparse_jacobian_prototype = __generate_sparse_jacobian_prototype(
            cache, cache.problem_type,
            @view(bcresid_prototype[1:prod(cache.resid_size[1])]),
            @view(bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]), cache.M, N
        )
        AutoSparse(
            get_dense_ad(jac_alg.diffmode);
            sparsity_detector = ADTypes.KnownJacobianSparsityDetector(sparse_jacobian_prototype),
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
        ) -> __mirk_2point_jacobian!(J, u, diffmode, diffcache, loss, resid, p)
    else
        @closure (
            u, p,
        ) -> __mirk_2point_jacobian(u, jac_prototype, diffmode, diffcache, loss, p)
    end

    cost_fun = __build_cost(
        prob.f.cost, cache, cache.mesh, cache.M;
        tune_parameters, cache.p
    )

    resid_prototype = copy(resid)
    return __construct_internal_problem(
        cache.prob, cache.problem_type, cache.alg, loss, jac, jac_prototype,
        resid_prototype, bcresid_prototype, f_prototype, y, cache.p, cache.M, N, cost_fun
    )
end

function __mirk_2point_jacobian!(J, x, diffmode, diffcache, loss_fn::L, resid, p) where {L}
    DI.jacobian!(loss_fn, resid, J, diffcache, diffmode, x, Constant(p))
    return J
end

function __mirk_2point_jacobian(x, J, diffmode, diffcache, loss_fn::L, p) where {L}
    DI.jacobian!(loss_fn, J, diffcache, diffmode, x, Constant(p))
    return J
end

BoundaryValueDiffEqCore.__bvp_device_unknowns(cache::MIRKCache) = cache.y
