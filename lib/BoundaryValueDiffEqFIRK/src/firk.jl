@concrete struct FIRKCacheNested{iip, T}
    order::Int                 # The order of MIRK method
    stage::Int                 # The state of MIRK method
    M::Int                     # The number of equations
    in_size
    f
    bc
    prob                       # BVProblem
    problem_type               # StandardBVProblem
    p                          # Parameters
    alg                        # FIRK methods
    TU                         # FIRK Tableau
    ITU                        # FIRK Interpolation Tableau
    bcresid_prototype
    # Everything below gets resized in adaptive methods
    mesh                       # Discrete mesh
    mesh_dt                    # Step size
    k_discrete                 # Stage information associated with the discrete Runge-Kutta method
    y
    y₀
    residual
    # The following 2 caches are never resized
    fᵢ_cache
    fᵢ₂_cache
    defect
    nest_prob
    nest_tol
    resid_size
    kwargs
end

Base.eltype(::FIRKCacheNested{iip, T}) where {iip, T} = T
@concrete struct FIRKCacheExpand{iip, T}
    order::Int                 # The order of MIRK method
    stage::Int                 # The state of MIRK method
    M::Int                     # The number of equations
    in_size
    f
    bc
    prob                       # BVProblem
    problem_type               # StandardBVProblem
    p                          # Parameters
    alg                        # FIRK methods
    TU                         # FIRK Tableau
    ITU                        # FIRK Interpolation Tableau
    bcresid_prototype
    # Everything below gets resized in adaptive methods
    mesh                       # Discrete mesh
    mesh_dt                    # Step size
    k_discrete                 # Stage information associated with the discrete Runge-Kutta method
    y
    y₀
    residual
    # The following 2 caches are never resized
    fᵢ_cache
    fᵢ₂_cache
    defect
    resid_size
    kwargs
end
Base.eltype(::FIRKCacheExpand{iip, T}) where {iip, T} = T

function extend_y(y, N::Int, stage::Int)
    y_extended = similar(y.u, (N - 1) * (stage + 1) + 1)
    for (i, ctr) in enumerate(2:(stage + 1):((N - 1) * (stage + 1) + 1))
        @views fill!(y_extended[ctr:(ctr + stage)], y.u[i + 1])
    end
    y_extended[1] = y.u[1]
    return VectorOfArray(y_extended)
end

function shrink_y(y, N, stage)
    y_shrink = similar(y, N)
    for (i, ctr) in enumerate((stage + 2):(stage + 1):((N - 1) * (stage + 1) + 1))
        y_shrink[i + 1] = y[ctr]
    end
    y_shrink[1] = y[1]
    return y_shrink
end

function SciMLBase.__init(prob::BVProblem, alg::AbstractFIRK; dt = 0.0,
        abstol = 1e-3, adaptive = true, kwargs...)
    if alg.nested_nlsolve
        return init_nested(
            prob, alg; dt = dt, abstol = abstol, adaptive = adaptive, kwargs...)
    else
        return init_expanded(
            prob, alg; dt = dt, abstol = abstol, adaptive = adaptive, kwargs...)
    end
end

function init_nested(prob::BVProblem, alg::AbstractFIRK; dt = 0.0,
        abstol = 1e-3, adaptive = true, kwargs...)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)

    iip = isinplace(prob)
    if adaptive && isa(alg, FIRKNoAdaptivity)
        error("Algorithm doesn't support adaptivity. Please choose a higher order algorithm.")
    end

    t₀, t₁ = prob.tspan
    ig, T, M, Nig, X = __extract_problem_details(prob; dt, check_positive_dt = true)
    mesh = __extract_mesh(prob.u0, t₀, t₁, Nig)
    mesh_dt = diff(mesh)

    chunksize = pickchunksize(M * (Nig - 1))
    __alloc = @closure x -> __maybe_allocate_diffcache(vec(x), chunksize, alg.jac_alg)

    fᵢ_cache = __alloc(zero(X))
    fᵢ₂_cache = vec(zero(X))

    # Don't flatten this here, since we need to expand it later if needed
    y₀ = __initial_guess_on_mesh(prob.u0, mesh, prob.p)

    y = __alloc.(copy.(y₀.u))
    TU, ITU = constructRK(alg, T)
    stage = alg_stage(alg)

    k_discrete = [__maybe_allocate_diffcache(
                      __similar(X, M, stage), chunksize, alg.jac_alg) for _ in 1:Nig]

    bcresid_prototype, resid₁_size = __get_bcresid_prototype(prob.problem_type, prob, X)

    residual = if iip
        if prob.problem_type isa TwoPointBVProblem
            vcat([__alloc(__vec(bcresid_prototype))], __alloc.(copy.(@view(y₀.u[2:end]))))
        else
            vcat([__alloc(bcresid_prototype)], __alloc.(copy.(@view(y₀.u[2:end]))))
        end
    else
        nothing
    end

    defect = VectorOfArray([__similar(X, ifelse(adaptive, M, 0)) for _ in 1:Nig])

    # Transform the functions to handle non-vector inputs
    bcresid_prototype = __vec(bcresid_prototype)
    f, bc = if X isa AbstractVector
        prob.f, prob.f.bc
    elseif iip
        vecf! = @closure (du, u, p, t) -> __vec_f!(du, u, p, t, prob.f, size(X))
        vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
            @closure (r, u, p, t) -> __vec_bc!(r, u, p, t, prob.f.bc, resid₁_size, size(X))
        else
            (
                @closure((r, u, p)->__vec_bc!(
                    r, u, p, first(prob.f.bc), resid₁_size[1], size(X))),
                @closure((r, u, p)->__vec_bc!(
                    r, u, p, last(prob.f.bc), resid₁_size[2], size(X))))
        end
        vecf!, vecbc!
    else
        vecf = @closure (u, p, t) -> __vec_f(u, p, t, prob.f, size(X))
        vecbc = if !(prob.problem_type isa TwoPointBVProblem)
            @closure (u, p, t) -> __vec_bc(u, p, t, prob.f.bc, size(X))
        else
            (@closure((u, p)->__vec_bc(u, p, first(prob.f.bc), size(X))),
                @closure((u, p)->__vec_bc(u, p, last(prob.f.bc), size(X))))
        end
        vecf, vecbc
    end

    prob_ = !(prob.u0 isa AbstractArray) ? remake(prob; u0 = X) : prob

    if isa(prob.u0, AbstractArray) && eltype(prob.u0) <: AbstractVector
        u0_mat = hcat(prob.u0...)
        avg_u0 = vec(sum(u0_mat, dims = 2)) / size(u0_mat, 2)
    else
        avg_u0 = prob.u0
    end

    K0 = repeat(avg_u0, 1, stage) # Somewhat arbitrary initialization of K

    nestprob_p = zeros(T, M + 2)
    nest_tol = alg.nest_tol

    if iip
        nestprob = NonlinearProblem(
            (res, K, p) -> FIRK_nlsolve!(res, K, p, f, TU, prob.p), K0, nestprob_p)
    else
        nestprob = NonlinearProblem(
            (K, p) -> FIRK_nlsolve(K, p, f, TU, prob.p), K0, nestprob_p)
    end

    return FIRKCacheNested{iip, T}(
        alg_order(alg), stage, M, size(X), f, bc, prob_, prob.problem_type,
        prob.p, alg, TU, ITU, bcresid_prototype, mesh, mesh_dt,
        k_discrete, y, y₀, residual, fᵢ_cache, fᵢ₂_cache, defect, nestprob,
        nest_tol, resid₁_size, (; abstol, dt, adaptive, kwargs...))
end

function init_expanded(prob::BVProblem, alg::AbstractFIRK; dt = 0.0,
        abstol = 1e-3, adaptive = true, kwargs...)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)

    if adaptive && isa(alg, FIRKNoAdaptivity)
        error("Algorithm $(alg) doesn't support adaptivity. Please choose a higher order algorithm.")
    end

    iip = isinplace(prob)

    t₀, t₁ = prob.tspan
    ig, T, M, Nig, X = __extract_problem_details(prob; dt, check_positive_dt = true)
    mesh = __extract_mesh(prob.u0, t₀, t₁, Nig)
    mesh_dt = diff(mesh)

    TU, ITU = constructRK(alg, T)
    stage = alg_stage(alg)

    chunksize = pickchunksize(M + M * Nig * (stage + 1))
    __alloc = @closure x -> __maybe_allocate_diffcache(vec(x), chunksize, alg.jac_alg)

    fᵢ_cache = __alloc(zero(X)) # Runtime dispatch
    fᵢ₂_cache = vec(zero(X))

    # Don't flatten this here, since we need to expand it later if needed
    _y₀ = __initial_guess_on_mesh(prob.u0, mesh, prob.p)
    y₀ = extend_y(_y₀, Nig + 1, stage)
    y = __alloc.(copy.(y₀.u)) # Runtime dispatch

    k_discrete = [__maybe_allocate_diffcache(
                      __similar(X, M, stage), chunksize, alg.jac_alg) for _ in 1:Nig] # Runtime dispatch

    bcresid_prototype, resid₁_size = __get_bcresid_prototype(prob.problem_type, prob, X)

    residual = if iip
        if prob.problem_type isa TwoPointBVProblem
            vcat([__alloc(__vec(bcresid_prototype))], __alloc.(copy.(@view(y₀.u[2:end]))))
        else
            vcat([__alloc(bcresid_prototype)], __alloc.(copy.(@view(y₀.u[2:end]))))
        end
    else
        nothing
    end

    defect = VectorOfArray([similar(X, ifelse(adaptive, M, 0)) for _ in 1:Nig])

    # Transform the functions to handle non-vector inputs
    bcresid_prototype = __vec(bcresid_prototype)
    f, bc = if X isa AbstractVector
        prob.f, prob.f.bc
    elseif iip
        vecf! = @closure (du, u, p, t) -> __vec_f!(du, u, p, t, prob.f, size(X))
        vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
            @closure (r, u, p, t) -> __vec_bc!(r, u, p, t, prob.f.bc, resid₁_size, size(X))
        else
            (
                @closure((r, u, p)->__vec_bc!(
                    r, u, p, first(prob.f.bc)[1], resid₁_size[1], size(X))),
                @closure ((r, u, p) -> __vec_bc!(
                    r, u, p, last(prob.f.bc)[2], resid₁_size[2], size(X))))
        end
        vecf!, vecbc!
    else
        vecf = @closure (u, p, t) -> __vec_f(u, p, t, prob.f, size(X))
        vecbc = if !(prob.problem_type isa TwoPointBVProblem)
            @closure (u, p, t) -> __vec_bc(u, p, t, prob.f.bc, size(X))
        else
            (@closure((u, p)->__vec_bc(u, p, first(prob.f.bc), size(X))),
                @closure((u, p)->__vec_bc(u, p, last(prob.f.bc), size(X))))
        end
        vecf, vecbc
    end

    prob_ = !(prob.u0 isa AbstractArray) ? remake(prob; u0 = X) : prob

    return FIRKCacheExpand{iip, T}(
        alg_order(alg), stage, M, size(X), f, bc, prob_, prob.problem_type, prob.p,
        alg, TU, ITU, bcresid_prototype, mesh, mesh_dt, k_discrete, y, y₀, residual,
        fᵢ_cache, fᵢ₂_cache, defect, resid₁_size, (; abstol, dt, adaptive, kwargs...))
end

"""
    __expand_cache!(cache::FIRKCache)

After redistributing or halving the mesh, this function expands the required vectors to
match the length of the new mesh.
"""
function __expand_cache!(cache::FIRKCacheExpand)
    Nₙ = length(cache.mesh)
    __append_similar!(cache.k_discrete, Nₙ - 1, cache.M, cache.TU)
    __append_similar!(cache.y, Nₙ, cache.M, cache.TU)
    __append_similar!(cache.y₀, Nₙ, cache.M, cache.TU)
    __append_similar!(cache.residual, Nₙ, cache.M, cache.TU)
    __append_similar!(cache.defect, Nₙ - 1, cache.M, cache.TU)
    return cache
end

function __expand_cache!(cache::FIRKCacheNested)
    Nₙ = length(cache.mesh)
    __append_similar!(cache.k_discrete, Nₙ - 1, cache.M)
    __append_similar!(cache.y, Nₙ, cache.M)
    __append_similar!(cache.y₀, Nₙ, cache.M)
    __append_similar!(cache.residual, Nₙ, cache.M)
    __append_similar!(cache.defect, Nₙ - 1, cache.M)
    return cache
end

function __split_mirk_kwargs(; abstol, dt, adaptive = true, kwargs...)
    return ((abstol, adaptive, dt), (; abstol, adaptive, kwargs...))
end

function SciMLBase.solve!(cache::FIRKCacheExpand)
    (abstol, adaptive, _), kwargs = __split_mirk_kwargs(; cache.kwargs...)
    info::ReturnCode.T = ReturnCode.Success

    # We do the first iteration outside the loop to preserve type-stability of the
    # `original` field of the solution
    sol_nlprob, info, defect_norm = __perform_firk_iteration(
        cache, abstol, adaptive; kwargs...)

    if adaptive
        while SciMLBase.successful_retcode(info) && defect_norm > abstol
            sol_nlprob, info, defect_norm = __perform_firk_iteration(
                cache, abstol, adaptive; kwargs...)
        end
    end

    u = shrink_y(
        [reshape(y, cache.in_size) for y in cache.y₀], length(cache.mesh), cache.stage)

    interpolation = __build_interpolation(cache, u)

    odesol = DiffEqBase.build_solution(
        cache.prob, cache.alg, cache.mesh, u; interp = interpolation, retcode = info)
    return __build_solution(cache.prob, odesol, sol_nlprob)
end

function SciMLBase.solve!(cache::FIRKCacheNested)
    (abstol, adaptive, _), kwargs = __split_mirk_kwargs(; cache.kwargs...)
    info::ReturnCode.T = ReturnCode.Success

    # We do the first iteration outside the loop to preserve type-stability of the
    # `original` field of the solution
    sol_nlprob, info, defect_norm = __perform_firk_iteration(
        cache, abstol, adaptive; kwargs...)

    if adaptive
        while SciMLBase.successful_retcode(info) && defect_norm > abstol
            sol_nlprob, info, defect_norm = __perform_firk_iteration(
                cache, abstol, adaptive; kwargs...)
        end
    end

    u = recursivecopy(cache.y₀)

    interpolation = __build_interpolation(cache, u.u)

    odesol = DiffEqBase.build_solution(
        cache.prob, cache.alg, cache.mesh, u.u; interp = interpolation, retcode = info)
    return __build_solution(cache.prob, odesol, sol_nlprob)
end

function __perform_firk_iteration(cache::Union{FIRKCacheExpand, FIRKCacheNested}, abstol,
        adaptive::Bool; nlsolve_kwargs = (;), kwargs...)
    nlprob = __construct_nlproblem(cache, vec(cache.y₀), copy(cache.y₀))
    nlsolve_alg = __concrete_nonlinearsolve_algorithm(nlprob, cache.alg.nlsolve)
    sol_nlprob = __solve(
        nlprob, nlsolve_alg; abstol, kwargs..., nlsolve_kwargs..., alias_u0 = true)
    recursive_unflatten!(cache.y₀, sol_nlprob.u)

    defect_norm = 2 * abstol

    # Early terminate if non-adaptive
    adaptive || return sol_nlprob, sol_nlprob.retcode, defect_norm

    info::ReturnCode.T = sol_nlprob.retcode

    if info == ReturnCode.Success # Nonlinear Solve was successful
        defect_norm = defect_estimate!(cache)
        # The defect is greater than 10%, the solution is not acceptable
        defect_norm > cache.alg.defect_threshold && (info = ReturnCode.Failure)
    end

    if info == ReturnCode.Success # Nonlinear Solve Successful and defect norm is acceptable
        if defect_norm > abstol
            # We construct a new mesh to equidistribute the defect
            mesh, mesh_dt, _, info = mesh_selector!(cache)
            if info == ReturnCode.Success
                __append_similar!(cache.y₀, length(cache.mesh), cache.M, cache.TU)
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

    return sol_nlprob, info, defect_norm
end

# Constructing the Nonlinear Problem
function __construct_nlproblem(cache::Union{FIRKCacheNested{iip}, FIRKCacheExpand{iip}},
        y::AbstractVector, y₀::AbstractVectorOfArray) where {iip}
    pt = cache.problem_type

    eval_sol = EvalSol(__restructure_sol(y₀.u, cache.in_size), cache.mesh, cache)

    loss_bc = if iip
        @closure (du, u, p) -> __firk_loss_bc!(
            du, u, p, pt, cache.bc, cache.y, cache.mesh, cache)
    else
        @closure (u, p) -> __firk_loss_bc(u, p, pt, cache.bc, cache.y, cache.mesh, cache)
    end

    loss_collocation = if iip
        @closure (du, u, p) -> __firk_loss_collocation!(
            du, u, p, cache.y, cache.mesh, cache.residual, cache)
    else
        @closure (u, p) -> __firk_loss_collocation(
            u, p, cache.y, cache.mesh, cache.residual, cache)
    end

    loss = if iip
        @closure (du, u, p) -> __firk_loss!(
            du, u, p, cache.y, pt, cache.bc, cache.residual, cache.mesh, cache, eval_sol)
    else
        @closure (u, p) -> __firk_loss(
            u, p, cache.y, pt, cache.bc, cache.mesh, cache, eval_sol)
    end

    return __construct_nlproblem(cache, y, loss_bc, loss_collocation, loss, pt)
end

function __construct_nlproblem(
        cache::FIRKCacheExpand{iip}, y, loss_bc::BC, loss_collocation::C,
        loss::LF, ::StandardBVProblem) where {iip, BC, C, LF}
    (; jac_alg) = cache.alg
    (; stage) = cache
    N = length(cache.mesh)

    resid_bc = cache.bcresid_prototype
    L = length(resid_bc)
    resid_collocation = __similar(y, cache.M * (N - 1) * (stage + 1))

    loss_bcₚ = (iip ? __Fix3 : Base.Fix2)(loss_bc, cache.p)
    loss_collocationₚ = (iip ? __Fix3 : Base.Fix2)(loss_collocation, cache.p)

    sd_bc = jac_alg.bc_diffmode isa AutoSparse ? SymbolicsSparsityDetection() :
            NoSparsityDetection()
    cache_bc = __sparse_jacobian_cache(
        Val(iip), jac_alg.bc_diffmode, sd_bc, loss_bcₚ, resid_bc, y)

    sd_collocation = if jac_alg.nonbc_diffmode isa AutoSparse
        if L < cache.M
            # For underdetermined problems we use sparse since we don't have banded qr
            colored_matrix = __generate_sparse_jacobian_prototype(
                cache, cache.problem_type, y, y, cache.M, N)
            J_full_band = nothing
            __sparsity_detection_alg(ColoredMatrix(
                sparse(colored_matrix.M), colored_matrix.row_colorvec,
                colored_matrix.col_colorvec))
        else
            block_size = cache.M * (stage + 2)
            J_full_band = BandedMatrix(
                Ones{eltype(y)}(L + cache.M * (stage + 1) * (N - 1),
                    cache.M * (stage + 1) * (N - 1) + cache.M),
                (block_size, block_size))
            __sparsity_detection_alg(__generate_sparse_jacobian_prototype(
                cache, cache.problem_type, y, y, cache.M, N))
        end
    else
        J_full_band = nothing
        NoSparsityDetection()
    end

    cache_collocation = __sparse_jacobian_cache(
        Val(iip), jac_alg.nonbc_diffmode, sd_collocation,
        loss_collocationₚ, resid_collocation, y)

    J_bc = zero(init_jacobian(cache_bc))
    J_c = zero(init_jacobian(cache_collocation))

    if J_full_band === nothing
        jac_prototype = vcat(J_bc, J_c)
    else
        jac_prototype = AlmostBandedMatrix{eltype(cache)}(J_full_band, J_bc)
    end

    jac = if iip
        @closure (J, u, p) -> __firk_mpoint_jacobian!(
            J, J_c, u, jac_alg.bc_diffmode, jac_alg.nonbc_diffmode, cache_bc,
            cache_collocation, loss_bcₚ, loss_collocationₚ, resid_bc, resid_collocation, L)
    else
        @closure (u, p) -> __firk_mpoint_jacobian(
            jac_prototype, J_c, u, jac_alg.bc_diffmode, jac_alg.nonbc_diffmode,
            cache_bc, cache_collocation, loss_bcₚ, loss_collocationₚ, L)
    end

    resid_prototype = vcat(resid_bc, resid_collocation)

    resid_prototype = vcat(resid_bc, resid_collocation)
    nlf = NonlinearFunction{iip}(
        loss; jac = jac, resid_prototype = resid_prototype, jac_prototype = jac_prototype)

    return __internal_nlsolve_problem(cache.prob, resid_prototype, y, nlf, y, cache.p)
end

function __construct_nlproblem(
        cache::FIRKCacheExpand{iip}, y, loss_bc::BC, loss_collocation::C,
        loss::LF, ::TwoPointBVProblem) where {iip, BC, C, LF}
    (; jac_alg) = cache.alg
    (; stage) = cache
    N = length(cache.mesh)

    lossₚ = iip ? ((du, u) -> loss(du, u, cache.p)) : (u -> loss(u, cache.p))

    resid_collocation = __similar(y, cache.M * (N - 1) * (stage + 1))

    resid = vcat(
        @view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]), resid_collocation,
        @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]))
    L = length(cache.bcresid_prototype)

    sd = if jac_alg.nonbc_diffmode isa AutoSparse
        block_size = cache.M * (stage + 2)
        J_full_band = BandedMatrix(
            Ones{eltype(y)}(L + cache.M * (stage + 1) * (N - 1),
                cache.M * (stage + 1) * (N - 1) + cache.M),
            (block_size, block_size))
        __sparsity_detection_alg(__generate_sparse_jacobian_prototype(
            cache, cache.problem_type,
            @view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
            @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]),
            cache.M, N))
    else
        J_full_band = nothing
        NoSparsityDetection()
    end

    diffcache = __sparse_jacobian_cache(Val(iip), jac_alg.diffmode, sd, lossₚ, resid, y)
    jac_prototype = zero(init_jacobian(diffcache))

    jac = if iip
        @closure (J, u, p) -> __firk_2point_jacobian!(
            J, u, jac_alg.diffmode, diffcache, lossₚ, resid)
    else
        @closure (u, p) -> __firk_2point_jacobian(
            u, jac_prototype, jac_alg.diffmode, diffcache, lossₚ)
    end

    resid_prototype = copy(resid)
    nlf = NonlinearFunction{iip}(
        loss; jac = jac, resid_prototype = resid_prototype, jac_prototype = jac_prototype)
    return __internal_nlsolve_problem(cache.prob, resid_prototype, y, nlf, y, cache.p)
end

function __construct_nlproblem(
        cache::FIRKCacheNested{iip}, y, loss_bc::BC, loss_collocation::C,
        loss::LF, ::StandardBVProblem) where {iip, BC, C, LF}
    (; jac_alg) = cache.alg
    N = length(cache.mesh)

    resid_bc = cache.bcresid_prototype
    L = length(resid_bc)
    resid_collocation = __similar(y, cache.M * (N - 1))

    loss_bcₚ = (iip ? __Fix3 : Base.Fix2)(loss_bc, cache.p)
    loss_collocationₚ = (iip ? __Fix3 : Base.Fix2)(loss_collocation, cache.p)

    sd_bc = jac_alg.bc_diffmode isa AutoSparse ? SymbolicsSparsityDetection() :
            NoSparsityDetection()
    cache_bc = __sparse_jacobian_cache(
        Val(iip), jac_alg.bc_diffmode, sd_bc, loss_bcₚ, resid_bc, y)

    sd_collocation = if jac_alg.nonbc_diffmode isa AutoSparse
        if L < cache.M
            # For underdetermined problems we use sparse since we don't have banded qr
            colored_matrix = __generate_sparse_jacobian_prototype(
                cache, cache.problem_type, y, y, cache.M, N)
            J_full_band = nothing
            __sparsity_detection_alg(ColoredMatrix(
                sparse(colored_matrix.M), colored_matrix.row_colorvec,
                colored_matrix.col_colorvec))
        else
            J_full_band = BandedMatrix(Ones{eltype(y)}(L + cache.M * (N - 1), cache.M * N),
                (L + 1, cache.M + max(cache.M - L, 0)))
            __sparsity_detection_alg(__generate_sparse_jacobian_prototype(
                cache, cache.problem_type, y, y, cache.M, N))
        end
    else
        J_full_band = nothing
        NoSparsityDetection()
    end
    cache_collocation = __sparse_jacobian_cache(
        Val(iip), jac_alg.nonbc_diffmode, sd_collocation,
        loss_collocationₚ, resid_collocation, y)

    J_bc = zero(init_jacobian(cache_bc))
    J_c = zero(init_jacobian(cache_collocation))
    if J_full_band === nothing
        jac_prototype = vcat(J_bc, J_c)
    else
        jac_prototype = AlmostBandedMatrix{eltype(cache)}(J_full_band, J_bc)
    end

    jac = if iip
        @closure (J, u, p) -> __firk_mpoint_jacobian!(
            J, J_c, u, jac_alg.bc_diffmode, jac_alg.nonbc_diffmode, cache_bc,
            cache_collocation, loss_bcₚ, loss_collocationₚ, resid_bc, resid_collocation, L)
    else
        @closure (u, p) -> __firk_mpoint_jacobian(
            jac_prototype, J_c, u, jac_alg.bc_diffmode, jac_alg.nonbc_diffmode,
            cache_bc, cache_collocation, loss_bcₚ, loss_collocationₚ, L)
    end

    resid_prototype = vcat(resid_bc, resid_collocation)
    nlf = NonlinearFunction{iip}(
        loss; jac = jac, resid_prototype = resid_prototype, jac_prototype = jac_prototype)

    return __internal_nlsolve_problem(cache.prob, resid_prototype, y, nlf, y, cache.p)
end

function __construct_nlproblem(
        cache::FIRKCacheNested{iip}, y, loss_bc::BC, loss_collocation::C,
        loss::LF, ::TwoPointBVProblem) where {iip, BC, C, LF}
    (; jac_alg) = cache.alg
    N = length(cache.mesh)

    lossₚ = iip ? ((du, u) -> loss(du, u, cache.p)) : (u -> loss(u, cache.p))

    resid = vcat(@view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
        __similar(y, cache.M * (N - 1)),
        @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]))
    L = length(cache.bcresid_prototype)

    sd = if jac_alg.diffmode isa AutoSparse
        __sparsity_detection_alg(__generate_sparse_jacobian_prototype(
            cache, cache.problem_type,
            @view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
            @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]),
            cache.M, N))
    else
        NoSparsityDetection()
    end
    diffcache = __sparse_jacobian_cache(Val(iip), jac_alg.diffmode, sd, lossₚ, resid, y)
    jac_prototype = zero(init_jacobian(diffcache))

    jac = if iip
        @closure (J, u, p) -> __firk_2point_jacobian!(
            J, u, jac_alg.diffmode, diffcache, lossₚ, resid)
    else
        @closure (u, p) -> __firk_2point_jacobian(
            u, jac_prototype, jac_alg.diffmode, diffcache, lossₚ)
    end

    resid_prototype = copy(resid)
    nlf = NonlinearFunction{iip}(
        loss; jac = jac, resid_prototype = resid_prototype, jac_prototype = jac_prototype)
    return __internal_nlsolve_problem(cache.prob, resid_prototype, y, nlf, y, cache.p)
end

@views function __firk_loss!(resid, u, p, y, pt::StandardBVProblem, bc!::BC,
        residual, mesh, cache, eval_sol) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    Φ!(resids[2:end], cache, y_, u, p)
    eval_sol.u[1:end] .= y_
    eval_bc_residual!(resids[1], pt, bc!, eval_sol, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __firk_loss!(resid, u, p, y, pt::TwoPointBVProblem, bc!::Tuple{BC1, BC2},
        residual, mesh, cache, _) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    soly_ = VectorOfArray(y_)
    resids = [get_tmp(r, u) for r in residual]
    resida = resids[1][1:prod(cache.resid_size[1])]
    residb = resids[1][(prod(cache.resid_size[1]) + 1):end]
    eval_bc_residual!((resida, residb), pt, bc!, soly_, p, mesh)
    Φ!(resids[2:end], cache, y_, u, p)
    recursive_flatten_twopoint!(resid, resids, cache.resid_size)
    return nothing
end

@views function __firk_loss(
        u, p, y, pt::StandardBVProblem, bc::BC, mesh, cache, eval_sol) where {BC}
    y_ = recursive_unflatten!(y, u)
    eval_sol.u[1:end] .= y_
    resid_bc = eval_bc_residual(pt, bc, eval_sol, p, mesh)
    resid_co = Φ(cache, y_, u, p)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end

@views function __firk_loss(u, p, y, pt::TwoPointBVProblem, bc::Tuple{BC1, BC2},
        mesh, cache, _) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    soly_ = VectorOfArray(y_)
    resid_bca, resid_bcb = eval_bc_residual(pt, bc, soly_, p, mesh)
    resid_co = Φ(cache, y_, u, p)
    return vcat(resid_bca, mapreduce(vec, vcat, resid_co), resid_bcb)
end

@views function __firk_loss_bc!(resid, u, p, pt, bc!::BC, y, mesh,
        cache::Union{FIRKCacheNested, FIRKCacheExpand}) where {BC}
    y_ = recursive_unflatten!(y, u)
    eval_sol = EvalSol(__restructure_sol(y_, cache.in_size), mesh, cache)
    eval_bc_residual!(resid, pt, bc!, eval_sol, p, mesh)
    return nothing
end

@views function __firk_loss_bc(u, p, pt, bc!::BC, y, mesh,
        cache::Union{FIRKCacheNested, FIRKCacheExpand}) where {BC}
    y_ = recursive_unflatten!(y, u)
    eval_sol = EvalSol(__restructure_sol(y_, cache.in_size), mesh, cache)
    return eval_bc_residual(pt, bc!, eval_sol, p, mesh)
end

@views function __firk_loss_collocation!(resid, u, p, y, mesh, residual, cache)
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual[2:end]]
    Φ!(resids, cache, y_, u, p)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __firk_loss_collocation(u, p, y, mesh, residual, cache)
    y_ = recursive_unflatten!(y, u)
    resids = Φ(cache, y_, u, p)
    return mapreduce(vec, vcat, resids)
end

function __firk_mpoint_jacobian!(
        J, _, x, bc_diffmode, nonbc_diffmode, bc_diffcache, nonbc_diffcache, loss_bc::BC,
        loss_collocation::C, resid_bc, resid_collocation, L::Int) where {BC, C}
    sparse_jacobian!(@view(J[1:L, :]), bc_diffmode, bc_diffcache, loss_bc, resid_bc, x)
    sparse_jacobian!(@view(J[(L + 1):end, :]), nonbc_diffmode,
        nonbc_diffcache, loss_collocation, resid_collocation, x)
    return nothing
end

function __firk_mpoint_jacobian!(J::AlmostBandedMatrix, J_c, x, bc_diffmode, nonbc_diffmode,
        bc_diffcache, nonbc_diffcache, loss_bc::BC, loss_collocation::C,
        resid_bc, resid_collocation, L::Int) where {BC, C}
    J_bc = fillpart(J)
    sparse_jacobian!(J_bc, bc_diffmode, bc_diffcache, loss_bc, resid_bc, x)
    sparse_jacobian!(
        J_c, nonbc_diffmode, nonbc_diffcache, loss_collocation, resid_collocation, x)
    exclusive_bandpart(J) .= J_c
    finish_part_setindex!(J)
    return nothing
end

function __firk_mpoint_jacobian(
        J, _, x, bc_diffmode, nonbc_diffmode, bc_diffcache, nonbc_diffcache,
        loss_bc::BC, loss_collocation::C, L::Int) where {BC, C}
    sparse_jacobian!(@view(J[1:L, :]), bc_diffmode, bc_diffcache, loss_bc, x)
    sparse_jacobian!(
        @view(J[(L + 1):end, :]), nonbc_diffmode, nonbc_diffcache, loss_collocation, x)
    return J
end

function __firk_mpoint_jacobian(
        J::AlmostBandedMatrix, J_c, x, bc_diffmode, nonbc_diffmode, bc_diffcache,
        nonbc_diffcache, loss_bc::BC, loss_collocation::C, L::Int) where {BC, C}
    J_bc = fillpart(J)
    sparse_jacobian!(J_bc, bc_diffmode, bc_diffcache, loss_bc, x)
    sparse_jacobian!(J_c, nonbc_diffmode, nonbc_diffcache, loss_collocation, x)
    exclusive_bandpart(J) .= J_c
    finish_part_setindex!(J)
    return J
end

function __firk_2point_jacobian!(J, x, diffmode, diffcache, loss_fn::L, resid) where {L}
    sparse_jacobian!(J, diffmode, diffcache, loss_fn, resid, x)
    return J
end

function __firk_2point_jacobian(x, J, diffmode, diffcache, loss_fn::L) where {L}
    sparse_jacobian!(J, diffmode, diffcache, loss_fn, x)
    return J
end
