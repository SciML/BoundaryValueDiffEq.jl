@concrete struct FIRKCacheNested{iip, T, diffcache, fit_parameters} <:
                 AbstractBoundaryValueDiffEqCache
    order::Int                 # The order of FIRK method
    stage::Int                 # The state of FIRK method
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
    resid_size
    nlsolve_kwargs
    kwargs
end

Base.eltype(::FIRKCacheNested{iip, T}) where {iip, T} = T

@concrete struct FIRKCacheExpand{iip, T, diffcache, fit_parameters} <:
                 AbstractBoundaryValueDiffEqCache
    order::Int                 # The order of FIRK method
    stage::Int                 # The state of FIRK method
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
    nlsolve_kwargs
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

function SciMLBase.__init(
        prob::BVProblem, alg::AbstractFIRK; dt = 0.0, abstol = 1e-6, adaptive = true,
        controller = DefectControl(), nlsolve_kwargs = (; abstol = abstol), kwargs...)
    if alg.nested_nlsolve
        return init_nested(prob, alg; dt = dt, abstol = abstol, adaptive = adaptive,
            controller = controller, nlsolve_kwargs = nlsolve_kwargs, kwargs...)
    else
        return init_expanded(prob, alg; dt = dt, abstol = abstol, adaptive = adaptive,
            controller = controller, nlsolve_kwargs = nlsolve_kwargs, kwargs...)
    end
end

function init_nested(
        prob::BVProblem, alg::AbstractFIRK; dt = 0.0, abstol = 1e-6, adaptive = true,
        controller = DefectControl(), nlsolve_kwargs = (; abstol = abstol), kwargs...)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)

    iip = isinplace(prob)
    if adaptive && isa(alg, FIRKNoAdaptivity)
        error("Algorithm doesn't support adaptivity. Please choose a higher order algorithm.")
    end
    diffcache = __cache_trait(alg.jac_alg)
    fit_parameters = haskey(prob.kwargs, :fit_parameters)

    t₀, t₁ = prob.tspan
    ig, T,
    M,
    Nig,
    X = __extract_problem_details(
        prob; dt, check_positive_dt = true, fit_parameters = fit_parameters)
    mesh = __extract_mesh(prob.u0, t₀, t₁, Nig)
    mesh_dt = diff(mesh)

    chunksize = pickchunksize(M * (Nig - 1))
    __alloc = @closure x -> __maybe_allocate_diffcache(vec(x), chunksize, alg.jac_alg)

    fᵢ_cache = __alloc(zero(X))
    fᵢ₂_cache = vec(zero(X))

    # Don't flatten this here, since we need to expand it later if needed
    y₀ = __initial_guess_on_mesh(X, mesh, prob.p)

    y = __alloc.(copy.(y₀.u))
    TU, ITU = constructRK(alg, T)
    stage = alg_stage(alg)

    k_discrete = [__maybe_allocate_diffcache(
                      safe_similar(X, M, stage), chunksize, alg.jac_alg) for _ in 1:Nig]

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

    defect = VectorOfArray([safe_similar(X, ifelse(adaptive, M, 0)) for _ in 1:Nig])

    # Transform the functions to handle non-vector inputs
    bcresid_prototype = __vec(bcresid_prototype)
    f,
    bc = if X isa AbstractVector
        if fit_parameters == true
            l_parameters = length(prob.p)
            vecf! = function (du, u, p, t)
                prob.f(du, u, @view(u[(end - l_parameters + 1):end]), t)
                du[(end - l_parameters + 1):end] .= 0
            end
            vecbc! = prob.f.bc
            vecf!, vecbc!
        else
            prob.f, prob.f.bc
        end
    elseif iip
        vecf! = @closure (du, u, p, t) -> __vec_f!(du, u, p, t, prob.f, size(X))
        vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
            @closure (r, u, p, t) -> __vec_bc!(r, u, p, t, prob.f.bc, resid₁_size, size(X))
        else
            (
                @closure((r, u,
                    p)->__vec_bc!(r, u, p, first(prob.f.bc), resid₁_size[1], size(X))),
                @closure((
                    r, u, p)->__vec_bc!(r, u, p, last(prob.f.bc), resid₁_size[2], size(X))))
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

    # Somewhat arbitrary initialization of K
    K0 = __K0_on_u0(prob, stage; fit_parameters = fit_parameters)

    nestprob_p = zeros(T, M + 2)

    if iip
        nestprob = NonlinearProblem(
            (res, K, p) -> FIRK_nlsolve!(res, K, p, f, TU, prob.p), K0, nestprob_p)
    else
        nestprob = NonlinearProblem(
            (K, p) -> FIRK_nlsolve(K, p, f, TU, prob.p), K0, nestprob_p)
    end

    return FIRKCacheNested{iip, T, typeof(diffcache), fit_parameters}(
        alg_order(alg), stage, M, size(X), f, bc, prob_, prob.problem_type,
        prob.p, alg, TU, ITU, bcresid_prototype, mesh, mesh_dt, k_discrete,
        y, y₀, residual, fᵢ_cache, fᵢ₂_cache, defect, nestprob, resid₁_size,
        nlsolve_kwargs, (; abstol, dt, adaptive, controller, kwargs...))
end

function init_expanded(
        prob::BVProblem, alg::AbstractFIRK; dt = 0.0, abstol = 1e-6, adaptive = true,
        controller = DefectControl(), nlsolve_kwargs = (; abstol = abstol), kwargs...)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)

    if adaptive && isa(alg, FIRKNoAdaptivity)
        error("Algorithm $(alg) doesn't support adaptivity. Please choose a higher order algorithm.")
    end
    diffcache = __cache_trait(alg.jac_alg)
    fit_parameters = haskey(prob.kwargs, :fit_parameters)

    iip = isinplace(prob)

    t₀, t₁ = prob.tspan
    ig, T,
    M,
    Nig,
    X = __extract_problem_details(
        prob; dt, check_positive_dt = true, fit_parameters = fit_parameters)
    mesh = __extract_mesh(prob.u0, t₀, t₁, Nig)
    mesh_dt = diff(mesh)

    TU, ITU = constructRK(alg, T)
    stage = alg_stage(alg)

    chunksize = pickchunksize(M + M * Nig * (stage + 1))
    __alloc = @closure x -> __maybe_allocate_diffcache(vec(x), chunksize, alg.jac_alg)

    fᵢ_cache = __alloc(zero(X)) # Runtime dispatch
    fᵢ₂_cache = vec(zero(X))

    # Don't flatten this here, since we need to expand it later if needed
    _y₀ = __initial_guess_on_mesh(X, mesh, prob.p)
    y₀ = extend_y(_y₀, Nig + 1, stage)
    y = __alloc.(copy.(y₀.u)) # Runtime dispatch

    k_discrete = [__maybe_allocate_diffcache(
                      safe_similar(X, M, stage), chunksize, alg.jac_alg) for _ in 1:Nig] # Runtime dispatch

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
    f,
    bc = if X isa AbstractVector
        if fit_parameters == true
            l_parameters = length(prob.p)
            vecf! = function (du, u, p, t)
                prob.f(du, u, @view(u[(end - l_parameters + 1):end]), t)
                du[(end - l_parameters + 1):end] .= 0
            end
            vecbc! = prob.f.bc
            vecf!, vecbc!
        else
            prob.f, prob.f.bc
        end
    elseif iip
        vecf! = @closure (du, u, p, t) -> __vec_f!(du, u, p, t, prob.f, size(X))
        vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
            @closure (r, u, p, t) -> __vec_bc!(r, u, p, t, prob.f.bc, resid₁_size, size(X))
        else
            (
                @closure((r, u,
                    p)->__vec_bc!(r, u, p, first(prob.f.bc)[1], resid₁_size[1], size(X))),
                @closure ((r, u,
                    p) -> __vec_bc!(r, u, p, last(prob.f.bc)[2], resid₁_size[2], size(X))))
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

    return FIRKCacheExpand{iip, T, typeof(diffcache), fit_parameters}(
        alg_order(alg), stage, M, size(X), f, bc, prob_, prob.problem_type,
        prob.p, alg, TU, ITU, bcresid_prototype, mesh, mesh_dt, k_discrete,
        y, y₀, residual, fᵢ_cache, fᵢ₂_cache, defect, resid₁_size,
        nlsolve_kwargs, (; abstol, dt, adaptive, controller, kwargs...))
end

"""
    __expand_cache!(cache::FIRKCache)

After redistributing or halving the mesh, this function expands the required vectors to
match the length of the new mesh.
"""
function __expand_cache!(cache::FIRKCacheExpand)
    Nₙ = length(cache.mesh)
    __resize!(cache.k_discrete, Nₙ - 1, cache.M, cache.TU)
    __resize!(cache.y, Nₙ, cache.M, cache.TU)
    __resize!(cache.y₀, Nₙ, cache.M, cache.TU)
    __resize!(cache.residual, Nₙ, cache.M, cache.TU)
    __resize!(cache.defect, Nₙ - 1, cache.M)
    return cache
end

function __expand_cache!(cache::FIRKCacheNested)
    Nₙ = length(cache.mesh)
    __resize!(cache.k_discrete, Nₙ - 1, cache.M)
    __resize!(cache.y, Nₙ, cache.M)
    __resize!(cache.y₀, Nₙ, cache.M)
    __resize!(cache.residual, Nₙ, cache.M)
    __resize!(cache.defect, Nₙ - 1, cache.M)
    return cache
end

function SciMLBase.solve!(cache::FIRKCacheExpand{
        iip, T, diffcache, fit_parameters}) where {iip, T, diffcache, fit_parameters}
    (abstol, adaptive, _), kwargs = __split_kwargs(; cache.kwargs...)
    info::ReturnCode.T = ReturnCode.Success
    prob = cache.prob
    length_u = cache.in_size

    # We do the first iteration outside the loop to preserve type-stability of the
    # `original` field of the solution
    sol_nlprob, info, defect_norm = __perform_firk_iteration(cache, abstol, adaptive)

    if adaptive
        while SciMLBase.successful_retcode(info) && defect_norm > abstol
            sol_nlprob, info,
            defect_norm = __perform_firk_iteration(cache, abstol, adaptive)
        end
    end

    # Parameter estimation, put the estimated parameters to sol.prob.p
    if fit_parameters
        length_u = cache.M - length(prob.p)
        prob = remake(prob; p = first(cache.y₀)[(length_u + 1):end])
        map(x -> resize!(x, length_u), cache.y₀)
        resize!(cache.fᵢ₂_cache, length_u)
    end

    u = shrink_y([reshape(y, length_u) for y in cache.y₀], length(cache.mesh), cache.stage)

    interpolation = __build_interpolation(cache, u)

    odesol = DiffEqBase.build_solution(
        prob, cache.alg, cache.mesh, u; interp = interpolation, retcode = info)
    return __build_solution(prob, odesol, sol_nlprob)
end

function SciMLBase.solve!(cache::FIRKCacheNested{
        iip, T, diffcache, fit_parameters}) where {iip, T, diffcache, fit_parameters}
    (abstol, adaptive, _), kwargs = __split_kwargs(; cache.kwargs...)
    info::ReturnCode.T = ReturnCode.Success
    prob = cache.prob

    # We do the first iteration outside the loop to preserve type-stability of the
    # `original` field of the solution
    sol_nlprob, info, defect_norm = __perform_firk_iteration(cache, abstol, adaptive)

    if adaptive
        while SciMLBase.successful_retcode(info) && defect_norm > abstol
            sol_nlprob, info,
            defect_norm = __perform_firk_iteration(cache, abstol, adaptive)
        end
    end

    # Parameter estimation, put the estimated parameters to sol.prob.p
    if fit_parameters
        length_u = cache.M - length(prob.p)
        prob = remake(prob; p = first(cache.y₀)[(length_u + 1):end])
        map(x -> resize!(x, length_u), cache.y₀)
        resize!(cache.fᵢ₂_cache, length_u)
    end

    u = recursivecopy(cache.y₀)

    interpolation = __build_interpolation(cache, u.u)

    odesol = DiffEqBase.build_solution(
        prob, cache.alg, cache.mesh, u.u; interp = interpolation, retcode = info)
    return __build_solution(prob, odesol, sol_nlprob)
end

function __perform_firk_iteration(
        cache::Union{FIRKCacheExpand, FIRKCacheNested}, abstol, adaptive::Bool)
    nlprob = __construct_nlproblem(cache, vec(cache.y₀), copy(cache.y₀))
    nlsolve_alg = __concrete_nonlinearsolve_algorithm(nlprob, cache.alg.nlsolve)
    sol_nlprob = __solve(nlprob, nlsolve_alg; cache.nlsolve_kwargs..., alias_u0 = true)
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
                __resize!(cache.y₀, length(cache.mesh), cache.M, cache.TU)
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
    (; jac_alg) = cache.alg

    eval_sol = EvalSol(__restructure_sol(y₀.u, cache.in_size), cache.mesh, cache)

    trait = __cache_trait(jac_alg)

    loss_bc = if iip
        @closure (du,
            u,
            p) -> __firk_loss_bc!(du, u, p, pt, cache.bc, cache.y, cache.mesh, cache, trait)
    else
        @closure (
            u, p) -> __firk_loss_bc(u, p, pt, cache.bc, cache.y, cache.mesh, cache, trait)
    end

    loss_collocation = if iip
        @closure (du,
            u,
            p) -> __firk_loss_collocation!(
            du, u, p, cache.y, cache.mesh, cache.residual, cache, trait)
    else
        @closure (u,
            p) -> __firk_loss_collocation(
            u, p, cache.y, cache.mesh, cache.residual, cache, trait)
    end

    loss = if iip
        @closure (du,
            u,
            p) -> __firk_loss!(du, u, p, cache.y, pt, cache.bc, cache.residual,
            cache.mesh, cache, eval_sol, trait)
    else
        @closure (u,
            p) -> __firk_loss(
            u, p, cache.y, pt, cache.bc, cache.mesh, cache, eval_sol, trait)
    end

    return __construct_nlproblem(cache, y, loss_bc, loss_collocation, loss, pt)
end

function __construct_nlproblem(
        cache::FIRKCacheExpand{iip}, y, loss_bc::BC, loss_collocation::C,
        loss::LF, ::StandardBVProblem) where {iip, BC, C, LF}
    (; alg, stage) = cache
    (; jac_alg) = alg
    (; bc_diffmode) = jac_alg
    N = length(cache.mesh)

    resid_bc = cache.bcresid_prototype
    L = length(resid_bc)
    resid_collocation = safe_similar(y, cache.M * (N - 1) * (stage + 1))

    cache_bc = if iip
        DI.prepare_jacobian(loss_bc, resid_bc, bc_diffmode, y, Constant(cache.p))
    else
        DI.prepare_jacobian(loss_bc, bc_diffmode, y, Constant(cache.p))
    end

    nonbc_diffmode = if jac_alg.nonbc_diffmode isa AutoSparse
        if L < cache.M
            # For underdetermined problems we use sparse since we don't have banded qr
            J_full_band = nothing
            sparse_jacobian_prototype = __generate_sparse_jacobian_prototype(
                cache, cache.problem_type, y, y, cache.M, N)
        else
            block_size = cache.M * (stage + 2)
            J_full_band = BandedMatrix(
                Ones{eltype(y)}(L + cache.M * (stage + 1) * (N - 1),
                    cache.M * (stage + 1) * (N - 1) + cache.M),
                (block_size, block_size))
            sparse_jacobian_prototype = __generate_sparse_jacobian_prototype(
                cache, cache.problem_type, y, y, cache.M, N)
        end
        AutoSparse(get_dense_ad(jac_alg.nonbc_diffmode);
            sparsity_detector = ADTypes.KnownJacobianSparsityDetector(sparse_jacobian_prototype),
            coloring_algorithm = __default_coloring_algorithm(jac_alg.nonbc_diffmode))
    else
        J_full_band = nothing
        jac_alg.nonbc_diffmode
    end

    cache_collocation = if iip
        DI.prepare_jacobian(
            loss_collocation, resid_collocation, nonbc_diffmode, y, Constant(cache.p))
    else
        DI.prepare_jacobian(loss_collocation, nonbc_diffmode, y, Constant(cache.p))
    end

    J_bc = if iip
        DI.jacobian(loss_bc, resid_bc, cache_bc, bc_diffmode, y, Constant(cache.p))
    else
        DI.jacobian(loss_bc, cache_bc, bc_diffmode, y, Constant(cache.p))
    end
    J_c = if iip
        DI.jacobian(loss_collocation, resid_collocation, cache_collocation,
            nonbc_diffmode, y, Constant(cache.p))
    else
        DI.jacobian(
            loss_collocation, cache_collocation, nonbc_diffmode, y, Constant(cache.p))
    end

    if J_full_band === nothing
        jac_prototype = vcat(J_bc, J_c)
    else
        jac_prototype = AlmostBandedMatrix{eltype(cache)}(J_full_band, J_bc)
    end

    jac = if iip
        @closure (J,
            u,
            p) -> __firk_mpoint_jacobian!(
            J, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc, cache_collocation,
            loss_bc, loss_collocation, resid_bc, resid_collocation, L, cache.p)
    else
        @closure (u,
            p) -> __firk_mpoint_jacobian(
            jac_prototype, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc,
            cache_collocation, loss_bc, loss_collocation, L, cache.p)
    end

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

    resid_collocation = safe_similar(y, cache.M * (N - 1) * (stage + 1))

    resid = vcat(
        @view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]), resid_collocation,
        @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]))
    L = length(cache.bcresid_prototype)

    diffmode = if jac_alg.diffmode isa AutoSparse
        block_size = cache.M * (stage + 2)
        J_full_band = BandedMatrix(
            Ones{eltype(y)}(L + cache.M * (stage + 1) * (N - 1),
                cache.M * (stage + 1) * (N - 1) + cache.M),
            (block_size, block_size))
        sparse_jacobian_prototype = __generate_sparse_jacobian_prototype(
            cache, cache.problem_type,
            @view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
            @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]),
            cache.M, N)
        AutoSparse(get_dense_ad(jac_alg.diffmode);
            sparsity_detector = ADTypes.KnownJacobianSparsityDetector(sparse_jacobian_prototype),
            coloring_algorithm = __default_coloring_algorithm(jac_alg.diffmode))
    else
        jac_alg.diffmode
    end

    diffcache = if iip
        DI.prepare_jacobian(loss, resid, diffmode, y, Constant(cache.p))
    else
        DI.prepare_jacobian(loss, diffmode, y, Constant(cache.p))
    end

    jac_prototype = if iip
        DI.jacobian(loss, resid, diffcache, diffmode, y, Constant(cache.p))
    else
        DI.jacobian(loss, diffcache, diffmode, y, Constant(cache.p))
    end

    jac = if iip
        @closure (J, u,
            p) -> __firk_2point_jacobian!(J, u, diffmode, diffcache, loss, resid, cache.p)
    else
        @closure (u,
            p) -> __firk_2point_jacobian(
            u, jac_prototype, diffmode, diffcache, loss, cache.p)
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
    (; bc_diffmode) = jac_alg
    N = length(cache.mesh)
    resid_bc = cache.bcresid_prototype
    L = length(resid_bc)
    resid_collocation = safe_similar(y, cache.M * (N - 1))
    cache_bc = if iip
        DI.prepare_jacobian(loss_bc, resid_bc, bc_diffmode, y, Constant(cache.p))
    else
        DI.prepare_jacobian(loss_bc, bc_diffmode, y, Constant(cache.p))
    end

    nonbc_diffmode = if jac_alg.nonbc_diffmode isa AutoSparse
        if L < cache.M
            # For underdetermined problems we use sparse since we don't have banded qr
            J_full_band = nothing
            sparse_jacobian_prototype = __generate_sparse_jacobian_prototype(
                cache, cache.problem_type, y, y, cache.M, N)
        else
            J_full_band = BandedMatrix(Ones{eltype(y)}(L + cache.M * (N - 1), cache.M * N),
                (L + 1, cache.M + max(cache.M - L, 0)))
            sparse_jacobian_prototype = __generate_sparse_jacobian_prototype(
                cache, cache.problem_type, y, y, cache.M, N)
        end
        AutoSparse(get_dense_ad(jac_alg.nonbc_diffmode);
            sparsity_detector = ADTypes.KnownJacobianSparsityDetector(sparse_jacobian_prototype),
            coloring_algorithm = __default_coloring_algorithm(jac_alg.nonbc_diffmode))
    else
        J_full_band = nothing
        jac_alg.nonbc_diffmode
    end

    cache_collocation = if iip
        DI.prepare_jacobian(
            loss_collocation, resid_collocation, nonbc_diffmode, y, Constant(cache.p))
    else
        DI.prepare_jacobian(loss_collocation, nonbc_diffmode, y, Constant(cache.p))
    end

    J_bc = if iip
        DI.jacobian(loss_bc, resid_bc, cache_bc, bc_diffmode, y, Constant(cache.p))
    else
        DI.jacobian(loss_bc, cache_bc, bc_diffmode, y, Constant(cache.p))
    end
    J_c = if iip
        DI.jacobian(loss_collocation, resid_collocation, cache_collocation,
            nonbc_diffmode, y, Constant(cache.p))
    else
        DI.jacobian(
            loss_collocation, cache_collocation, nonbc_diffmode, y, Constant(cache.p))
    end

    if J_full_band === nothing
        jac_prototype = vcat(J_bc, J_c)
    else
        jac_prototype = AlmostBandedMatrix{eltype(cache)}(J_full_band, J_bc)
    end

    jac = if iip
        @closure (J,
            u,
            p) -> __firk_mpoint_jacobian!(
            J, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc, cache_collocation,
            loss_bc, loss_collocation, resid_bc, resid_collocation, L, cache.p)
    else
        @closure (u,
            p) -> __firk_mpoint_jacobian(
            jac_prototype, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc,
            cache_collocation, loss_bc, loss_collocation, L, cache.p)
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

    resid = vcat(@view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
        safe_similar(y, cache.M * (N - 1)),
        @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]))

    diffmode = if jac_alg.diffmode isa AutoSparse
        sparse_jacobian_prototype = __generate_sparse_jacobian_prototype(
            cache, cache.problem_type,
            @view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
            @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]),
            cache.M, N)
        AutoSparse(get_dense_ad(jac_alg.diffmode);
            sparsity_detector = ADTypes.KnownJacobianSparsityDetector(sparse_jacobian_prototype),
            coloring_algorithm = __default_coloring_algorithm(jac_alg.diffmode))
    else
        jac_alg.diffmode
    end

    diffcache = if iip
        DI.prepare_jacobian(loss, resid, diffmode, y, Constant(cache.p))
    else
        DI.prepare_jacobian(loss, diffmode, y, Constant(cache.p))
    end

    jac_prototype = if iip
        DI.jacobian(loss, resid, diffcache, diffmode, y, Constant(cache.p))
    else
        DI.jacobian(loss, diffcache, diffmode, y, Constant(cache.p))
    end

    jac = if iip
        @closure (J, u,
            p) -> __firk_2point_jacobian!(J, u, diffmode, diffcache, loss, resid, cache.p)
    else
        @closure (u,
            p) -> __firk_2point_jacobian(
            u, jac_prototype, diffmode, diffcache, loss, cache.p)
    end

    resid_prototype = copy(resid)
    nlf = NonlinearFunction{iip}(
        loss; jac = jac, resid_prototype = resid_prototype, jac_prototype = jac_prototype)
    return __internal_nlsolve_problem(cache.prob, resid_prototype, y, nlf, y, cache.p)
end

@views function __firk_loss!(resid, u, p, y, pt::StandardBVProblem, bc!::BC, residual,
        mesh, cache, eval_sol, trait::DiffCacheNeeded) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    Φ!(resids[2:end], cache, y_, u, trait)
    eval_sol.u[1:end] .= y_
    eval_bc_residual!(resids[1], pt, bc!, eval_sol, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __firk_loss!(resid, u, p, y, pt::StandardBVProblem, bc!::BC, residual,
        mesh, cache, eval_sol, trait::NoDiffCacheNeeded) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [r for r in residual]
    Φ!(resids[2:end], cache, y_, u, trait)
    eval_sol.u[1:end] .= y_
    eval_bc_residual!(resids[1], pt, bc!, eval_sol, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __firk_loss!(
        resid, u, p, y::AbstractVector, pt::TwoPointBVProblem, bc!::Tuple{BC1, BC2},
        residual, mesh, cache, _, trait::DiffCacheNeeded) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    resida = resids[1][1:prod(cache.resid_size[1])]
    residb = resids[1][(prod(cache.resid_size[1]) + 1):end]
    eval_bc_residual!((resida, residb), pt, bc!, y_, p, mesh)
    Φ!(resids[2:end], cache, y_, u, trait)
    recursive_flatten_twopoint!(resid, resids, cache.resid_size)
    return nothing
end

@views function __firk_loss!(resid, u, p, y, pt::TwoPointBVProblem, bc!::Tuple{BC1, BC2},
        residual, mesh, cache, _, trait::NoDiffCacheNeeded) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    soly_ = VectorOfArray(y_)
    resida = residual[1][1:prod(cache.resid_size[1])]
    residb = residual[1][(prod(cache.resid_size[1]) + 1):end]
    eval_bc_residual!((resida, residb), pt, bc!, soly_, p, mesh)
    Φ!(residual[2:end], cache, y_, u, trait)
    recursive_flatten_twopoint!(resid, residual, cache.resid_size)
    return nothing
end

@views function __firk_loss(
        u, p, y, pt::StandardBVProblem, bc::BC, mesh, cache, eval_sol, trait) where {BC}
    y_ = recursive_unflatten!(y, u)
    eval_sol.u[1:end] .= y_
    resid_bc = eval_bc_residual(pt, bc, eval_sol, p, mesh)
    resid_co = Φ(cache, y_, u, trait)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end

@views function __firk_loss(u, p, y::AbstractVector, pt::TwoPointBVProblem,
        bc::Tuple{BC1, BC2}, mesh, cache, _, trait) where {BC1, BC2}
    y_ = recursive_unflatten!(y, u)
    soly_ = VectorOfArray(y_)
    resid_bca, resid_bcb = eval_bc_residual(pt, bc, y_, p, mesh)
    resid_co = Φ(cache, y_, u, trait)
    return vcat(resid_bca, mapreduce(vec, vcat, resid_co), resid_bcb)
end

@views function __firk_loss_bc!(resid, u, p, pt, bc!::BC, y, mesh,
        cache::Union{FIRKCacheNested, FIRKCacheExpand}, trait) where {BC}
    y_ = recursive_unflatten!(y, u)
    eval_sol = EvalSol(__restructure_sol(y_, cache.in_size), mesh, cache)
    eval_bc_residual!(resid, pt, bc!, eval_sol, p, mesh)
    return nothing
end

@views function __firk_loss_bc(u, p, pt, bc!::BC, y, mesh,
        cache::Union{FIRKCacheNested, FIRKCacheExpand}, trait) where {BC}
    y_ = recursive_unflatten!(y, u)
    eval_sol = EvalSol(__restructure_sol(y_, cache.in_size), mesh, cache)
    return eval_bc_residual(pt, bc!, eval_sol, p, mesh)
end

@views function __firk_loss_collocation!(
        resid, u, p, y, mesh, residual, cache, trait::DiffCacheNeeded)
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual[2:end]]
    Φ!(resids, cache, y_, u, trait)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __firk_loss_collocation!(
        resid, u, p, y, mesh, residual, cache, trait::NoDiffCacheNeeded)
    y_ = recursive_unflatten!(y, u)
    resids = [r for r in residual[2:end]]
    Φ!(resids, cache, y_, u, trait)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __firk_loss_collocation(u, p, y, mesh, residual, cache, trait)
    y_ = recursive_unflatten!(y, u)
    resids = Φ(cache, y_, u, trait)
    return mapreduce(vec, vcat, resids)
end

function __firk_mpoint_jacobian!(
        J, _, x, bc_diffmode, nonbc_diffmode, bc_diffcache, nonbc_diffcache, loss_bc::BC,
        loss_collocation::C, resid_bc, resid_collocation, L::Int, p) where {BC, C}
    DI.jacobian!(
        loss_bc, resid_bc, @view(J[1:L, :]), bc_diffcache, bc_diffmode, x, Constant(p))
    DI.jacobian!(loss_collocation, resid_collocation, @view(J[(L + 1):end, :]),
        nonbc_diffcache, nonbc_diffmode, x, Constant(p))
    return nothing
end

function __firk_mpoint_jacobian!(J::AlmostBandedMatrix, J_c, x, bc_diffmode, nonbc_diffmode,
        bc_diffcache, nonbc_diffcache, loss_bc::BC, loss_collocation::C,
        resid_bc, resid_collocation, L::Int, p) where {BC, C}
    J_bc = fillpart(J)
    DI.jacobian!(loss_bc, resid_bc, J_bc, bc_diffcache, bc_diffmode, x, Constant(p))
    DI.jacobian!(loss_collocation, resid_collocation, J_c,
        nonbc_diffcache, nonbc_diffmode, x, Constant(p))
    exclusive_bandpart(J) .= J_c
    finish_part_setindex!(J)
    return nothing
end

function __firk_mpoint_jacobian(
        J, _, x, bc_diffmode, nonbc_diffmode, bc_diffcache, nonbc_diffcache,
        loss_bc::BC, loss_collocation::C, L::Int, p) where {BC, C}
    DI.jacobian!(loss_bc, @view(J[1:L, :]), bc_diffcache, bc_diffmode, x, Constant(p))
    DI.jacobian!(loss_collocation, @view(J[(L + 1):end, :]),
        nonbc_diffcache, nonbc_diffmode, x, Constant(p))
    return J
end

function __firk_mpoint_jacobian(
        J::AlmostBandedMatrix, J_c, x, bc_diffmode, nonbc_diffmode, bc_diffcache,
        nonbc_diffcache, loss_bc::BC, loss_collocation::C, L::Int, p) where {BC, C}
    J_bc = fillpart(J)
    DI.jacobian!(loss_bc, J_bc, bc_diffcache, bc_diffmode, x, Constant(p))
    DI.jacobian!(loss_collocation, J_c, nonbc_diffcache, nonbc_diffmode, x, Constant(p))
    exclusive_bandpart(J) .= J_c
    finish_part_setindex!(J)
    return J
end

function __firk_2point_jacobian!(J, x, diffmode, diffcache, loss_fn::L, resid, p) where {L}
    DI.jacobian!(loss_fn, resid, J, diffcache, diffmode, x, Constant(p))
    return nothing
end

function __firk_2point_jacobian(x, J, diffmode, diffcache, loss_fn::L, p) where {L}
    DI.jacobian!(loss_fn, J, diffcache, diffmode, x, Constant(p))
    return J
end
