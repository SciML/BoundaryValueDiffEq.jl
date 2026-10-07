@concrete struct AscherCache{iip, T} <: AbstractBoundaryValueDiffEqCache
    prob
    f
    jac
    bc
    bcjac
    k
    original_mesh
    mesh
    mesh_dt
    ncomp
    ny
    p
    alg
    pt
    f_prototype
    bcresid_prototype

    # One scratch bundle per mesh interval, so backend work items do not alias
    collocation_cache

    error

    g
    w
    v
    z
    y
    dmz
    delz
    deldmz
    dmzo
    dmv
    ipvtg
    ipvtw
    TU
    valstr
    nlsolve_kwargs
    optimize_kwargs
    kwargs
    verbose
end

Base.eltype(::AscherCache{iip, T}) where {iip, T} = T

function SciMLBase.__init(
        prob::BVProblem, alg::AbstractAscher; dt = 0.0, controller = GlobalErrorControl(),
        adaptive = true, abstol = 1.0e-4, nlsolve_kwargs = (; abstol),
        optimize_kwargs = (; abstol), verbose = DEFAULT_VERBOSE, kwargs...
    )
    verbose_spec = _process_verbose_param(verbose)
    (; tspan, p) = prob
    _, T, ncy, n, u0 = __extract_problem_details(prob; dt, check_positive_dt = true)
    t₀, t₁ = tspan
    ny = prob.f.mass_matrix isa LinearAlgebra.UniformScaling ? 0 : ncy - rank(prob.f.mass_matrix)
    ncomp = ncy - ny

    k = alg_stage(alg)
    kdy = k * ncy
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)

    # initialize collocation points, constants, mesh
    n = Int(cld(t₁ - t₀, dt))
    mesh = __extract_mesh(prob.u0, t₀, t₁, n)
    mesh_dt = diff(mesh)

    TU = constructAscher(alg, T)

    zval = Vector{T}(undef, ncomp)
    yval = Vector{T}(undef, ny)
    collocation_cache = [__ascher_collocation_scratch(T, ncomp, ny) for _ in 1:n]
    lz = [similar(zval) for _ in 1:(n + 1)]
    fill!.(lz, T(0))
    ly = [similar(yval) for _ in 1:(n + 1)]
    fill!.(ly, T(0))
    if u0 isa AbstractArray{<:Number}
        for zi in lz
            zi .= vec(u0)[1:ncomp]
        end
    end
    dmz = [[zeros(T, ncy) for _ in 1:k] for _ in 1:n]
    dmv = [[zeros(T, ncy) for _ in 1:k] for _ in 1:n]
    delz = [similar(zval) for _ in 1:(n + 1)]
    deldmz = [[zeros(T, ncy) for _ in 1:k] for _ in 1:n]
    dqdmz = [[zeros(T, ncy) for _ in 1:k] for _ in 1:n]
    w = [zeros(kdy, kdy) for _ in 1:n]
    v = [zeros(kdy, ncomp) for _ in 1:n]
    pvtg = zeros(Int, ncomp * (n + 1))
    pvtw = [zeros(Int, kdy) for _ in 1:n]
    valst = [[similar(zval) for _ in 1:4] for _ in 1:(2 * n)]

    err = [similar(zval) for _ in 1:n]

    iip = isinplace(prob)

    f,
        bc = if prob.u0 isa AbstractVector
        prob.f, prob.f.bc
    elseif iip
        vecf! = @closure (du, u, p, t) -> __vec_f!(du, u, p, t, prob.f, size(u0))
        vecbc! = @closure (r, u, p, t) -> __vec_bc!(r, u, p, t, prob.f.bc, ncomp, size(u0))
        vecf!, vecbc!
    else
        vecf = @closure (u, p, t) -> __vec_f(u, p, t, prob.f, size(u0))
        vecbc = @closure (u, p, t) -> __vec_bc(u, p, t, prob.f.bc, size(u0))
        vecf, vecbc
    end

    f_prototype = isnothing(prob.f.f_prototype) ? nothing : __vec(prob.f.f_prototype)
    bcresid_prototype, _ = __get_bcresid_prototype(prob.problem_type, prob, u0)

    if prob.f.jac === nothing
        if iip
            jac = (df, u, p, t) -> begin
                _du = similar(u)
                prob.f(_du, u, p, t)
                _f = @closure (du, u) -> prob.f(du, u, p, t)
                ForwardDiff.jacobian!(df, _f, _du, u)
                return
            end
        else
            jac = (df, u, p, t) -> begin
                _du = prob.f(u, p, t)
                _f = @closure (du, u) -> (du .= prob.f(u, p, t))
                ForwardDiff.jacobian!(df, _f, _du, u)
                return
            end
        end
    else
        jac = prob.f.jac
    end

    if prob.f.bcjac === nothing
        bcjac = prob.problem_type isa StandardBVProblem ? nothing : construct_bc_jac(prob)
    else
        bcjac = prob.f.bcjac
    end

    g = if prob.problem_type isa TwoPointBVProblem
        build_almost_block_diagonals(length(first(bcresid_prototype)), ncomp, mesh, T)
    else
        nothing
    end
    cache = AscherCache{iip, T}(
        prob, f, jac, bc, bcjac, k, copy(mesh), mesh, mesh_dt, ncomp, ny, p,
        alg, prob.problem_type, f_prototype, bcresid_prototype, collocation_cache,
        err, g, w, v, lz, ly, dmz, delz, deldmz, dqdmz, dmv, pvtg, pvtw, TU, valst,
        nlsolve_kwargs, optimize_kwargs, (; abstol, dt, adaptive, controller, kwargs...), verbose_spec
    )
    return cache
end

function SciMLBase.solve!(cache::AscherCache{iip, T}) where {iip, T}
    (abstol, adaptive, _, _), _ = __split_kwargs(; cache.kwargs...)
    info::ReturnCode.T = ReturnCode.Success

    # We do the first iteration outside the loop to preserve type-stability of the
    # `original` field of the solution
    z, y, info, error_norm = __perform_ascher_iteration(cache, abstol, adaptive)

    if adaptive
        while SciMLBase.successful_retcode(info) && norm(error_norm) > abstol
            z, y, info, error_norm = __perform_ascher_iteration(cache, abstol, adaptive)
        end
    end
    u = [vcat(zᵢ, yᵢ) for (zᵢ, yᵢ) in zip(z, y)]

    return SciMLBase.build_solution(
        cache.prob, cache.alg, cache.original_mesh, u;
        interp = __ascher_interpolation(cache), retcode = info
    )
end

function __perform_ascher_iteration(cache::AscherCache{iip, T}, abstol, adaptive::Bool) where {
        iip, T,
    }
    info::ReturnCode.T = ReturnCode.Success
    nlprob = __construct_nlproblem(cache)
    solve_alg = __concrete_solve_algorithm(nlprob, cache.alg.nlsolve, cache.alg.optimize)
    kwargs = __concrete_kwargs(
        cache.alg.nlsolve, cache.alg.optimize, cache.nlsolve_kwargs, cache.optimize_kwargs,
        cache.verbose
    )
    nlsol = __internal_solve(nlprob, solve_alg; kwargs...)
    __ascher_uses_global_system(cache) && __ascher_store_global!(cache, nlsol.u)
    error_norm = 2 * abstol
    info = nlsol.retcode

    z = map(copy, cache.z)
    y = map(copy, cache.y)
    for (i, m) in enumerate(cache.mesh)
        @views approx(cache, m, z[i], y[i])
    end

    # Early terminate if non-adaptive
    (adaptive == false) && return z, y, info, error_norm

    # Preserve only the values needed for adaptive mesh selection.
    dmz = [map(copy, interval) for interval in cache.dmz]
    mesh = copy(cache.mesh)
    mesh_dt = copy(cache.mesh_dt)

    # for error estimation
    # we construct a double mesh and solve the problem on halved mesh again to obtain the error estimation
    # since we got the previous convergence on the initial mesh, we utilize this as the initial guess for our next nonlinear solving
    if info == ReturnCode.Success
        halve_mesh!(cache)
        __expand_cache_for_error!(cache)

        _nlprob = __construct_nlproblem(cache)
        nlsol = solve(_nlprob, solve_alg; kwargs...)

        __ascher_uses_global_system(cache) && __ascher_store_global!(cache, nlsol.u)
        error_norm = error_estimate!(cache)
        if norm(error_norm) > abstol
            mesh_selector!(cache, z, dmz, mesh, mesh_dt, abstol)
            __expand_cache_for_next_iter!(cache)
        end
    else # Something bad happened
        if 2 * (length(cache.mesh) - 1) > cache.alg.max_num_subintervals
            # The solving process failed
            info = ReturnCode.Failure
        else
            # doesn't need to halve the mesh again, just use the expanded cache
            info = ReturnCode.Success # Force a restart, use the expanded cache for the next iteration
            __expand_cache_for_next_iter!(cache)
        end
    end

    return z, y, info, error_norm
end

# expand cache to compute the errors
function __expand_cache_for_error!(cache::AscherCache)
    (; ncomp, ny, mesh) = cache
    Nₙ = length(mesh)
    cache.pt isa TwoPointBVProblem && __append_abd!(cache)
    __append_similar!(cache.z, Nₙ)
    __append_similar!(cache.y, Nₙ)
    __append_similar!(cache.dmz, Nₙ - 1)
    __append_similar!(cache.dmv, Nₙ - 1)
    __append_similar!(cache.delz, Nₙ)
    __append_similar!(cache.deldmz, Nₙ - 1)
    __append_similar!(cache.dmzo, Nₙ - 1)
    __append_similar!(cache.w, Nₙ - 1)
    __append_similar!(cache.v, Nₙ - 1)
    __append_similar!(cache.ipvtg, Nₙ * ncomp)
    __append_similar!(cache.ipvtw, Nₙ - 1)
    __append_similar!(cache.error, Nₙ - 1)
    for _ in 1:((Nₙ - 1) - length(cache.collocation_cache))
        push!(cache.collocation_cache, __ascher_collocation_scratch(eltype(cache), ncomp, ny))
    end
    resize!(cache.collocation_cache, Nₙ - 1)
    return cache
end

# expand the cache to start next iteration
function __expand_cache_for_next_iter!(cache::AscherCache)
    (; mesh) = cache
    Nₙ = length(mesh)
    __expand_cache_for_error!(cache)
    resize!(cache.original_mesh, Nₙ)
    copyto!(cache.original_mesh, mesh)
    __append_similar!(cache.valstr, 2 * Nₙ)
    return cache
end

function __append_similar!(x::AbstractVector{T}, n) where {T}
    N = n - length(x)
    N <= 0 && return resize!(x, n)
    append!(x, [zero(T) for _ in 1:N])
    return x
end

function __append_similar!(x::AbstractVector{<:AbstractArray{T}}, n) where {
        T <:
        AbstractArray,
    }
    N = n - length(x)
    N <= 0 && return resize!(x, n)
    append!(x, [zero.(last(x)) for _ in 1:N])
    return x
end

function __append_similar!(x::AbstractVector{<:AbstractArray{T}}, n) where {T <: Real}
    N = n - length(x)
    N <= 0 && return resize!(x, n)
    append!(x, [zero(last(x)) for _ in 1:N])
    return x
end

function __append_similar(x::AbstractVector{T}, n) where {T}
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [zero(last(x)) for _ in 1:N])
    return deepcopy(x)
end

function __append_similar(x::AbstractVector{<:AbstractArray{T}}, n) where {
        T <:
        AbstractArray,
    }
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [zero.(last(x)) for _ in 1:N])
    return deepcopy(x)
end

function __append_similar(x::AbstractVector{<:AbstractArray{T}}, n) where {T <: Real}
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [zero(last(x)) for _ in 1:N])
    return deepcopy(x)
end

__construct_nlproblem(cache::AscherCache) = __construct_nlproblem(cache, cache.pt)

__ascher_uses_global_system(cache::AscherCache) = cache.pt isa StandardBVProblem || cache.ny > 0

__construct_nlproblem(cache::AscherCache, ::StandardBVProblem) = __construct_ascher_global_nlproblem(cache)

function __construct_ascher_global_nlproblem(cache::AscherCache{iip, T}) where {iip, T}
    x = vcat(reduce(vcat, cache.z), [v for interval in cache.dmz for stage in interval for v in stage])
    nbc = if cache.pt isa TwoPointBVProblem
        sum(length, cache.bcresid_prototype)
    else
        isnothing(cache.prob.f.bcresid_prototype) ? cache.ncomp : length(cache.bcresid_prototype)
    end
    nres = length(x) - cache.ncomp + nbc
    loss! = (res, x, p) -> __ascher_global_loss!(res, x, p, cache)
    loss = isinplace(cache.prob) ? loss! : (x, p) -> (res = similar(x, nres); loss!(res, x, p); res)
    prototype = similar(x, nres)
    jac_prototype, jac! = __ascher_global_jacobian(cache, x, prototype, nbc, loss!)
    jac = isinplace(cache.prob) ? jac! :
        ((x, p) -> (J = copy(jac_prototype); jac!(J, x, p); J))
    cost = isnothing(cache.prob.f.cost) ? ((x, p) -> 0.0) :
        ((x, p) -> cache.prob.f.cost(__ascher_global_solution(cache, x), p))
    return __construct_internal_problem(
        cache.prob, cache.pt, cache.alg, loss, jac, jac_prototype,
        prototype, zeros(eltype(x), nbc), cache.f_prototype, x, cache.p,
        cache.ncomp, length(cache.mesh), cost
    )
end

function __construct_nlproblem(cache::AscherCache{iip, T}, ::TwoPointBVProblem) where {iip, T}
    # DAE constraints must participate in the nonlinear residual, including
    # their stage values. The condensed ODE path updates its cache while
    # evaluating a residual and cannot be differentiated as a DAE system.
    cache.ny > 0 && return __construct_ascher_global_nlproblem(cache)
    (; alg, pt, prob, f_prototype, bcresid_prototype) = cache
    (; jac_alg) = alg
    loss = if iip
        @closure (res, z, p) -> @views Φ!(cache, z, res, pt)
    else
        @closure (z, p) -> @views Φ(cache, z, pt)
    end

    lz = reduce(vcat, cache.z)
    resid_prototype = zero(lz)
    diffmode = if jac_alg.diffmode isa AutoSparse
        #AutoSparse(get_dense_ad(jac_alg.diffmode);
        #    sparsity_detector = __default_sparsity_detector(jac_alg.diffmode),
        #    coloring_algorithm = __default_coloring_algorithm(jac_alg.diffmode))
        # Ascher collocation need more generalized collocation to support AutoSparse
        get_dense_ad(jac_alg.diffmode)
    else
        jac_alg.diffmode
    end

    jac_cache = if iip
        DI.prepare_jacobian(
            loss, resid_prototype, diffmode, lz, Constant(cache.p); strict = Val(false)
        )
    else
        DI.prepare_jacobian(
            loss, diffmode, lz, Constant(cache.p); strict = Val(false)
        )
    end

    jac_prototype = if iip
        DI.jacobian(loss, resid_prototype, jac_cache, diffmode, lz, Constant(cache.p))
    else
        DI.jacobian(loss, jac_cache, diffmode, lz, Constant(cache.p))
    end

    jac = if iip
        @closure (
            J, u,
            p,
        ) -> __ascher_mpoint_jacobian!(J, u, diffmode, jac_cache, loss, lz, cache.p)
    else
        @closure (
            u,
            p,
        ) -> __ascher_mpoint_jacobian(
            jac_prototype, u, diffmode, jac_cache, loss, cache.p
        )
    end

    cost_fun = __build_cost(prob.f.cost, cache, cache.mesh, cache.ncomp + cache.ny)

    return __construct_internal_problem(
        prob, prob.problem_type, alg, loss, jac, jac_prototype,
        resid_prototype, bcresid_prototype, f_prototype, lz,
        cache.p, cache.ncomp, length(cache.mesh), cost_fun
    )
end

function __ascher_mpoint_jacobian!(J, x, diffmode, diffcache, loss, resid, p)
    DI.jacobian!(loss, resid, J, diffcache, diffmode, x, Constant(p))
    return nothing
end
function __ascher_mpoint_jacobian(J, x, diffmode, diffcache, loss, p)
    DI.jacobian!(loss, J, diffcache, diffmode, x, Constant(p))
    return J
end

# rebuild a new g with new mesh
function __append_abd!(cache::AscherCache)
    (; ncomp, mesh, g, bcresid_prototype) = cache
    (; blocks, rows, cols, lasts) = g
    T = eltype(first(blocks))
    n = length(mesh) - 1
    ncol = 2 * ncomp
    resize!(rows, n)
    resize!(cols, n)
    fill!(cols, ncol)
    resize!(lasts, n)
    fill!(lasts, ncomp)
    fill!(rows, ncomp + length(first(bcresid_prototype)))
    lasts[end] = ncol
    rows[end] = ncol
    resize!(blocks, n)
    for i in 1:n
        blocks[i] = zeros(T, rows[i], cols[i])
    end
    return
end

function __ascher_global_solution(cache, x)
    n = length(cache.mesh) - 1
    nz = cache.ncomp * (n + 1)
    z = reshape(view(x, 1:nz), cache.ncomp, n + 1)
    stages = reshape(view(x, (nz + 1):length(x)), cache.ncomp + cache.ny, cache.k, n)
    return AscherBoundarySolution(cache, z, stages)
end

function __ascher_store_global!(cache, x)
    sol = __ascher_global_solution(cache, x)
    for i in eachindex(cache.z)
        @views cache.z[i] .= sol.z[:, i]
    end
    for i in eachindex(cache.dmz), j in 1:cache.k
        @views cache.dmz[i][j] .= sol.stages[:, j, i]
    end
    return nothing
end

function __ascher_interpolation(cache)
    metadata = (;
        mesh = copy(cache.mesh), mesh_dt = copy(cache.mesh_dt),
        ncomp = cache.ncomp, ny = cache.ny, k = cache.k, TU = cache.TU,
        prob = cache.prob,
    )
    z = reduce(hcat, cache.z)
    stages = reshape(
        [v for interval in cache.dmz for stage in interval for v in stage],
        cache.ncomp + cache.ny, cache.k, length(cache.mesh) - 1
    )
    return AscherInterpolation(AscherBoundarySolution(metadata, z, stages))
end
