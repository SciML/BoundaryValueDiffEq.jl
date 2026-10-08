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
    host_mesh
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
    M::Int
    left::Int
    mass
    x
    residual
    jacobian
    work
    new_mesh
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
    initial = BoundaryValueDiffEqCore.__extract_u0(prob.u0, prob.p, first(prob.tspan))
    if alg.device || !(alg.platform isa CPU) || !(__ascher_initial_backend(initial) isa CPU)
        return __init_ascher_device(
            prob, alg; dt, controller, adaptive, abstol, nlsolve_kwargs,
            optimize_kwargs, verbose, kwargs...
        )
    end
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
        prob, f, jac, bc, bcjac, k, copy(mesh), mesh, mesh_dt, nothing, ncomp, ny, p,
        alg, prob.problem_type, f_prototype, bcresid_prototype, collocation_cache,
        err, g, w, v, lz, ly, dmz, delz, deldmz, dqdmz, dmv, pvtg, pvtw, TU, valst,
        ncy, 0, nothing, nothing, nothing, nothing, nothing, nothing,
        nlsolve_kwargs, optimize_kwargs, (; abstol, dt, adaptive, controller, kwargs...), verbose_spec
    )
    return cache
end

function SciMLBase.solve!(cache::AscherCache{iip, T}) where {iip, T}
    cache.host_mesh === nothing || return __solve_ascher_device!(cache)
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
    __ascher_store_global!(cache, nlsol.u)
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

        __ascher_store_global!(cache, nlsol.u)
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

function __construct_nlproblem(cache::AscherCache)
    return cache.host_mesh === nothing ? __construct_nlproblem(cache, cache.pt) :
        __construct_ascher_device_nlproblem(cache)
end

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

# Use the pure global residual for ODEs as well as DAEs. The condensed
# residual mutates its input and cache and cannot support sparse coloring.
__construct_nlproblem(cache::AscherCache, ::TwoPointBVProblem) = __construct_ascher_global_nlproblem(cache)

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

@inline __ascher_jacobian(cache::AscherCache) = cache.jacobian[nothing]

function __init_ascher_device(
        prob, alg; dt, controller = GlobalErrorControl(), adaptive = true,
        abstol = 1.0e-4, nlsolve_kwargs = (; abstol), optimize_kwargs = (; abstol),
        verbose = DEFAULT_VERBOSE, host_mesh = nothing, kwargs...
    )
    controller isa Union{GlobalErrorControl, NoErrorControl} ||
        throw(ArgumentError("Device Ascher supports GlobalErrorControl or NoErrorControl."))
    adaptive && controller isa NoErrorControl && throw(ArgumentError("NoErrorControl requires adaptive = false."))
    alg.optimize === nothing && prob.f.cost === nothing && prob.f.inequality === nothing &&
        prob.f.equality === nothing && prob.lb === nothing && prob.ub === nothing &&
        prob.lcons === nothing && prob.ucons === nothing ||
        throw(ArgumentError("Device Ascher currently supports unconstrained square nonlinear BVPs, not optimization problems."))
    get(prob.kwargs, :tune_parameters, false) && throw(ArgumentError("Device Ascher does not support parameter tuning."))
    initial = BoundaryValueDiffEqCore.__extract_u0(prob.u0, prob.p, first(prob.tspan))
    initial isa AbstractVector{<:Union{Float32, Float64}} ||
        throw(ArgumentError("Device Ascher requires a vector state with Float32 or Float64 elements."))
    T, M = eltype(initial), length(initial)
    platform = alg.platform isa CPU ? __ascher_initial_backend(initial) : alg.platform
    @set! alg.platform = platform
    mode = __ascher_device_mode(alg)
    mass, d = __ascher_device_mass(prob.f.mass_matrix, M, T)
    t0, t1 = prob.tspan
    isfinite(dt) && dt > 0 && isfinite(t0) && isfinite(t1) && t1 > t0 || throw(ArgumentError("Device Ascher requires dt > 0 and an increasing finite tspan."))
    isfinite(abstol) && abstol > 0 || throw(ArgumentError("abstol must be finite and positive."))
    twopoint = prob.problem_type isa TwoPointBVProblem
    left = 0
    if twopoint
        prob.f.bcresid_prototype === nothing && throw(ArgumentError("Device two-point Ascher requires bcresid_prototype = (left, right)."))
        prototype, _ = __get_bcresid_prototype(prob.problem_type, prob, initial)
        left = length(first(prototype))
        left + length(last(prototype)) == d || throw(DimensionMismatch("Ascher requires one boundary condition per differential variable."))
    else
        prob.f.bcresid_prototype === nothing || length(prob.f.bcresid_prototype) == d ||
            throw(DimensionMismatch("Ascher requires one boundary residual per differential variable."))
    end
    if host_mesh === nothing
        n = Int(cld(t1 - t0, dt))
        host_mesh = collect(__extract_mesh(prob.u0, t0, t1, n))
    end
    host_mesh = sort!(unique!(T.(host_mesh)))
    all(isfinite, host_mesh) && all(>(zero(T)), diff(host_mesh)) || throw(ArgumentError("Ascher mesh must be finite and strictly increasing."))
    n = length(host_mesh) - 1
    n <= alg.max_num_subintervals || throw(ArgumentError("Initial mesh exceeds max_num_subintervals."))
    k = alg_stage(alg)
    table = constructAscher(alg, T)
    # acol contains the integrated basis divided by rho, as in CPU vwblok.
    a = T.(table.acol) .* reshape(T.(table.rho), 1, k)
    upload(a) = __ascher_upload(platform, a)
    TU = (; a = upload(a), b = upload(T.(table.b)), rho = upload(T.(table.rho)), coef = upload(T.(table.coef)))
    width = d + M * k
    host_x = zeros(T, n * width + d)
    guess = prob.u0 isa AbstractVector{<:Number} ? Array(prob.u0) : prob.u0
    # Initial values are setup data. Newton states and Jacobians never pass
    # through this host initialization after the device cache has been built.
    function guess_at(t)
        if guess isa AbstractVector{<:Number}
            return guess
        elseif guess isa Function
            return Array(BoundaryValueDiffEqCore.__initial_guess(guess, prob.p, t))
        elseif guess isa SciMLBase.ODESolution
            return Array(guess(t))
        else
            values = hasproperty(guess, :u) ? guess.u : guess
            times = hasproperty(guess, :t) ? guess.t : range(t0, t1; length = length(values))
            length(times) == 1 && return Array(first(values))
            index = clamp(searchsortedlast(times, t), 1, length(times) - 1)
            theta = (t - times[index]) / (times[index + 1] - times[index])
            return (1 - theta) .* Array(values[index]) .+ theta .* Array(values[index + 1])
        end
    end
    for i in 1:(n + 1)
        value = guess_at(host_mesh[i])
        length(value) == M || throw(DimensionMismatch("Initial state dimension changed on the mesh."))
        copyto!(view(host_x, ((i - 1) * width + 1):((i - 1) * width + d)), view(value, 1:d))
        i > n && continue
        for s in 1:k
            stage_value = guess_at(host_mesh[i] + T(table.rho[s]) * (host_mesh[i + 1] - host_mesh[i]))
            offset = (i - 1) * width + d + (s - 1) * M
            copyto!(view(host_x, (offset + d + 1):(offset + M)), view(stage_value, (d + 1):M))
        end
    end
    x = upload(host_x)
    jacobian = __ascher_prepare_device_jacobian(x, platform, mode, d, M, k, n, twopoint ? left : nothing)
    nlkwargs = __concrete_kwargs(alg.nlsolve, nothing, nlsolve_kwargs, optimize_kwargs, _process_verbose_param(verbose))
    return AscherCache{isinplace(prob), T}(
        prob, prob.f.f, nothing, prob.f.bc, nothing, k, copy(host_mesh), upload(host_mesh), nothing,
        host_mesh, d, M - d, upload(prob.p), alg, prob.problem_type, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing, TU, nothing,
        M, left, upload(mass), x, similar(x), Dict(nothing => jacobian),
        Dict{DataType, Any}(),
        (; x = similar(x, 0), mesh = similar(x, 0), coarse = similar(x, 0), fine = similar(x, 0)),
        nlkwargs, optimize_kwargs, (; abstol = T(abstol), dt, adaptive, controller, kwargs...),
        _process_verbose_param(verbose)
    )
end

function __ascher_device_work(cache, ::Type{T}) where {T}
    # Retain owning vectors for each scalar/AD type. Never retain a reshape of a
    # resizable device allocation across refinement.
    buffers = get!(cache.work, T) do
        (;
            x = similar(cache.x, T, 0), r = similar(cache.x, T, 0), minus = similar(cache.x, T, 0),
            stages = similar(cache.x, T, 0), boundary = similar(cache.x, T, cache.ncomp^2),
        )
    end
    for buffer in (buffers.x, buffers.r, buffers.minus)
        resize!(buffer, length(cache.x))
    end
    n = length(cache.host_mesh) - 1
    resize!(buffers.stages, cache.M * cache.k * n)
    return (;
        buffers.x, buffers.r, buffers.minus,
        stages = __reshape_buffer(buffers.stages, cache.M, cache.k, n),
        boundary = __reshape_buffer(buffers.boundary, cache.ncomp, cache.ncomp),
    )
end

function __construct_ascher_device_nlproblem(cache::AscherCache)
    residual! = (r, x, p) -> __ascher_device_residual!(r, x, cache)
    jac! = (J, x, p) -> __ascher_device_jacobian!(J, x, cache)
    # Explicit products also keep user-selected line searches on sparse storage.
    product = copy(__ascher_jacobian(cache).matrix)
    jvp! = (out, v, x, p) -> begin
        jac!(product, x, p)
        LinearAlgebra.mul!(out, product, v)
    end
    vjp! = (out, v, x, p) -> begin
        jac!(product, x, p)
        LinearAlgebra.mul!(out, adjoint(product), v)
    end
    nf = SciMLBase.NonlinearFunction{true}(
        residual!; jac = jac!, jvp = jvp!, vjp = vjp!,
        jac_prototype = __ascher_jacobian(cache).matrix, resid_prototype = cache.residual
    )
    return SciMLBase.NonlinearProblem(nf, copy(cache.x), cache.p)
end

function __ascher_device_solve_once!(cache)
    __ascher_refresh_parameter!(cache.p, cache.prob.p)
    nlprob = __construct_nlproblem(cache)
    algorithm = __concrete_device_solve_algorithm(
        nlprob, cache.alg.nlsolve, cache.alg.optimize;
        linsolve = __default_sparse_linsolve(__ascher_jacobian(cache).matrix)
    )
    result = __internal_solve(nlprob, algorithm; cache.nlsolve_kwargs...)
    copyto!(cache.x, result.u)
    return result
end

function __solve_ascher_device!(cache::AscherCache)
    result = __ascher_device_solve_once!(cache)
    current, result, retcode = __ascher_refine_solution(cache, result)
    values = __ascher_device_sample(current, current.mesh)
    # `values` is owned by this solution. Views avoid one GPU allocation/copy
    # per mesh node, while subsequent solve! calls cannot change this storage.
    u = [view(values, :, i) for i in axes(values, 2)]
    interpolation = AscherDeviceInterpolation(copy(current.x), copy(current.mesh), current.TU.coef, current.ncomp, current.M, current.k, current.alg.platform)
    return SciMLBase.build_solution(
        cache.prob, cache.alg, copy(current.host_mesh), u;
        interp = interpolation, retcode, original = result
    )
end

# A conservative, branch-independent graph follows directly from Ascher's
# stencil. Dense local RHS/BC blocks allow arbitrary state dependencies, while
# global storage and color count stay bounded with increasing mesh length.
function __ascher_device_pattern(T, d, M, k, n, left = nothing)
    rows, cols = Int[], Int[]
    width = d + M * k
    entries = (left === nothing ? d * (n * width + d) : d * width) + n * (d * (k + 2) + k * M * (k * d + M))
    sizehint!(rows, entries)
    sizehint!(cols, entries)
    # General boundary callbacks can couple arbitrary intervals and algebraic
    # stages. Differentiate these rows separately from colored collocation rows.
    if left === nothing
        for c in 1:(n * width + d), j in 1:d
            push!(rows, j)
            push!(cols, c)
        end
    else
        # Endpoint callbacks only inspect the first/last local polynomial,
        # including every algebraic stage. Color these rows with collocation.
        for (interval, bcrows) in ((1, 1:left), (n, (left + 1):d))
            offset = (interval - 1) * width
            for c in (offset + 1):(offset + width), j in bcrows
                push!(rows, j)
                push!(cols, c)
            end
        end
    end
    for i in 1:n
        offset = (i - 1) * width
        for j in 1:d
            row = d + offset + j
            for col in (offset + j, offset + width + j)
                push!(rows, row)
                push!(cols, col)
            end
            for s in 1:k
                push!(rows, row)
                push!(cols, offset + d + (s - 1) * M + j)
            end
        end
        for s in 1:k, j in 1:M
            row = d + offset + d + (s - 1) * M + j
            for c in 1:d
                push!(rows, row)
                push!(cols, offset + c)
            end
            for stage in 1:k, c in 1:d
                push!(rows, row)
                push!(cols, offset + d + (stage - 1) * M + c)
            end
            for c in (d + 1):M
                push!(rows, row)
                push!(cols, offset + d + (s - 1) * M + c)
            end
        end
    end
    size = n * width + d
    return sparse(rows, cols, ones(T, length(rows)), size, size)
end

function __ascher_prepare_device_jacobian(x, platform, mode, d, M, k, n, left = nothing)
    __device_sparse_supported(x) || throw(ArgumentError("No sparse Ascher storage extension is loaded for $(typeof(x)); CUDA is supported by loading CUDA.jl."))
    pattern = __ascher_device_pattern(eltype(x), d, M, k, n, left)
    algorithm = BoundaryValueDiffEqCore.__default_coloring_algorithm(mode)
    algorithm isa ADTypes.NoColoringAlgorithm &&
        (algorithm = BoundaryValueDiffEqCore.__default_coloring_algorithm(nothing))
    separate_boundary = left === nothing
    color_pattern = separate_boundary ? pattern[(d + 1):end, :] : pattern
    colors = collect(Int, ADTypes.column_coloring(color_pattern, algorithm))
    length(colors) == length(x) && all(>(0), colors) ||
        throw(ArgumentError("Ascher requires positive column colors for every unknown."))
    # Validate explicitly supplied colorings. The built-in greedy algorithm
    # already guarantees this property; avoid a second graph traversal and one
    # Set allocation per residual row on large meshes for the default case.
    if mode isa AutoSparse && !(mode.coloring_algorithm isa ADTypes.NoColoringAlgorithm)
        rowcolors = [Set{Int}() for _ in axes(pattern, 1)]
        for col in axes(pattern, 2), index in SparseArrays.nzrange(pattern, col)
            row = SparseArrays.rowvals(pattern)[index]
            separate_boundary && row <= d && continue
            set = rowcolors[row]
            colors[col] in set && throw(ArgumentError("Ascher column coloring has a collision."))
            push!(set, colors[col])
        end
    end
    storage = __device_sparse_matrix(x, pattern)
    upload(a) = __ascher_upload(platform, a)
    return (;
        storage.matrix, rows = upload(storage.rows), cols = upload(storage.cols),
        colors = upload(colors), ncolors = maximum(colors), mode = get_dense_ad(mode), separate_boundary,
    )
end

@kernel function __ascher_seed!(dual, x, colors, firstcolor, ::Val{C}) where {C}
    i = @index(Global, Linear)
    @inbounds dual[i] = eltype(dual)(
        x[i], ForwardDiff.Partials(
            ntuple(j -> (colors === nothing ? i : colors[i]) == firstcolor + j - 1 ? one(eltype(x)) : zero(eltype(x)), Val(C))
        )
    )
end

@kernel function __ascher_extract_partials!(values, r, rows, cols, colors, firstcolor, d, boundary, ::Val{C}) where {C}
    index = @index(Global, Linear)
    @inbounds begin
        isboundary = rows[index] <= d
        color = isboundary ? cols[index] : colors[cols[index]]
        part = color - firstcolor + 1
        if isboundary == boundary && 1 <= part <= C
            values[index] = ForwardDiff.partials(r[rows[index]])[part]
        end
    end
end

@inline __ascher_fd_step(x, relstep, absstep, direction) = max(abs(x) * relstep, absstep) * direction

@kernel function __ascher_perturb!(out, x, colors, color, relstep, absstep, direction)
    i = @index(Global, Linear)
    @inbounds out[i] = x[i] + ((colors === nothing ? i : colors[i]) == color ? __ascher_fd_step(x[i], relstep, absstep, direction) : zero(eltype(x)))
end

@kernel function __ascher_extract_fd!(values, plus, minus, x, rows, cols, colors, color, relstep, absstep, direction, central, d, boundary)
    index = @index(Global, Linear)
    @inbounds begin
        col = cols[index]
        isboundary = rows[index] <= d
        if isboundary == boundary && (isboundary ? col : colors[col]) == color
            h = __ascher_fd_step(x[col], relstep, absstep, direction)
            values[index] = (plus[rows[index]] - minus[rows[index]]) / (central ? 2h : h)
        end
    end
end

function __ascher_device_jacobian!(J, x, cache)
    return __ascher_device_jacobian!(J, x, cache, __ascher_jacobian(cache).mode)
end

function __ascher_device_jacobian!(J, x, cache, mode::AutoForwardDiff{C}) where {C}
    chunk = Val(C === nothing ? min(8, __ascher_jacobian(cache).ncolors) : min(C, __ascher_jacobian(cache).ncolors))
    return __ascher_device_forward_jacobian!(J, x, cache, mode, chunk)
end

function __ascher_device_forward_jacobian!(J, x, cache, mode, chunk::Val{C}) where {C}
    tag = mode.tag === nothing ? typeof(ForwardDiff.Tag(__ascher_device_residual!, eltype(x))) : typeof(mode.tag)
    D = ForwardDiff.Dual{tag, eltype(x), C}
    work = __ascher_device_work(cache, D)
    (; rows, cols, colors, ncolors, separate_boundary) = __ascher_jacobian(cache)
    platform = cache.alg.platform
    values = SparseArrays.nonzeros(J)
    for color in 1:C:ncolors
        __ascher_seed!(platform)(work.x, x, colors, color, chunk; ndrange = length(x))
        __ascher_device_residual!(work.r, work.x, cache)
        __ascher_extract_partials!(platform)(values, work.r, rows, cols, colors, color, separate_boundary ? cache.ncomp : 0, false, chunk; ndrange = length(values))
    end
    for column in 1:C:(separate_boundary ? length(x) : 0)
        __ascher_seed!(platform)(work.x, x, nothing, column, chunk; ndrange = length(x))
        __ascher_device_boundary_residual!(work.r, work.x, cache)
        __ascher_extract_partials!(platform)(values, work.r, rows, cols, colors, column, cache.ncomp, true, chunk; ndrange = length(values))
    end
    synchronize(platform)
    return J
end

function __ascher_device_jacobian!(J, x, cache, mode::AutoFiniteDiff)
    T = eltype(x)
    central = mode.fdjtype isa Val{:central}
    default_step = central ? cbrt(eps(T)) : sqrt(eps(T))
    relstep = hasproperty(mode, :relstep) && mode.relstep !== nothing ? T(mode.relstep) : default_step
    absstep = hasproperty(mode, :absstep) && mode.absstep !== nothing ? T(mode.absstep) : relstep
    direction = hasproperty(mode, :dir) && !mode.dir ? -one(T) : one(T)
    work = __ascher_device_work(cache, T)
    (; rows, cols, colors, ncolors, separate_boundary) = __ascher_jacobian(cache)
    platform = cache.alg.platform
    values = SparseArrays.nonzeros(J)
    central || __ascher_device_residual!(work.minus, x, cache)
    for color in 1:ncolors
        __ascher_perturb!(platform)(work.x, x, colors, color, relstep, absstep, direction; ndrange = length(x))
        __ascher_device_residual!(work.r, work.x, cache)
        if central
            __ascher_perturb!(platform)(work.x, x, colors, color, relstep, absstep, -direction; ndrange = length(x))
            __ascher_device_residual!(work.minus, work.x, cache)
        end
        __ascher_extract_fd!(platform)(values, work.r, work.minus, x, rows, cols, colors, color, relstep, absstep, direction, central, separate_boundary ? cache.ncomp : 0, false; ndrange = length(values))
    end
    central || __ascher_device_boundary_residual!(work.minus, x, cache)
    for column in 1:(separate_boundary ? length(x) : 0)
        __ascher_perturb!(platform)(work.x, x, nothing, column, relstep, absstep, direction; ndrange = length(x))
        __ascher_device_boundary_residual!(work.r, work.x, cache)
        if central
            __ascher_perturb!(platform)(work.x, x, nothing, column, relstep, absstep, -direction; ndrange = length(x))
            __ascher_device_boundary_residual!(work.minus, work.x, cache)
        end
        __ascher_extract_fd!(platform)(values, work.r, work.minus, x, rows, cols, colors, column, relstep, absstep, direction, central, cache.ncomp, true; ndrange = length(values))
    end
    synchronize(platform)
    return J
end
