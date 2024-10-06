@concrete struct AscherCache{iip, T}
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
    zeta
    fixpnt
    alg

    residual
    zval
    yval
    gval

    error
    error_norm

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
    kwargs
end

Base.eltype(::AscherCache{iip, T}) where {iip, T} = T

function get_fixed_points(prob::BVProblem, alg::AbstractAscher)
    t₀ = prob.tspan[1]
    t₁ = prob.tspan[2]
    fixpnt = sort(alg.zeta)
    zeta = copy(alg.zeta)

    if prob.problem_type isa TwoPointBVProblem
        return zeta, Vector{eltype{zeta}}(undef, 0)
    else
        filter!(x -> (x ≉ t₀) && (x ≉ t₁), fixpnt)
        return zeta, fixpnt
    end
end

function SciMLBase.__init(prob::BVProblem, alg::AbstractAscher; dt = 0.0,
        adaptive = true, abstol = 1e-4, kwargs...)
    (; tspan, p) = prob
    _, T, ncy, n, u0 = __extract_problem_details(prob; dt, check_positive_dt = true)
    t₀, t₁ = tspan
    ny = ncy - rank(prob.f.mass_matrix)
    ncomp = ncy - ny

    k = alg_stage(alg)
    zeta::Vector, fixpnt = get_fixed_points(prob, alg)
    kdy = k * ncy
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)

    # initialize collocation points, constants, mesh
    n = Int(cld(t₁ - t₀, dt))
    mesh = collect(t₀:dt:t₁)
    mesh_dt = diff(mesh)

    TU = constructAscher(alg, T)

    residual = Vector{T}(undef, ncy)
    zval = Vector{T}(undef, ncomp)
    yval = Vector{T}(undef, ny)
    gval = Vector{T}(undef, ncomp)
    lz = [similar(zval) for _ in 1:(n + 1)]
    ly = [similar(yval) for _ in 1:(n + 1)]
    dmz = [[zeros(ncy) for _ in 1:k] for _ in 1:n]
    dmv = [[zeros(ncy) for _ in 1:k] for _ in 1:n]
    delz = [similar(zval) for _ in 1:(n + 1)]
    deldmz = [[zeros(ncy) for _ in 1:k] for _ in 1:n]
    dqdmz = [[zeros(ncy) for _ in 1:k] for _ in 1:n]
    w = [zeros(kdy, kdy) for _ in 1:n]
    v = [zeros(kdy, ncomp) for _ in 1:n]
    pvtg = zeros(Int, ncomp * (n + 1))
    pvtw = [zeros(Int, kdy) for _ in 1:n]
    valst = [[similar(zval) for _ in 1:4] for _ in 1:(2 * n)]

    err = [similar(zval) for _ in 1:n]
    errest = Vector{T}(undef, ncomp)

    iip = isinplace(prob)

    f, bc = if prob.u0 isa AbstractVector
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
        if iip
            bcjac = (df, u, p, t) -> begin
                _du = similar(u)
                prob.f.bc(_du, u, p, t)
                _f = @closure (du, u) -> prob.f.bc(du, u, p, t)
                ForwardDiff.jacobian!(df, _f, _du, u)
                return
            end
        else
            bcjac = (df, u, p, t) -> begin
                _du = prob.f.bc(u, p, t)
                _f = @closure (du, u) -> (du .= prob.f.bc(u, p, t))
                ForwardDiff.jacobian!(df, _f, _du, u)
                return
            end
        end
    else
        bcjac = prob.f.bcjac
    end

    g = build_almost_block_diagonals(zeta, ncomp, mesh, T)
    cache = AscherCache{iip, T}(
        prob, f, jac, bc, bcjac, k, copy(mesh), mesh, mesh_dt, ncomp, ny, p, zeta,
        fixpnt, alg, residual, zval, yval, gval, err, errest, g, w, v, lz, ly, dmz, delz,
        deldmz, dqdmz, dmv, pvtg, pvtw, TU, valst, (; abstol, dt, adaptive, kwargs...))
    return cache
end

function __split_ascher_kwargs(; abstol, dt, adaptive = true, kwargs...)
    return ((abstol, adaptive, dt), (; abstol, adaptive, kwargs...))
end

function SciMLBase.solve!(cache::AscherCache{iip, T}) where {iip, T}
    (abstol, adaptive, _), kwargs = __split_ascher_kwargs(; cache.kwargs...)
    info::ReturnCode.T = ReturnCode.Success

    # We do the first iteration outside the loop to preserve type-stability of the
    # `original` field of the solution
    z, y, info, error_norm = __perform_ascher_iteration(cache, abstol, adaptive; kwargs...)

    if adaptive
        while SciMLBase.successful_retcode(info) && norm(error_norm) > abstol
            z, y, info, error_norm = __perform_ascher_iteration(
                cache, abstol, adaptive; kwargs...)
        end
    end
    u = [vcat(zᵢ, yᵢ) for (zᵢ, yᵢ) in zip(z, y)]

    return SciMLBase.build_solution(
        cache.prob, cache.alg, cache.original_mesh, u; retcode = info)
end

function __perform_ascher_iteration(cache::AscherCache{iip, T}, abstol, adaptive::Bool;
        nlsolve_kwargs = (;), kwargs...) where {iip, T}
    info::ReturnCode.T = ReturnCode.Success
    nlprob::NonlinearProblem = __construct_nlproblem(cache)
    nlsolve_alg = __concrete_nonlinearsolve_algorithm(nlprob, cache.alg.nlsolve)
    nlsol = __solve(
        nlprob, nlsolve_alg; abstol, verbose = true, kwargs..., nlsolve_kwargs...)
    error_norm = 2 * abstol
    info = nlsol.retcode

    N = length(cache.mesh)

    z = copy(cache.z)
    y = copy(cache.y)
    for i in 1:N
        @views approx(cache, cache.mesh[i], z[i], y[i])
    end

    dmz = copy(cache.dmz)
    mesh = copy(cache.mesh)
    mesh_dt = copy(cache.mesh_dt)

    # Early terminate if non-adaptive
    (adaptive == false) && return z, y, info, error_norm

    # for error estimation
    # we construct a double mesh and solve the problem on halved mesh again to obtain the error estimation
    # since we got the previous convergence on the initial mesh, we utilize this as the initial guess for our next nonlinear solving
    if info == ReturnCode.Success
        halve_mesh!(cache)
        __expand_cache_for_error!(cache)

        _nlprob = __construct_nlproblem(cache)
        nlsol = __solve(
            _nlprob, nlsolve_alg; abstol, verbose = true, kwargs..., nlsolve_kwargs...)

        error_norm = error_estimate!(cache)
        if norm(error_norm) > abstol
            mesh_selector!(cache, z, dmz, mesh, mesh_dt)
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
    __append_abd!(cache)
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
    return cache
end

# expand the cache to start next iteration
function __expand_cache_for_next_iter!(cache::AscherCache)
    (; mesh) = cache
    Nₙ = length(mesh)
    resize!(cache.original_mesh, Nₙ)
    copyto!(cache.original_mesh, mesh)
    return cache
end

function __append_similar!(x::AbstractVector{T}, n) where {T}
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [zero(T) for _ in 1:N])
    return x
end

function __append_similar!(
        x::AbstractVector{<:AbstractArray{T}}, n) where {T <: AbstractArray}
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [zero.(last(x)) for _ in 1:N])
    return x
end

function __append_similar!(x::AbstractVector{<:AbstractArray{T}}, n) where {T <: Real}
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
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

function __append_similar(
        x::AbstractVector{<:AbstractArray{T}}, n) where {T <: AbstractArray}
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

function __construct_nlproblem(cache::AscherCache{iip, T}) where {iip, T}
    (; alg) = cache
    loss = if iip
        @closure (rhs, z, p) -> @views Φ!(cache, z, rhs)
    else
        @closure (z, p) -> @views Φ(cache, z)
    end
    lz = reduce(vcat, cache.z)
    sd = alg.jac_alg.diffmode isa AutoSparse ? SymbolicsSparsityDetection() :
         NoSparsityDetection()
    ad = alg.jac_alg.diffmode
    lossₚ = (iip ? __Fix3 : Base.Fix2)(loss, cache.p)
    jac_cache = __sparse_jacobian_cache(Val(iip), ad, sd, lossₚ, lz, lz)
    jac_prototype = init_jacobian(jac_cache)
    jac = if iip
        @closure (J, u, p) -> __ascher_mpoint_jacobian!(J, u, ad, jac_cache, lossₚ, lz)
    else
        @closure (u, p) -> __ascher_mpoint_jacobian(
            jac_prototype, u, ad, jac_cache, lossₚ, lz)
    end
    resid_prototype = zero(lz)
    _nlf = __unsafe_nonlinearfunction{iip}(loss; resid_prototype, jac, jac_prototype)
    nlprob::NonlinearProblem = NonlinearProblem(_nlf, lz, cache.p)
    return nlprob
end

function __ascher_mpoint_jacobian!(J, x, diffmode, diffcache, loss, resid)
    sparse_jacobian!(J, diffmode, diffcache, loss, resid, x)
    return nothing
end
function __ascher_mpoint_jacobian(J, x, diffmode, diffcache, loss, resid)
    sparse_jacobian!(J, diffmode, diffcache, loss, resid, x)
    return J
end

# rebuild a new g with new mesh
function __append_abd!(cache::AscherCache)
    (; zeta, ncomp, mesh, g) = cache
    (; blocks, rows, cols, lasts) = g
    T = eltype(first(blocks))
    n = length(mesh) - 1
    ncol = 2 * ncomp
    resize!(rows, n)
    resize!(cols, n)
    fill!(cols, ncol)
    resize!(lasts, n)
    fill!(lasts, ncomp)
    # build integs (describing block structure of matrix)
    let lside = 0
        for i in 1:(n - 1)
            lside = first(findall(x::Float64 -> x > mesh[i], zeta)) - 1
            (lside == ncomp) && break
            rows[i] = ncomp + lside
        end
    end
    lasts[end] = ncol
    rows[end] = ncol
    resize!(blocks, n)
    for i in 1:n
        blocks[i] = zeros(T, rows[i], cols[i])
    end
end
