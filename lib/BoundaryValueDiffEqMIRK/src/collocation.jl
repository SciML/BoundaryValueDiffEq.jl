function Φ!(residual, cache::MIRKCache{iip}, y, u, trait, constraint) where {iip}
    if cache.device_cache !== nothing && isbitstype(eltype(u))
        return __mirk_device_collocation!(
            residual, cache.device_cache, cache, y, u, trait, constraint
        )
    end
    # Sparsity tracers contain host objects. Only this structural analysis runs on CPU;
    # numerical residuals (including ForwardDiff duals) use the selected device.
    platform = cache.device_cache === nothing ? cache.alg.platform : CPU()
    return Φ!(
        residual, cache.collocation_cache, cache.k_discrete, cache.f, cache.TU, y, u,
        cache.p, cache.mesh, cache.mesh_dt, cache.stage, cache.f_prototype,
        cache.singular_term, trait, constraint, Val(iip), platform,
        cache.mass_matrix, cache.algebraic_indices
    )
end

@inline _collocation_tmp(cache, u, ::DiffCacheNeeded) = get_tmp(cache, u)
@inline _collocation_tmp(cache, _, ::NoDiffCacheNeeded) = cache

# Both nested CPU storage and packed device storage use these interval equations.
# RHS dispatch handles state shapes, parameter tuning, and singular terms.
@inline function __mirk_collocation_interval_values!(
        residual, tmp, K, y_left, y_right, p, t_left, h, c, v, x, b,
        f::F, iip, singular_term, metadata, mass_matrix, algebraic_indices
    ) where {F}
    stage = size(K, 2)
    nstates = size(K, 1)
    @inbounds for r in 1:stage
        for j in eachindex(tmp)
            stage_sum = zero(eltype(tmp))
            # Extra components are control variables, with linear interpolation.
            if j <= nstates
                for s in 1:(r - 1)
                    stage_sum += K[j, s] * x[r, s]
                end
            end
            tmp[j] = (1 - v[r]) * y_left[j] + v[r] * y_right[j] + h * stage_sum
        end
        __mirk_rhs!(
            view(K, :, r), f, tmp, p, t_left + c[r] * h, iip, singular_term, metadata
        )
    end

    # Enforce algebraic equations at the right mesh node for index-1 DAEs.
    if algebraic_indices !== nothing
        __mirk_rhs!(residual, f, y_right, p, t_left + h, iip, nothing, metadata)
    end
    @inbounds for j in eachindex(residual)
        if algebraic_indices !== nothing && j in algebraic_indices
            continue
        end
        stage_sum = zero(eltype(residual))
        for r in 1:stage
            stage_sum += K[j, r] * b[r]
        end
        residual[j] = __mass_mesh_entry(mass_matrix, y_right, y_left, j) - h * stage_sum
    end
    return nothing
end

# Evaluation of user RHS and boundary functions, usable inside device kernels.

@inline function __mirk_rhs!(du, f::F, u, p, t, iip::Val{true}, singular_term, ::Nothing) where {F}
    __device_eval!(du, f, (u, p, t), iip)
    __add_singular_term!(du, singular_term, u, t)
    return nothing
end
@inline function __mirk_rhs!(du, f::F, u, p, t, ::Val{false}, singular_term, ::Nothing) where {F}
    # Keep the original CPU broadcast behavior for out-of-place right-hand sides.
    du .= f(u, p, t)
    __add_singular_term!(du, singular_term, u, t)
    return nothing
end

@inline function __mirk_collocation_interval!(
        i, residual, collocation_cache, k_discrete, f!, TU::MIRKTableau, y, u, p,
        mesh, mesh_dt, stage::Int, f_prototype, singular_term, trait, constraint, iip,
        mass_matrix, algebraic_indices
    )
    (; c, v, x, b) = TU
    tmp = _collocation_tmp(collocation_cache[i], u, trait)
    K = _collocation_tmp(k_discrete[i], u, trait)
    y_left = _collocation_tmp(y[i], u, trait)
    y_right = _collocation_tmp(y[i + 1], u, trait)
    return __mirk_collocation_interval_values!(
        residual[i], tmp, K, y_left, y_right, p, mesh[i], mesh_dt[i], c, v, x, b,
        f!, iip, constraint isa Val{true} ? nothing : singular_term, nothing,
        mass_matrix, algebraic_indices
    )
end

@kernel function __mirk_collocation_kernel!(
        residual, collocation_cache, k_discrete, f!, TU, y, u, p, mesh, mesh_dt,
        stage, f_prototype, singular_term, trait, constraint, iip, mass_matrix, algebraic_indices
    )
    i = @index(Global, Linear)
    __mirk_collocation_interval!(
        i, residual, collocation_cache, k_discrete, f!, TU, y, u, p, mesh, mesh_dt,
        stage, f_prototype, singular_term, trait, constraint, iip, mass_matrix, algebraic_indices
    )
end

function Φ!(
        residual, collocation_cache, k_discrete, f!, TU::MIRKTableau, y, u, p,
        mesh, mesh_dt, stage::Int, f_prototype, singular_term, trait, constraint,
        iip, platform::Backend, mass_matrix, algebraic_indices
    )
    kernel! = __mirk_collocation_kernel!(platform)
    kernel!(
        residual, collocation_cache, k_discrete, f!, TU, y, u, p, mesh, mesh_dt,
        stage, f_prototype, singular_term, trait, constraint, iip, mass_matrix, algebraic_indices;
        ndrange = length(k_discrete)
    )
    synchronize(platform)
    return nothing
end

function Φ(cache::MIRKCache, y, u, trait)
    residuals = if cache.device_cache !== nothing && isbitstype(eltype(u))
        [similar(u, cache.M) for _ in eachindex(cache.mesh_dt)]
    else
        [safe_similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    end
    Φ!(residuals, cache, y, u, trait, Val(false))
    return residuals
end

# Host state with reusable packed buffers on the selected device.
__mirk_validate_device_problem(::CPU, prob, alg, u0, tune_parameters) = nothing

function __mirk_validate_device_problem(::Backend, prob, alg, u0, tune_parameters)
    fast_scalar_indexing(u0) || throw(
        ArgumentError(
            "MIRK GPU collocation requires a CPU initial guess; select the GPU with `platform`."
        )
    )
    isbitstype(eltype(u0)) ||
        throw(ArgumentError("MIRK GPU collocation requires an isbits state element type."))
    modes = prob.problem_type isa TwoPointBVProblem ?
        (alg.jac_alg.diffmode,) : (alg.jac_alg.nonbc_diffmode,)
    for mode in modes
        get_dense_ad(mode) isa Union{AutoForwardDiff, AutoFiniteDiff} ||
            throw(
            ArgumentError(
                "MIRK GPU collocation supports AutoForwardDiff or AutoFiniteDiff " *
                    "(optionally wrapped in AutoSparse) for the collocation Jacobian."
            )
        )
    end
    if tune_parameters
        isinplace(prob) && prob.p isa AbstractVector{<:Number} || throw(
            ArgumentError(
                "MIRK GPU parameter tuning requires an in-place RHS and a numeric parameter vector."
            )
        )
    end
    return nothing
end

# Copy array parameters with KernelAbstractions; isbits values need no conversion.
__mirk_device_cache(::CPU, prob, alg, u0, TU, tune_parameters) = nothing
__mirk_device_cache(platform::Backend, prob, alg, u0, TU, tune_parameters) =
    __mirk_device_cache_impl(platform, prob, alg, u0, TU, tune_parameters)

# Separate constructor for CPU-backed kernel tests
function __mirk_device_cache_impl(platform, prob, alg, u0, TU, tune_parameters)
    tableau = map(x -> __device_parameter(platform, x), (TU.c, TU.v, TU.x, TU.b))
    return (;
        platform, tableau, p = __device_parameter(platform, prob.p),
        singular_term = __device_parameter(platform, prob.singular_term),
        mass_matrix = __device_parameter(platform, prob.f.mass_matrix),
        algebraic_indices = __device_parameter(platform, __get_algebraic_indices(prob.f.mass_matrix)),
        buffers = Dict{DataType, Any}(),
    )
end

__mirk_reset_device_cache!(::Nothing) = nothing
__mirk_reset_device_cache!(cache::NamedTuple) = empty!(cache.buffers)

function __mirk_device_buffers(device_cache, cache, ::Type{T}, constraint) where {T}
    N = length(cache.mesh_dt)
    M = cache.M
    L = constraint isa Val{true} ? length(cache.f_prototype) : M
    # Cache primal and ForwardDiff buffers by element type.
    buffers = get(device_cache.buffers, T, nothing)
    if buffers === nothing || size(buffers.y) != (M, N + 1) || size(buffers.k, 1) != L
        platform = device_cache.platform
        buffers = (;
            y = KernelAbstractions.allocate(platform, T, (M, N + 1)),
            tmp = KernelAbstractions.allocate(platform, T, (M, N)),
            k = KernelAbstractions.allocate(platform, T, (L, cache.stage, N)),
            residual = KernelAbstractions.allocate(platform, T, (L, N)),
            mesh = KernelAbstractions.allocate(platform, eltype(cache.mesh), (N + 1,)),
            mesh_dt = KernelAbstractions.allocate(platform, eltype(cache.mesh_dt), (N,)),
            host_y = Matrix{T}(undef, M, N + 1),
            host_k = Array{T}(undef, L, cache.stage, N),
            host_residual = Matrix{T}(undef, L, N),
        )
        device_cache.buffers[T] = buffers
    end
    return buffers
end

function __mirk_device_collocation!(
        residual, device_cache, cache::MIRKCache{iip, T, UB, DC, tune_parameters},
        y, u, trait, constraint
    ) where {iip, T, UB, DC, tune_parameters}
    buffers = __mirk_device_buffers(device_cache, cache, eltype(u), constraint)
    return __mirk_device_collocation!(
        residual, buffers, device_cache, cache, y, u, trait, constraint,
        Val(iip), Val(tune_parameters)
    )
end

function __mirk_device_collocation!(
        residual, buffers, device_cache, cache, y, u, trait, constraint, iip, tune_parameters
    )
    (; platform) = device_cache
    for i in axes(buffers.host_y, 2)
        copyto!(view(buffers.host_y, :, i), _collocation_tmp(y[i], u, trait))
    end
    copyto!(buffers.y, buffers.host_y)
    # Refresh mutable mesh and parameter values before each launch.
    copyto!(buffers.mesh, cache.mesh)
    copyto!(buffers.mesh_dt, cache.mesh_dt)
    __device_copy_parameter!(device_cache.p, cache.p)
    __device_copy_parameter!(device_cache.singular_term, cache.singular_term)
    c, v, x, b = device_cache.tableau
    nparameters = tune_parameters isa Val{true} ? length(cache.p) : 0
    f_size = constraint isa Val{true} && !isnothing(cache.prob.f.f_prototype) ?
        size(cache.prob.f.f_prototype) : cache.in_size
    prod(f_size) == size(buffers.k, 1) || throw(DimensionMismatch("MIRK RHS prototype size mismatch."))
    prod(cache.in_size) == size(buffers.tmp, 1) || throw(DimensionMismatch("MIRK state size mismatch."))
    if constraint isa Val{false} && !isnothing(cache.singular_term)
        size(cache.singular_term) == (size(buffers.k, 1), cache.M) ||
            throw(DimensionMismatch("MIRK singular matrix size must match the state size."))
    end
    kernel! = __mirk_packed_collocation_kernel!(platform)
    kernel!(
        buffers.residual, buffers.tmp, buffers.k, __device_function(cache.prob.f.f), buffers.y,
        device_cache.p, buffers.mesh, buffers.mesh_dt, c, v, x, b,
        device_cache.singular_term, cache.in_size, f_size, nparameters,
        iip, constraint, tune_parameters, device_cache.mass_matrix, device_cache.algebraic_indices; ndrange = length(cache.mesh_dt)
    )
    synchronize(platform)
    copyto!(buffers.host_residual, buffers.residual)
    copyto!(buffers.host_k, buffers.k)
    # Keep host stages in sync for BC interpolation and defect control.
    for i in eachindex(cache.mesh_dt)
        copyto!(residual[i], view(buffers.host_residual, :, i))
        copyto!(_collocation_tmp(cache.k_discrete[i], u, trait), view(buffers.host_k, :, :, i))
    end
    return nothing
end

# Shape metadata selects the packed-array path without a separate RHS type.
@inline function __mirk_rhs!(
        du, f::F, u, p, t, iip, singular_term, metadata::NamedTuple
    ) where {F}
    (; in_size, f_size, nparameters, tune_parameters) = metadata
    parameters = if tune_parameters isa Val{true}
        @inbounds view(u, (length(u) - nparameters + 1):length(u))
    else
        p
    end
    __device_eval!(
        __device_reshape(du, f_size), f,
        (__device_reshape(u, in_size), parameters, t), iip
    )
    @inbounds for j in (length(du) - nparameters + 1):length(du)
        du[j] = zero(eltype(du))
    end
    __device_singular!(du, singular_term, u, t)
    return nothing
end

@kernel function __mirk_packed_collocation_kernel!(
        residual, tmp, K, f, y, p, mesh, mesh_dt, c, v, x, b,
        singular_term, in_size, f_size, nparameters, iip, constraint, tune_parameters,
        mass_matrix, algebraic_indices
    )
    i = @index(Global, Linear)
    @inbounds __mirk_collocation_interval_values!(
        view(residual, :, i), view(tmp, :, i), view(K, :, :, i),
        view(y, :, i), view(y, :, i + 1), p, mesh[i], mesh_dt[i], c, v, x, b,
        f, iip, constraint isa Val{true} ? nothing : singular_term,
        (; in_size, f_size, nparameters, tune_parameters), mass_matrix, algebraic_indices
    )
end
