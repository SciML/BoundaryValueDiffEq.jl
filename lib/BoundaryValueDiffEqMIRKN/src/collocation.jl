# Both vector-of-array and packed storage use the same interval formula.
# Shape metadata dispatches RHS evaluation without duplicating the stages.
@inline function __mirkn_rhs!(out, f::F, du, u, p, t, iip::Val{true}, ::Nothing) where {F}
    return __device_eval!(out, f, (du, u, p, t), iip)
end
@inline function __mirkn_rhs!(out, f::F, du, u, p, t, ::Val{false}, ::Nothing) where {F}
    out .= f(du, u, p, t)
    return nothing
end
@inline function __mirkn_rhs!(out, f::F, du, u, p, t, iip, in_size::Tuple) where {F}
    return __device_eval!(
        __device_reshape(out, in_size), f,
        (__device_reshape(du, in_size), __device_reshape(u, in_size), p, t), iip
    )
end

@inline function __mirkn_collocation_values!(
        residual, derivative_residual, tmp, tmpd, K, y_left, y_right, d_left, d_right,
        f, p, t, h, c, v, w, b, x, vp, bp, xp, iip, in_size
    )
    for r in axes(K, 2)
        for j in eachindex(tmp)
            su = zero(eltype(tmp))
            sd = zero(eltype(tmpd))
            for s in 1:(r - 1)
                su += K[j, s] * x[r, s]
                sd += K[j, s] * xp[r, s]
            end
            tmp[j] = (1 - v[r]) * y_left[j] + v[r] * y_right[j] +
                h * ((c[r] - v[r] - w[r]) * d_left[j] + w[r] * d_right[j]) + h^2 * su
            tmpd[j] = (1 - vp[r]) * d_left[j] + vp[r] * d_right[j] + h * sd
        end
        __mirkn_rhs!(view(K, :, r), f, tmpd, tmp, p, t + c[r] * h, iip, in_size)
    end
    for j in eachindex(residual)
        su = zero(eltype(residual))
        sd = zero(eltype(derivative_residual))
        for r in axes(K, 2)
            su += K[j, r] * b[r]
            sd += K[j, r] * bp[r]
        end
        residual[j] = y_right[j] - y_left[j] - h * d_left[j] - h^2 * su
        derivative_residual[j] = d_right[j] - d_left[j] - h * sd
    end
    return nothing
end

function Φ!(residual, cache::MIRKNCache, y, u, p = cache.p)
    return __mirkn_collocation!(residual, cache, y, u, p, Val(true))
end
function Φ(cache::MIRKNCache, y, u, p = cache.p)
    residuals = [similar(yᵢ) for yᵢ in y[1:(end - 2)]]
    __mirkn_collocation!(residuals, cache, y, u, p, Val(false))
    return residuals
end
function __mirkn_collocation!(residual, cache::MIRKNCache, y, u, p, iip)
    if cache.device_cache !== nothing && isbitstype(eltype(u))
        return __mirkn_device_collocation!(residual, cache, y, u, p)
    end
    platform = cache.device_cache === nothing ? cache.alg.platform : CPU()
    __mirkn_collocation_kernel!(platform)(
        residual, cache.collocation_cache, cache.k_discrete, cache.f, cache.TU,
        y, u, p, cache.mesh, cache.mesh_dt, iip;
        ndrange = length(cache.k_discrete)
    )
    synchronize(platform)
    return nothing
end

@kernel function __mirkn_collocation_kernel!(
        residual, collocation_cache, k_discrete, f, TU, y, u, p, mesh, mesh_dt, iip
    )
    i = @index(Global, Linear)
    L = length(mesh)
    (; c, v, w, b, x, vp, bp, xp) = TU
    __mirkn_collocation_values!(
        residual[i], residual[L + i - 1],
        get_tmp(collocation_cache[i][1], u), get_tmp(collocation_cache[i][2], u),
        get_tmp(k_discrete[i], u), get_tmp(y[i], u), get_tmp(y[i + 1], u),
        get_tmp(y[L + i], u), get_tmp(y[L + i + 1], u),
        f, p, mesh[i], mesh_dt[i], c, v, w, b, x, vp, bp, xp, iip, nothing
    )
end

@kernel function __mirkn_packed_collocation_kernel!(
        residual, tmp, K, f, y, p, mesh, mesh_dt,
        c, v, w, b, x, vp, bp, xp, in_size, iip
    )
    i = @index(Global, Linear)
    M = size(K, 1)
    @inbounds __mirkn_collocation_values!(
        view(residual, 1:M, i), view(residual, (M + 1):2M, i),
        view(tmp, 1:M, i), view(tmp, (M + 1):2M, i), view(K, :, :, i),
        view(y, 1:M, i), view(y, 1:M, i + 1),
        view(y, (M + 1):2M, i), view(y, (M + 1):2M, i + 1),
        f, p, mesh[i], mesh_dt[i], c, v, w, b, x, vp, bp, xp, iip, in_size
    )
end

@kernel function __mirkn_device_bc_kernel!(residual, bc, y, p, mesh, in_size, bc_sizes, iip, twopoint)
    @inbounds begin
        M, nodes = size(y, 1) ÷ 2, size(y, 2)
        left = prod(bc_sizes[1])
        if twopoint isa Val{true}
            right = prod(bc_sizes[2])
            __device_eval!(
                __device_reshape(view(residual, 1:left), bc_sizes[1]), bc[1],
                (
                    __device_reshape(view(y, (M + 1):2M, 1), in_size),
                    __device_reshape(view(y, 1:M, 1), in_size), p,
                ), iip
            )
            __device_eval!(
                __device_reshape(view(residual, (length(residual) - right + 1):length(residual)), bc_sizes[2]), bc[2],
                (
                    __device_reshape(view(y, (M + 1):2M, nodes), in_size),
                    __device_reshape(view(y, 1:M, nodes), in_size), p,
                ), iip
            )
        else
            sol = MIRKNDeviceEvalSol(y, mesh, in_size, 0)
            dsol = MIRKNDeviceEvalSol(y, mesh, in_size, M)
            __device_eval!(__device_reshape(view(residual, 1:left), bc_sizes[1]), bc, (dsol, sol, p, mesh), iip)
        end
    end
end

function BoundaryValueDiffEqCore.__device_residual!(resid, u, cache::MIRKNCache{iip}, boundary = true) where {iip}
    work = __mirkn_device_buffers(cache, eltype(u))
    y = __reshape_buffer(u, size(__mirkn_states(cache)))
    M, nodes = size(y)
    left = prod(cache.resid_size[1])
    collocation = reshape(view(resid, (left + 1):(left + M * (nodes - 1))), M, nodes - 1)
    (; c, v, w, b, x, vp, bp, xp) = cache.TU
    __mirkn_packed_collocation_kernel!(cache.alg.platform)(
        collocation, work.tmp, work.k, cache.f,
        y, cache.p, cache.mesh, cache.mesh_dt, c, v, w, b, x, vp, bp, xp, cache.in_size, Val(iip); ndrange = nodes - 1
    )
    if boundary
        __mirkn_device_bc_kernel!(cache.alg.platform)(
            resid, cache.bc, y, cache.p, cache.mesh, cache.in_size,
            cache.resid_size, Val(iip), Val(cache.problem_type isa TwoPointSecondOrderBVProblem); ndrange = 1
        )
    end
    synchronize(cache.alg.platform)
    return resid
end

# CPU nonlinear solves with explicit device collocation. Sparsity tracers stay on
# the host; primal and ForwardDiff dual values use reusable packed device arrays.
__mirkn_device_cache(::CPU, prob, alg, u0, TU) = nothing
function __mirkn_device_cache(platform::Backend, prob, alg, u0, TU)
    isbitstype(eltype(u0)) || throw(ArgumentError("MIRKN GPU collocation requires isbits state elements."))
    mode = prob.problem_type isa TwoPointSecondOrderBVProblem ? alg.jac_alg.diffmode : alg.jac_alg.nonbc_diffmode
    __device_validate_ad(mode)
    return __mirkn_device_cache_impl(platform, prob, TU)
end
function __mirkn_device_cache_impl(platform, prob, TU)
    tableau = map(
        x -> __device_parameter(platform, x),
        (TU.c, TU.v, TU.w, TU.b, TU.x, TU.vp, TU.bp, TU.xp)
    )
    return (; platform, tableau, p = __device_parameter(platform, prob.p), buffers = Dict{DataType, Any}())
end

function __mirkn_device_collocation!(residual, cache::MIRKNCache{iip}, y, u, p) where {iip}
    device = cache.device_cache
    T, M, N = eltype(u), cache.M, length(cache.mesh_dt)
    buffers = get!(device.buffers, T) do
        allocate(dims) = KernelAbstractions.allocate(device.platform, T, dims)
        (;
            y = allocate((2M, N + 1)), tmp = allocate((2M, N)),
            k = allocate((M, cache.stage, N)), residual = allocate((2M, N)),
            mesh = __device_parameter(device.platform, cache.mesh),
            mesh_dt = __device_parameter(device.platform, cache.mesh_dt),
            host_y = Matrix{T}(undef, 2M, N + 1), host_k = Array{T}(undef, M, cache.stage, N),
            host_residual = Matrix{T}(undef, 2M, N),
        )
    end
    for i in 1:(N + 1)
        copyto!(view(buffers.host_y, 1:M, i), get_tmp(y[i], u))
        copyto!(view(buffers.host_y, (M + 1):2M, i), get_tmp(y[N + 1 + i], u))
    end
    copyto!(buffers.y, buffers.host_y)
    copyto!(buffers.mesh, cache.mesh)
    copyto!(buffers.mesh_dt, cache.mesh_dt)
    __device_copy_parameter!(device.p, p)
    # Scalar parameters may also be replaced between calls.
    parameters = isbits(p) ? p : device.p
    __mirkn_packed_collocation_kernel!(device.platform)(
        buffers.residual, buffers.tmp, buffers.k,
        cache.prob.f.f, buffers.y, parameters, buffers.mesh, buffers.mesh_dt,
        device.tableau..., cache.in_size, Val(iip); ndrange = N
    )
    synchronize(device.platform)
    copyto!(buffers.host_residual, buffers.residual)
    copyto!(buffers.host_k, buffers.k)
    for i in 1:N
        copyto!(residual[i], view(buffers.host_residual, 1:M, i))
        copyto!(residual[N + i], view(buffers.host_residual, (M + 1):2M, i))
        copyto!(get_tmp(cache.k_discrete[i], u), view(buffers.host_k, :, :, i))
    end
    return nothing
end

# Copy array parameters with KernelAbstractions; isbits values need no conversion.
# Evaluation of user RHS and boundary functions, usable inside device kernels.
