function Φ!(residual, cache::FIRKCacheExpand, y, u, trait, constraint)
    return __firk_collocation!(residual, cache, y, u, trait, Val(true), constraint)
end

function Φ!(residual, cache::FIRKCacheNested, y, u, trait, constraint)
    return __firk_collocation!(residual, cache, y, u, trait, Val(true))
end

@inline _collocation_tmp(cache, u, ::DiffCacheNeeded) = get_tmp(cache, u)
@inline _collocation_tmp(cache, _, ::NoDiffCacheNeeded) = cache

@inline __firk_stage_residual(residual::AbstractVector, r) = residual[r]
@inline __firk_stage_residual(residual::AbstractMatrix, r) = view(residual, :, r)

@inline function __firk_rhs!(out, f::F, u, p, t, iip::Val{true}, singular, ::Nothing) where {F}
    __device_eval!(out, f, (u, p, t), iip)
    __add_singular_term!(out, singular, u, t)
    return nothing
end
@inline function __firk_rhs!(out, f::F, u, p, t, ::Val{false}, singular, ::Nothing) where {F}
    out .= f(u, p, t)
    __add_singular_term!(out, singular, u, t)
    return nothing
end
@inline function __firk_rhs!(out, f::F, u, p, t, iip, singular, metadata::NamedTuple) where {F}
    (; in_size, f_size, nparameters) = metadata
    parameters = nparameters == 0 ? p : view(u, (length(u) - nparameters + 1):length(u))
    __device_eval!(
        __device_reshape(out, f_size), f,
        (__device_reshape(u, in_size), parameters, t), iip
    )
    for j in (length(out) - nparameters + 1):length(out)
        out[j] = zero(eltype(out))
    end
    __device_singular!(out, singular, u, t)
    return nothing
end

@inline __firk_stage_mass(mass, K, j, r, ::Nothing) = __mass_stage_entry(mass, K, j, r)
@inline __firk_stage_mass(mass, K, j, r, ::NamedTuple) = __firk_mass_stage(mass, K, j, r)

# Shared by expanded CPU and packed device intervals; storage dispatch only
# selects stage residual views and the shape of the user's RHS arguments.
@inline function __firk_collocation_values!(
        residual, stage_residuals, tmp, K, y_left, y_right, f, p, t, t_right, h,
        a, b, c, mass, algebraic_indices, iip, singular, metadata
    )
    for r in eachindex(b)
        for j in eachindex(residual)
            stage_sum = zero(eltype(tmp))
            for s in eachindex(b)
                stage_sum += K[j, s] * a[s, r]
            end
            tmp[j] = y_left[j] + h * stage_sum
        end
        for j in (length(residual) + 1):length(tmp)
            tmp[j] = y_left[j]
        end
        out = __firk_stage_residual(stage_residuals, r)
        __firk_rhs!(out, f, tmp, p, t + c[r] * h, iip, singular, metadata)
        for j in eachindex(out)
            out[j] -= __firk_stage_mass(mass, K, j, r, metadata)
        end
    end
    return __firk_continuity!(
        residual, K, y_left, y_right, b, h, f, p, t_right, iip, algebraic_indices, metadata
    )
end

@inline function __firk_continuity!(
        residual, K, y_left, y_right, b, h, f, p, t, iip, algebraic_indices, metadata
    )
    if algebraic_indices !== nothing
        __firk_rhs!(residual, f, y_right, p, t, iip, nothing, metadata)
    end
    for j in eachindex(residual)
        algebraic_indices !== nothing && j in algebraic_indices && continue
        stage_sum = zero(eltype(residual))
        for r in eachindex(b)
            stage_sum += K[j, r] * b[r]
        end
        residual[j] = y_right[j] - y_left[j] - h * stage_sum
    end
    return nothing
end

@views function __firk_collocation_interval!(
        i, residual, collocation_cache, k_discrete, f, TU::FIRKTableau{false}, y, u,
        p, mass_matrix, algebraic_indices, mesh, mesh_dt, stage::Int, f_prototype,
        singular_term, trait, iip, ::Val{constraint}
    ) where {constraint}
    (; c, a, b) = TU
    tmp = _collocation_tmp(collocation_cache[i], u, trait)
    K = _collocation_tmp(k_discrete[i], u, trait)
    ctr = (i - 1) * (stage + 1) + 1
    y_left = _collocation_tmp(y[ctr], u, trait)
    y_right = _collocation_tmp(y[ctr + stage + 1], u, trait)
    nstate = constraint ? length(f_prototype) : length(y_left)
    for r in 1:stage
        stage_y = _collocation_tmp(y[ctr + r], u, trait)
        for j in 1:nstate
            K[j, r] = stage_y[j]
        end
    end
    return __firk_collocation_values!(
        residual[ctr], residual[(ctr + 1):(ctr + stage)], tmp, K, y_left, y_right,
        f, p, mesh[i], mesh[i + 1], mesh_dt[i], a, b, c, mass_matrix, algebraic_indices,
        iip, constraint ? nothing : singular_term, nothing
    )
end

@kernel function __firk_collocation_kernel!(
        residual, collocation_cache, k_discrete, f, TU, y, u, p, mass_matrix,
        algebraic_indices, mesh, mesh_dt, stage, f_prototype, singular_term, trait,
        iip, constraint
    )
    i = @index(Global, Linear)
    __firk_collocation_interval!(
        i, residual, collocation_cache, k_discrete, f, TU, y, u, p, mass_matrix,
        algebraic_indices, mesh, mesh_dt, stage, f_prototype, singular_term, trait,
        iip, constraint
    )
end

function __firk_collocation!(residual, cache::FIRKCacheExpand, y, u, trait, iip, constraint)
    if cache.device_cache !== nothing && isbitstype(eltype(u))
        return __firk_offload_collocation!(residual, cache, y, u, trait, iip, constraint)
    end
    platform = cache.device_cache === nothing ? cache.alg.platform : CPU()
    kernel! = __firk_collocation_kernel!(platform)
    kernel!(
        residual, cache.collocation_cache, cache.k_discrete, cache.f, cache.TU, y, u,
        cache.p, cache.mass_matrix, cache.algebraic_indices, cache.mesh, cache.mesh_dt,
        cache.stage, cache.f_prototype, cache.singular_term, trait, iip, constraint;
        ndrange = length(cache.mesh_dt)
    )
    synchronize(platform)
    return nothing
end

@views function __firk_collocation_nested_interval!(
        i, residual, collocation_cache, k_discrete, f, TU::FIRKTableau{true}, y, u,
        p, algebraic_indices, mesh, mesh_dt, nest_prob, nest_nlsolve_alg,
        nested_nlsolve_kwargs, trait, ::Val{iip}
    ) where {iip}
    (; b) = TU
    nestprob_p = _collocation_tmp(collocation_cache[i], u, trait)
    yᵢ = _collocation_tmp(y[i], u, trait)
    yᵢ₊₁ = _collocation_tmp(y[i + 1], u, trait)
    h = mesh_dt[i]
    nestprob_p[1] = mesh[i]
    nestprob_p[2] = h
    nestprob_p[3:end] .= yᵢ

    # Keep the initial guess private even when the nested solver aliases u0.
    nestprob = remake(nest_prob; u0 = copy(nest_prob.u0), p = nestprob_p)
    nestsol = if trait isa DiffCacheNeeded
        __solve(nestprob, nest_nlsolve_alg; nested_nlsolve_kwargs...)
    else
        solve(nestprob, nest_nlsolve_alg; nested_nlsolve_kwargs...)
    end
    K = _collocation_tmp(k_discrete[i], u, trait)
    K .= nestsol.u
    return __firk_continuity!(
        residual[i], K, yᵢ, yᵢ₊₁, b, h, f, p, mesh[i + 1],
        Val(iip), algebraic_indices, nothing
    )
end

@kernel function __firk_collocation_nested_kernel!(
        residual, collocation_cache, k_discrete, f, TU, y, u, p, algebraic_indices,
        mesh, mesh_dt, nest_prob, nest_nlsolve_alg, nested_nlsolve_kwargs, trait, iip
    )
    i = @index(Global, Linear)
    __firk_collocation_nested_interval!(
        i, residual, collocation_cache, k_discrete, f, TU, y, u, p, algebraic_indices,
        mesh, mesh_dt, nest_prob, nest_nlsolve_alg, nested_nlsolve_kwargs, trait, iip
    )
end

function __firk_collocation!(residual, cache::FIRKCacheNested, y, u, trait, iip)
    (; alg, nest_prob) = cache
    nest_nlsolve_alg = __concrete_solve_algorithm(nest_prob, alg.nlsolve)
    if cache.device_cache !== nothing
        # Nested nonlinear solvers are host orchestration; their numerical stage
        # residuals use packed kernels. The scratch buffers must not be shared
        # by concurrent host interval solves.
        for i in eachindex(cache.mesh_dt)
            __firk_collocation_nested_interval!(
                i, residual, cache.collocation_cache, cache.k_discrete, cache.f,
                cache.TU, y, u, cache.p, cache.algebraic_indices, cache.mesh,
                cache.mesh_dt, nest_prob, nest_nlsolve_alg,
                alg.nested_nlsolve_kwargs, trait, iip
            )
        end
        return nothing
    end
    kernel! = __firk_collocation_nested_kernel!(alg.platform)
    kernel!(
        residual, cache.collocation_cache, cache.k_discrete, cache.f, cache.TU, y, u,
        cache.p, cache.algebraic_indices, cache.mesh, cache.mesh_dt, nest_prob,
        nest_nlsolve_alg, alg.nested_nlsolve_kwargs, trait, iip;
        ndrange = length(cache.mesh_dt)
    )
    synchronize(alg.platform)
    return nothing
end

function Φ(cache::FIRKCacheExpand, y, u, trait)
    residuals = [safe_similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    __firk_collocation!(residuals, cache, y, u, trait, Val(false), Val(false))
    return residuals
end

function Φ(cache::FIRKCacheNested, y, u, trait)
    residuals = [safe_similar(yᵢ) for yᵢ in y[1:(end - 1)]]
    __firk_collocation!(residuals, cache, y, u, trait, Val(false))
    return residuals
end

function FIRK_nlsolve!(res, K, p_nlsolve, f, TU::FIRKTableau{true}, p, mass_matrix)
    return FIRK_nlsolve!(res, K, p_nlsolve, f, TU, p, mass_matrix, Val(true))
end
function FIRK_nlsolve(K, p_nlsolve, f, TU::FIRKTableau{true}, p, mass_matrix)
    res = similar(K, promote_type(eltype(K), eltype(p_nlsolve)), size(K))
    FIRK_nlsolve!(res, K, p_nlsolve, f, TU, p, mass_matrix, Val(false))
    return res
end
function FIRK_nlsolve!(res, K, p_nlsolve, f, TU::FIRKTableau{true}, p, mass_matrix, iip)
    (; a, c, s) = TU
    mesh_i, h = p_nlsolve[1], p_nlsolve[2]
    yᵢ = @view p_nlsolve[3:end]
    T = promote_type(eltype(K), eltype(yᵢ))
    tmp1 = similar(K, T, size(K, 1))
    tmp2 = similar(tmp1)
    for r in 1:s
        tmp1 .= yᵢ
        @views __maybe_matmul!(tmp1, K, a[:, r], h, T(1))
        __firk_rhs!(view(res, :, r), f, tmp1, p, mesh_i + c[r] * h, iip, nothing, nothing)
        @views __subtract_mass_stage!(res[:, r], mass_matrix, K[:, r], tmp2)
    end
    return nothing
end

@inline function __firk_device_rhs!(out, f, u, p, t, in_size, iip, singular_term)
    __device_eval!(
        __device_reshape(out, in_size), f,
        (__device_reshape(u, in_size), p, t), iip
    )
    __device_singular!(out, singular_term, u, t)
    return nothing
end

@kernel function __firk_packed_collocation_kernel!(
        residual, tmp, f, y, p, mesh, mesh_dt, a, b, c, in_size, iip,
        singular_term, f_size, nparameters, mass_matrix, algebraic_indices
    )
    i = @index(Global, Linear)
    stage = length(b)
    ctr = (i - 1) * (stage + 1) + 1
    @inbounds __firk_collocation_values!(
        view(residual, :, ctr), view(residual, :, (ctr + 1):(ctr + stage)),
        view(tmp, :, i), view(y, :, (ctr + 1):(ctr + stage)),
        view(y, :, ctr), view(y, :, ctr + stage + 1),
        f, p, mesh[i], mesh[i + 1], mesh_dt[i], a, b, c, mass_matrix, algebraic_indices,
        iip, size(residual, 1) == size(y, 1) ? singular_term : nothing,
        (; in_size, f_size, nparameters)
    )
end

@kernel function __firk_device_bc_kernel!(
        resid, bc, y, coefficients, p, mesh, mesh_dt, in_size, bc_sizes, stage, iip, twopoint, nparameters
    )
    @inbounds begin
        left = prod(bc_sizes[1])
        parameters = nparameters == 0 ? p : view(y, (size(y, 1) - nparameters + 1):size(y, 1), 1)
        if twopoint isa Val{true}
            right = prod(bc_sizes[2])
            __device_eval!(
                __device_reshape(view(resid, 1:left), bc_sizes[1]), bc[1],
                (__device_reshape(view(y, :, 1), in_size), parameters), iip
            )
            __device_eval!(
                __device_reshape(view(resid, (length(resid) - right + 1):length(resid)), bc_sizes[2]),
                bc[2], (
                    __device_reshape(view(y, :, size(y, 2)), in_size),
                    nparameters == 0 ? p : view(y, (size(y, 1) - nparameters + 1):size(y, 1), size(y, 2)),
                ), iip
            )
        else
            sol = __firk_eval_sol(y, mesh, mesh_dt, coefficients, in_size, stage)
            __device_eval!(
                __device_reshape(view(resid, 1:left), bc_sizes[1]),
                bc, (sol, parameters, mesh), iip
            )
        end
    end
end

function BoundaryValueDiffEqCore.__device_residual!(resid, u, cache::Union{FIRKCacheExpand{iip}, FIRKCacheNested{iip}}, boundary = true, reuse_nested = false) where {iip}
    cache.alg.nested_nlsolve && return __firk_nested_residual!(resid, u, cache, boundary, reuse_nested)
    work = __firk_device_buffers(cache, eltype(u))
    y = reshape(u, size(__firk_states(cache)))
    M, nodes = size(y)
    left = prod(cache.resid_size[1])
    collocation = reshape(view(resid, (left + 1):(left + M * (nodes - 1))), M, nodes - 1)
    (; a, b, c) = cache.TU
    platform = cache.alg.platform
    __firk_packed_collocation_kernel!(platform)(
        collocation, work.tmp, cache.f, y, cache.p,
        cache.mesh, cache.mesh_dt, a, b, c, cache.in_size, Val(iip), cache.singular_term, cache.in_size, 0,
        cache.mass_matrix, cache.algebraic_indices;
        ndrange = length(cache.host_mesh) - 1
    )
    synchronize(platform)
    if boundary
        if !(cache.problem_type isa TwoPointBVProblem)
            __firk_device_interp_setup!(cache, y, work)
        end
        __firk_device_bc_kernel!(platform)(
            resid, cache.bc, y, work.coefficients, cache.p,
            cache.mesh, cache.mesh_dt, cache.in_size, cache.resid_size, cache.TU.s,
            Val(iip), Val(cache.problem_type isa TwoPointBVProblem), cache.nparameters; ndrange = 1
        )
        synchronize(platform)
    end
    return resid
end

# Reduced FIRK on device: the outer unknowns are mesh states. Each work item
# solves one interval's stages, including pivoted dense LU and backtracking.
# No stage vectors, Jacobian blocks or Newton steps cross the device boundary.

__firk_nested_value(x) = x
__firk_nested_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
__firk_nested_value_type(::Type{T}) where {T} = T
__firk_nested_value_type(::Type{<:ForwardDiff.Dual{Tag, T}}) where {Tag, T} = T
__firk_nested_lift(value, derivative::D) where {D <: ForwardDiff.Dual} =
    D(value, ForwardDiff.partials(derivative))

function __firk_nested_options(alg, ::Type{T}, abstol) where {T}
    options = alg.nested_nlsolve_kwargs
    for key in keys(options)
        key in (:abstol, :reltol, :maxiters) || throw(
            ArgumentError(
                "Device nested FIRK accepts abstol, reltol and maxiters in nested_nlsolve_kwargs; unsupported option: $key."
            )
        )
    end
    default_tol = max(10eps(T), min(T(abstol) / 100, sqrt(eps(T)) / 100))
    # Finite-difference outer Jacobians divide residual differences by a small
    # perturbation. Solve stages to roundoff so inner errors are not amplified.
    modes = (alg.jac_alg.diffmode, alg.jac_alg.bc_diffmode, alg.jac_alg.nonbc_diffmode)
    any(mode -> mode !== nothing && get_dense_ad(mode) isa AutoFiniteDiff, modes) &&
        (default_tol = 10eps(T))
    atol = T(get(options, :abstol, default_tol))
    rtol = T(get(options, :reltol, zero(T)))
    maxiters = get(options, :maxiters, 50)
    isfinite(atol) && atol >= 0 && isfinite(rtol) && rtol >= 0 ||
        throw(ArgumentError("Nested tolerances must be finite and nonnegative."))
    maxiters isa Integer && maxiters >= 0 || throw(ArgumentError("Nested maxiters must be a nonnegative integer."))
    return (; abstol = atol, reltol = rtol, maxiters = Int(maxiters))
end

function __firk_nested_diffmode(cache)
    jac = cache.alg.jac_alg
    mode = get_dense_ad(cache.problem_type isa TwoPointBVProblem ? jac.diffmode : jac.nonbc_diffmode)
    return mode isa AutoForwardDiff ? Val(:ad) :
        (mode.fdjtype isa Val{:central} ? Val(:central) : Val(:forward))
end
__firk_nested_jacobian_scalar(::Val{:ad}, ::Type{T}) where {T} =
    ForwardDiff.Dual{FIRKNestedJacobianTag, T, 1}
__firk_nested_jacobian_scalar(::Val, ::Type{T}) where {T} = T

@kernel function __firk_nested_initial_stages!(stages, packed, M, stage)
    index = @index(Global, Linear)
    @inbounds begin
        k = (index - 1) % size(stages, 1) + 1
        i = (index - 1) ÷ size(stages, 1) + 1
        row = (k - 1) % M + 1
        r = (k - 1) ÷ M + 1
        stages[k, i] = packed[row, (i - 1) * (stage + 1) + r + 1]
    end
end

function __firk_nested_buffers(cache, ::Type{D}) where {D}
    return get!(cache.device_cache, Tuple{D, Val{:nested}}) do
        T = __firk_nested_value_type(D)
        M, N, stage = size(__firk_states(cache), 1), length(cache.host_mesh) - 1, cache.TU.s
        Q = M * stage
        J = __firk_nested_jacobian_scalar(__firk_nested_diffmode(cache), T)
        allocate(name, S, dims...) = __firk_work_array(cache, (D, :nested, name), S, dims...)
        stages = allocate(:stages, T, Q, N)
        __firk_nested_initial_stages!(cache.alg.platform)(
            stages, __firk_states(cache), M, stage; ndrange = length(stages)
        )
        synchronize(cache.alg.platform)
        (;
            stages, trial = allocate(:trial, T, Q, N), residual = allocate(:residual, T, Q, N),
            step = allocate(:step, T, Q, N), scratch = allocate(:scratch, T, M, N),
            jac_input = allocate(:jac_input, J, M, N), jac_output = allocate(:jac_output, J, M, N),
            rhs_value = allocate(:rhs_value, T, M, N),
            # The interval is the contiguous dimension of dense Jacobian blocks
            # so neighboring work items access neighboring memory during LU.
            rhs_jacobian = allocate(:rhs_jacobian, T, N, M, M, stage),
            lu = allocate(:lu, T, N, Q, Q), pivots = allocate(:pivots, Int32, Q, N),
            sensitivity = allocate(:sensitivity, D, Q, N), status = allocate(:status, Int32, N),
            iterations = allocate(:iterations, Int32, N),
            packed = D === eltype(cache) ? __firk_states(cache) : allocate(:packed, D, size(__firk_states(cache))...),
        )
    end
end

@inline function __firk_nested_stage_state!(scratch, u, stages, a, h, i, r, M, stage)
    @inbounds for j in 1:M
        v = zero(eltype(stages))
        for s in 1:stage
            v += a[s, r] * stages[(s - 1) * M + j, i]
        end
        scratch[j, i] = __firk_nested_value(u[j, i]) + h * v
    end
    return nothing
end

@inline function __firk_nested_stage_residual!(
        residual, scratch, stages, u, f, p, a, c, h, t, in_size, iip, singular, mass, i
    )
    M, stage = size(scratch, 1), length(c)
    K = __device_reshape(view(stages, :, i), (M, stage))
    @inbounds for r in 1:stage
        __firk_nested_stage_state!(scratch, u, stages, a, h, i, r, M, stage)
        out = view(residual, ((r - 1) * M + 1):(r * M), i)
        __firk_device_rhs!(out, f, view(scratch, :, i), p, t + c[r] * h, in_size, iip, singular)
        for j in 1:M
            out[j] -= __firk_mass_stage(mass, K, j, r)
        end
    end
    return nothing
end

@inline __firk_nested_mass_entry(mass::LinearAlgebra.UniformScaling, j, k) =
    j == k ? mass.λ : zero(mass.λ)
@inline function __firk_nested_mass_entry(mass::AbstractMatrix, j, k)
    j > size(mass, 1) && return j == k ? one(eltype(mass)) : zero(eltype(mass))
    k > size(mass, 2) && return zero(eltype(mass))
    return @inbounds mass[j, k]
end

@inline function __firk_nested_rhs_jacobian!(work, f, p, t, in_size, iip, singular, i, r, ::Val{:ad})
    M = size(work.scratch, 1)
    D = eltype(work.jac_input)
    @inbounds for k in 1:M
        for j in 1:M
            work.jac_input[j, i] = D(
                work.scratch[j, i], ForwardDiff.Partials((j == k ? one(eltype(work.scratch)) : zero(eltype(work.scratch)),))
            )
        end
        __firk_device_rhs!(
            view(work.jac_output, :, i), f, view(work.jac_input, :, i), p,
            t, in_size, iip, singular
        )
        for j in 1:M
            work.rhs_jacobian[i, j, k, r] = ForwardDiff.partials(work.jac_output[j, i])[1]
        end
    end
    return nothing
end

@inline function __firk_nested_rhs_jacobian!(work, f, p, t, in_size, iip, singular, i, r, mode::Val)
    T, M = eltype(work.scratch), size(work.scratch, 1)
    central = mode isa Val{:central}
    relstep = central ? cbrt(eps(T)) : sqrt(eps(T))
    @inbounds for j in 1:M
        work.jac_input[j, i] = work.scratch[j, i]
    end
    if !central
        __firk_device_rhs!(view(work.rhs_value, :, i), f, view(work.jac_input, :, i), p, t, in_size, iip, singular)
    end
    @inbounds for k in 1:M
        value = work.scratch[k, i]
        step = max(abs(value), one(T)) * relstep
        work.jac_input[k, i] = value + step
        __firk_device_rhs!(view(work.jac_output, :, i), f, view(work.jac_input, :, i), p, t, in_size, iip, singular)
        if central
            work.jac_input[k, i] = value - step
            __firk_device_rhs!(view(work.rhs_value, :, i), f, view(work.jac_input, :, i), p, t, in_size, iip, singular)
        end
        for j in 1:M
            work.rhs_jacobian[i, j, k, r] =
                (work.jac_output[j, i] - work.rhs_value[j, i]) / (central ? 2step : step)
        end
        work.jac_input[k, i] = value
    end
    return nothing
end

@inline function __firk_nested_jacobian!(work, u, f, p, a, c, h, t, in_size, iip, singular, mass, mode, i)
    M, stage = size(work.scratch, 1), length(c)
    @inbounds for r in 1:stage
        __firk_nested_stage_state!(work.scratch, u, work.stages, a, h, i, r, M, stage)
        __firk_nested_rhs_jacobian!(work, f, p, t + c[r] * h, in_size, iip, singular, i, r, mode)
        for s in 1:stage, k in 1:M, j in 1:M
            work.lu[i, (r - 1) * M + j, (s - 1) * M + k] =
                h * a[s, r] * work.rhs_jacobian[i, j, k, r] -
                (r == s ? __firk_nested_mass_entry(mass, j, k) : zero(h))
        end
    end
    return nothing
end

# In-place partial-pivoting LU of one small dense stage system. Pivots and
# factors are reused for every direction in the implicit differentiation solve.
@inline function __firk_nested_lu!(A, pivots, i)
    Q = size(A, 2)
    @inbounds for k in 1:Q
        pivot, largest = k, abs(A[i, k, k])
        for row in (k + 1):Q
            value = abs(A[i, row, k])
            if value > largest
                pivot, largest = row, value
            end
        end
        isfinite(largest) && largest > 0 || return false
        pivots[k, i] = pivot
        if pivot != k
            for column in 1:Q
                A[i, k, column], A[i, pivot, column] = A[i, pivot, column], A[i, k, column]
            end
        end
        for row in (k + 1):Q
            A[i, row, k] /= A[i, k, k]
            for column in (k + 1):Q
                A[i, row, column] -= A[i, row, k] * A[i, k, column]
            end
        end
    end
    return true
end

@inline function __firk_nested_ldiv!(rhs, A, pivots, i)
    Q = size(A, 2)
    @inbounds for k in 1:Q
        pivot = pivots[k, i]
        rhs[k, i], rhs[pivot, i] = rhs[pivot, i], rhs[k, i]
    end
    @inbounds for row in 1:Q
        value = rhs[row, i]
        for column in 1:(row - 1)
            value -= A[i, row, column] * rhs[column, i]
        end
        rhs[row, i] = value
    end
    @inbounds for row in Q:-1:1
        value = rhs[row, i]
        for column in (row + 1):Q
            value -= A[i, row, column] * rhs[column, i]
        end
        rhs[row, i] = value / A[i, row, row]
    end
    return nothing
end

@inline function __firk_nested_norm(rhs, i)
    result = zero(eltype(rhs))
    @inbounds for row in axes(rhs, 1)
        value = abs(rhs[row, i])
        isfinite(value) || return oftype(result, Inf)
        result = max(result, value)
    end
    return result
end

@inline function __firk_nested_newton!(work, u, f, p, a, c, h, t, in_size, iip, singular, mass, mode, options, i)
    T = eltype(work.stages)
    __firk_nested_stage_residual!(work.residual, work.scratch, work.stages, u, f, p, a, c, h, t, in_size, iip, singular, mass, i)
    initial = __firk_nested_norm(work.residual, i)
    tolerance = options.abstol + options.reltol * initial
    status, iterations = Int32(1), Int32(0)
    @inbounds for iteration in 0:options.maxiters
        error = __firk_nested_norm(work.residual, i)
        iterations = Int32(iteration)
        if !isfinite(error)
            status = Int32(3)
            break
        elseif error <= tolerance
            status = Int32(0)
            break
        elseif iteration == options.maxiters
            break
        end
        __firk_nested_jacobian!(work, u, f, p, a, c, h, t, in_size, iip, singular, mass, mode, i)
        if !__firk_nested_lu!(work.lu, work.pivots, i)
            status = Int32(2)
            break
        end
        for k in axes(work.stages, 1)
            work.step[k, i] = -work.residual[k, i]
        end
        __firk_nested_ldiv!(work.step, work.lu, work.pivots, i)
        alpha, accepted = one(T), false
        for trial in 1:12
            for k in axes(work.stages, 1)
                work.trial[k, i] = work.stages[k, i] + alpha * work.step[k, i]
            end
            __firk_nested_stage_residual!(work.residual, work.scratch, work.trial, u, f, p, a, c, h, t, in_size, iip, singular, mass, i)
            candidate = __firk_nested_norm(work.residual, i)
            if candidate <= tolerance || candidate < (one(T) - T(1.0e-4) * alpha) * error
                for k in axes(work.stages, 1)
                    work.stages[k, i] = work.trial[k, i]
                end
                accepted = true
                break
            end
            alpha /= 2
        end
        if !accepted
            status = Int32(4)
            break
        end
    end
    work.status[i], work.iterations[i] = status, iterations
    return nothing
end

@inline function __firk_nested_pack_stages!(work, u, i, M, stage, ::Type{T}) where {T}
    ctr = (i - 1) * (stage + 1) + 1
    @inbounds for r in 1:stage, j in 1:M
        work.packed[j, ctr + r] = work.stages[(r - 1) * M + j, i]
    end
    return nothing
end

@inline function __firk_nested_pack_stages!(work, u, i, M, stage, ::Type{D}) where {D <: ForwardDiff.Dual}
    # G(K, y_i) = 0 => G_K dK = -G_y dy_i. Differentiate the converged
    # equations, independent of Newton iteration counts and warm-start history.
    @inbounds for r in 1:stage, j in 1:M
        value = zero(D)
        for k in 1:M
            value -= work.rhs_jacobian[i, j, k, r] * (u[k, i] - __firk_nested_value(u[k, i]))
        end
        work.sensitivity[(r - 1) * M + j, i] = value
    end
    __firk_nested_ldiv!(work.sensitivity, work.lu, work.pivots, i)
    ctr = (i - 1) * (stage + 1) + 1
    @inbounds for r in 1:stage, j in 1:M
        k = (r - 1) * M + j
        work.packed[j, ctr + r] = __firk_nested_lift(work.stages[k, i], work.sensitivity[k, i])
    end
    return nothing
end

@kernel function __firk_nested_stages_kernel!(work, u, f, p, a, c, mesh, mesh_dt, in_size, iip, singular, mass, mode, options, ::Val{reuse}) where {reuse}
    i = @index(Global, Linear)
    @inbounds begin
        M, stage = size(u, 1), length(c)
        h, t = mesh_dt[i], mesh[i]
        if !reuse
            __firk_nested_newton!(work, u, f, p, a, c, h, t, in_size, iip, singular, mass, mode, options, i)
            if eltype(u) <: ForwardDiff.Dual && iszero(work.status[i])
                __firk_nested_jacobian!(work, u, f, p, a, c, h, t, in_size, iip, singular, mass, mode, i)
                __firk_nested_lu!(work.lu, work.pivots, i) || (work.status[i] = Int32(2))
            end
        end
        # Each work item owns its left node and stages. Only the last interval
        # writes the final node, so no adjacent intervals race on packed states.
        ctr = (i - 1) * (stage + 1) + 1
        for j in 1:M
            work.packed[j, ctr] = u[j, i]
            if i == length(mesh_dt)
                work.packed[j, ctr + stage + 1] = u[j, i + 1]
            end
        end
        if iszero(work.status[i])
            __firk_nested_pack_stages!(work, u, i, M, stage, eltype(u))
        else
            for r in 1:stage, j in 1:M
                # Keep finite guesses available for mesh-bisection recovery.
                # Failure is carried by status and the outer residual instead.
                work.packed[j, ctr + r] = eltype(u)(work.stages[(r - 1) * M + j, i])
            end
        end
    end
end

@kernel function __firk_nested_continuity_kernel!(residual, packed, b, mesh_dt, f, p, mesh, in_size, iip, algebraic_indices, status)
    i = @index(Global, Linear)
    @inbounds begin
        M, stage = size(residual, 1), length(b)
        ctr = (i - 1) * (stage + 1) + 1
        out = view(residual, :, i)
        if algebraic_indices !== nothing
            __device_eval!(
                __device_reshape(out, in_size), f,
                (__device_reshape(view(packed, :, ctr + stage + 1), in_size), p, mesh[i + 1]), iip
            )
        end
        for j in 1:M
            algebraic_indices !== nothing && j in algebraic_indices && continue
            value = zero(eltype(packed))
            for r in 1:stage
                value += b[r] * packed[j, ctr + r]
            end
            out[j] = packed[j, ctr + stage + 1] - packed[j, ctr] - mesh_dt[i] * value
        end
        if !iszero(status[i])
            for j in 1:M
                out[j] = eltype(residual)(NaN)
            end
        end
    end
end

function __firk_nested_residual!(resid, u, cache::Union{FIRKCacheExpand{iip}, FIRKCacheNested{iip}}, boundary = true, reuse = false) where {iip}
    work = __firk_nested_buffers(cache, eltype(u))
    platform, TU = cache.alg.platform, cache.TU
    states = reshape(u, size(__firk_unknowns(cache)))
    options = __firk_nested_options(cache.alg, eltype(cache), cache.kwargs.abstol)
    __firk_nested_stages_kernel!(platform)(
        work, states, cache.f, cache.p, TU.a, TU.c, cache.mesh, cache.mesh_dt,
        cache.in_size, Val(iip), cache.singular_term, cache.mass_matrix,
        __firk_nested_diffmode(cache), options, Val(reuse); ndrange = length(cache.host_mesh) - 1
    )
    synchronize(platform)
    M, N = size(states, 1), size(states, 2) - 1
    left = prod(cache.resid_size[1])
    collocation = reshape(view(resid, (left + 1):(left + M * N)), M, N)
    __firk_nested_continuity_kernel!(platform)(
        collocation, work.packed, TU.b, cache.mesh_dt, cache.f, cache.p,
        cache.mesh, cache.in_size, Val(iip), cache.algebraic_indices, work.status; ndrange = N
    )
    synchronize(platform)
    if boundary
        interpolation = __firk_device_buffers(cache, eltype(u))
        if !(cache.problem_type isa TwoPointBVProblem)
            __firk_device_interp_setup!(cache, work.packed, interpolation)
        end
        __firk_device_bc_kernel!(platform)(
            resid, cache.bc, work.packed, interpolation.coefficients, cache.p,
            cache.mesh, cache.mesh_dt, cache.in_size, cache.resid_size, TU.s,
            Val(iip), Val(cache.problem_type isa TwoPointBVProblem), cache.nparameters; ndrange = 1
        )
        synchronize(platform)
    end
    return resid
end

# Host-owned solves preserve the CPU nonlinear/optimization and differentiation
# interfaces while batching the numeric collocation evaluations on the backend.
# Tracers with heap-backed dependency sets remain on the host.
__firk_offload_cache(::CPU, prob, alg, u0, TU) = nothing
__firk_offload_cache(platform::Backend, prob, alg, u0, TU) =
    __firk_offload_cache_impl(platform, prob, alg, u0, TU)
function __firk_offload_cache_impl(platform, prob, alg, u0, TU)
    modes = prob.problem_type isa TwoPointBVProblem ? (alg.jac_alg.diffmode,) :
        (alg.jac_alg.nonbc_diffmode,)
    foreach(__device_validate_ad, modes)
    tune_parameters = haskey(prob.kwargs, :tune_parameters)
    if tune_parameters && !(isinplace(prob) && prob.p isa AbstractVector{<:Number})
        throw(ArgumentError("GPU FIRK parameter tuning requires an in-place RHS and a numeric parameter vector."))
    end
    nparameters = tune_parameters ? length(prob.p) : 0
    f_size = isnothing(prob.f.f_prototype) ? size(u0) : size(prob.f.f_prototype)
    upload(x) = __device_parameter(platform, x)
    return (;
        platform, f = __device_function(prob.f.f), a = upload(TU.a), b = upload(TU.b), c = upload(TU.c),
        p = upload(prob.p), singular_term = upload(prob.singular_term),
        mass_matrix = upload(prob.f.mass_matrix), host_mass_matrix = prob.f.mass_matrix,
        algebraic_indices = upload(__get_algebraic_indices(prob.f.mass_matrix)),
        in_size = size(u0), f_size, nparameters, buffers = Dict{DataType, Any}(),
    )
end
function __firk_offload_buffers(device, ::Type{T}, M, nodes, nr = M) where {T}
    old = get(device.buffers, T, nothing)
    if old === nothing || size(old.y) != (M, nodes) || size(old.residual, 1) != nr
        N = (nodes - 1) ÷ (length(device.b) + 1)
        allocate(dims) = KernelAbstractions.allocate(device.platform, T, dims)
        old = (;
            y = allocate((M, nodes)), residual = allocate((nr, nodes - 1)),
            tmp = allocate((M, N)), mesh = allocate((N + 1,)), mesh_dt = allocate((N,)),
            host_y = Matrix{T}(undef, M, nodes), host_residual = Matrix{T}(undef, nr, nodes - 1),
        )
        device.buffers[T] = old
    end
    return old
end
function __firk_offload_collocation!(residual, cache, y, u, trait, iip, constraint)
    device = cache.device_cache
    M, nodes = cache.M, length(y)
    work = __firk_offload_buffers(device, eltype(u), M, nodes, length(first(residual)))
    for i in eachindex(y)
        copyto!(view(work.host_y, :, i), _collocation_tmp(y[i], u, trait))
    end
    copyto!(work.y, work.host_y)
    copyto!(work.mesh, eltype(u).(cache.mesh))
    copyto!(work.mesh_dt, eltype(u).(cache.mesh_dt))
    __device_copy_parameter!(device.p, cache.p)
    __device_copy_parameter!(device.singular_term, cache.singular_term)
    singular_term = constraint isa Val{true} ? nothing : device.singular_term
    __firk_packed_collocation_kernel!(device.platform)(
        work.residual, work.tmp,
        __device_function(cache.prob.f.f), work.y, device.p, work.mesh, work.mesh_dt,
        device.a, device.b, device.c, device.in_size, iip, singular_term,
        device.f_size, device.nparameters, device.mass_matrix, device.algebraic_indices;
        ndrange = length(cache.mesh_dt)
    )
    synchronize(device.platform)
    copyto!(work.host_residual, work.residual)
    for i in eachindex(residual)
        copyto!(residual[i], view(work.host_residual, :, i))
    end
    return nothing
end

function __firk_offload_nested!(res, K, nest_p, f, TU, p, device, iip)
    T = promote_type(eltype(K), eltype(nest_p))
    if !isbitstype(T)
        if iip isa Val{true}
            return FIRK_nlsolve!(res, K, nest_p, f, TU, p, device.host_mass_matrix)
        end
        copyto!(res, FIRK_nlsolve(K, nest_p, f, TU, p, device.host_mass_matrix))
        return nothing
    end
    M, stage = size(K)
    work = __firk_offload_buffers(device, T, M, stage + 2)
    copyto!(view(work.host_y, :, 1), view(nest_p, 3:length(nest_p)))
    copyto!(view(work.host_y, :, 2:(stage + 1)), K)
    fill!(view(work.host_y, :, stage + 2), zero(T))
    copyto!(work.y, work.host_y)
    copyto!(work.mesh, T[nest_p[1], nest_p[1] + nest_p[2]])
    copyto!(work.mesh_dt, T[nest_p[2]])
    __device_copy_parameter!(device.p, p)
    __firk_packed_collocation_kernel!(device.platform)(
        work.residual, work.tmp,
        device.f, work.y, device.p, work.mesh, work.mesh_dt,
        device.a, device.b, device.c, device.in_size, iip, device.singular_term,
        device.f_size, device.nparameters, device.mass_matrix, device.algebraic_indices; ndrange = 1
    )
    synchronize(device.platform)
    copyto!(work.host_residual, work.residual)
    copyto!(res, view(work.host_residual, :, 2:(stage + 1)))
    return nothing
end
function __firk_offload_nested(K, nest_p, f, TU, p, device)
    res = similar(K, promote_type(eltype(K), eltype(nest_p)))
    __firk_offload_nested!(res, K, nest_p, f, TU, p, device, Val(false))
    return res
end
