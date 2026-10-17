# FIRK Expand Interpolation
struct FIRKExpandInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
    cache
end

function SciMLBase.interp_summary(interp::FIRKExpandInterpolation)
    return "FIRK Order $(interp.cache.order) Interpolation"
end

function (id::FIRKExpandInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::FIRKExpandInterpolation)(
        val, tvals, idxs, deriv, p, continuity::Symbol = :left
    )
    return interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end

# FIRK Nested Interpolation
struct FIRKNestedInterpolation{T1, T2} <: AbstractDiffEqInterpolation
    t::T1
    u::T2
    cache
end

function SciMLBase.interp_summary(interp::FIRKNestedInterpolation)
    return "FIRK Order $(interp.cache.order) Interpolation"
end

function (id::FIRKNestedInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::FIRKNestedInterpolation)(
        val, tvals, idxs, deriv, p, continuity::Symbol = :left
    )
    return interpolation!(val, tvals, id, idxs, deriv, p, continuity)
end

@inline function interpolation(
        tvals, id::FIRKNestedInterpolation, idxs,
        deriv, p, continuity::Symbol = :left
    )
    (; t, u, cache) = id
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    if idxs isa Number
        vals = Vector{eltype(first(u))}(undef, length(tvals))
    elseif idxs isa AbstractVector
        vals = Vector{Vector{eltype(first(u))}}(undef, length(tvals))
    else
        vals = Vector{eltype(u)}(undef, length(tvals))
    end

    for j in idx
        z = similar(cache.fᵢ₂_cache)
        interpolant!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt, deriv)
        vals[j] = idxs !== nothing ? z[idxs] : z
    end
    return DiffEqArray(vals, tvals)
end

@inline function interpolation!(
        vals, tvals, id::FIRKNestedInterpolation, idxs,
        deriv, p, continuity::Symbol = :left
    )
    (; t, cache) = id
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    for j in idx
        z = similar(cache.fᵢ₂_cache)
        interpolant!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt, deriv)
        vals[j] = z
    end
    return
end

@inline function interpolation(
        tval::Number, id::FIRKNestedInterpolation, idxs,
        deriv, p, continuity::Symbol = :left
    )
    z = similar(id.cache.fᵢ₂_cache)
    interpolant!(z, id.cache, tval, id.cache.mesh, id.cache.mesh_dt, deriv)
    return idxs !== nothing ? z[idxs] : z
end

@inline function interpolant!(
        z::AbstractArray, cache::FIRKCacheNested{iip, T, diffcache, fit_parameters},
        t, mesh, mesh_dt, ::Type{Val{0}}
    ) where {iip, T, diffcache, fit_parameters}
    (; f, ITU, nest_prob, alg) = cache
    (; q_coeff) = ITU

    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀.u) - 1) / (length(cache.y) - 1)
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])
    length_z = length(z)

    nest_nlsolve_alg = __concrete_solve_algorithm(nest_prob, alg.nlsolve)
    nestprob_p = zeros(T, cache.M + 2)

    yᵢ = copy(cache.y[j].du)
    yᵢ₊₁ = copy(cache.y[j + 1].du)

    if iip
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, cache.p, mesh[j])
        f(dyᵢ₊₁, yᵢ₊₁, cache.p, mesh[j + 1])
    else
        dyᵢ = f(yᵢ, cache.p, mesh[j])
        dyᵢ₊₁ = f(yᵢ₊₁, cache.p, mesh[j + 1])
    end

    nestprob_p[1] = mesh[j]
    nestprob_p[2] = mesh_dt[j]
    nestprob_p[3:end] .= ifelse(fit_parameters, vcat(yᵢ, __tunable_part(cache.p)), yᵢ)

    _nestprob = remake(nest_prob, p = nestprob_p)
    nestsol = __solve(_nestprob, nest_nlsolve_alg; alg.nested_nlsolve_kwargs...)
    K = nestsol.u

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, @view(K[1:length_z, :])) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    return S_interpolate!(z, τ, S_coeffs)
end

@inline function interpolant!(
        dz::AbstractArray, cache::FIRKCacheNested{iip, T, diffcache, fit_parameters},
        t, mesh, mesh_dt, ::Type{Val{1}}
    ) where {iip, T, diffcache, fit_parameters}
    (; f, ITU, nest_prob, alg) = cache
    (; q_coeff) = ITU

    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀.u) - 1) / (length(cache.y) - 1)
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])
    length_dz = length(dz)

    nest_nlsolve_alg = __concrete_solve_algorithm(nest_prob, alg.nlsolve)
    nestprob_p = zeros(T, cache.M + 2)

    yᵢ = copy(cache.y[j].du)
    yᵢ₊₁ = copy(cache.y[j + 1].du)

    if iip
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, cache.p, mesh[j])
        f(dyᵢ₊₁, yᵢ₊₁, cache.p, mesh[j + 1])
    else
        dyᵢ = f(yᵢ, cache.p, mesh[j])
        dyᵢ₊₁ = f(yᵢ₊₁, cache.p, mesh[j + 1])
    end

    nestprob_p[1] = mesh[j]
    nestprob_p[2] = mesh_dt[j]
    nestprob_p[3:end] .= ifelse(fit_parameters, vcat(yᵢ, __tunable_part(cache.p)), yᵢ)

    _nestprob = remake(nest_prob, p = nestprob_p)
    nestsol = __solve(_nestprob, nest_nlsolve_alg; alg.nested_nlsolve_kwargs...)
    K = nestsol.u

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, @view(K[1:length_dz, :]))
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    return dS_interpolate!(dz, τ, S_coeffs)
end

## Expanded
@inline function interpolation(
        tvals, id::FIRKExpandInterpolation, idxs,
        deriv, p, continuity::Symbol = :left
    )
    (; t, u, cache) = id
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    if idxs isa Number
        vals = Vector{eltype(first(u))}(undef, length(tvals))
    elseif idxs isa AbstractVector
        vals = Vector{Vector{eltype(first(u))}}(undef, length(tvals))
    else
        vals = Vector{eltype(u)}(undef, length(tvals))
    end

    for j in idx
        z = similar(cache.fᵢ₂_cache)
        interpolant!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt, deriv)
        vals[j] = idxs !== nothing ? z[idxs] : z
    end
    return DiffEqArray(vals, tvals)
end

@inline function interpolation!(
        vals, tvals, id::FIRKExpandInterpolation, idxs,
        deriv, p, continuity::Symbol = :left
    )
    (; t, cache) = id
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    for j in idx
        z = similar(cache.fᵢ₂_cache)
        interpolant!(z, id.cache, tvals[j], id.cache.mesh, id.cache.mesh_dt, deriv)
        vals[j] = z
    end
    return
end

@inline function interpolation(
        tval::Number, id::FIRKExpandInterpolation, idxs,
        deriv, p, continuity::Symbol = :left
    )
    z = similar(id.cache.fᵢ₂_cache)
    interpolant!(z, id.cache, tval, id.cache.mesh, id.cache.mesh_dt, deriv)
    return idxs !== nothing ? z[idxs] : z
end

@inline function interpolant!(
        z::AbstractArray, cache::FIRKCacheExpand{iip},
        t, mesh, mesh_dt, ::Type{Val{0}}
    ) where {iip}
    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀.u) - 1) / (length(cache.y) - 1)
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])
    length_z = length(z)

    (; f, M, stage, p, ITU) = cache
    (; q_coeff) = ITU

    K = safe_similar(cache.y[1].du, M, stage)

    ctr_y = (j - 1) * (stage + 1) + 1

    yᵢ = cache.y[ctr_y].du
    yᵢ₊₁ = cache.y[ctr_y + stage + 1].du

    if iip
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, p, mesh[j])
        f(dyᵢ₊₁, yᵢ₊₁, p, mesh[j + 1])
    else
        dyᵢ = f(yᵢ, p, mesh[j])
        dyᵢ₊₁ = f(yᵢ₊₁, p, mesh[j + 1])
    end

    # Load interpolation residual
    for jj in 1:stage
        K[1:length_z, jj] = cache.y[ctr_y + jj].du
    end

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, @view(K[1:length_z, :])) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    return S_interpolate!(z, τ, S_coeffs)
end

@inline function interpolant!(
        dz::AbstractArray, cache::FIRKCacheExpand{iip},
        t, mesh, mesh_dt, ::Type{Val{1}}
    ) where {iip}
    j = interval(mesh, t)
    h = mesh_dt[j]
    lf = (length(cache.y₀.u) - 1) / (length(cache.y) - 1)
    if lf > 1
        h *= lf
    end
    τ = (t - mesh[j])
    length_dz = length(dz)

    (; f, M, stage, p, ITU) = cache
    (; q_coeff) = ITU

    K = safe_similar(cache.y[1].du, M, stage)

    ctr_y = (j - 1) * (stage + 1) + 1

    yᵢ = cache.y[ctr_y].du
    yᵢ₊₁ = cache.y[ctr_y + stage + 1].du

    if iip
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, p, mesh[j])
        f(dyᵢ₊₁, yᵢ₊₁, p, mesh[j + 1])
    else
        dyᵢ = f(yᵢ, p, mesh[j])
        dyᵢ₊₁ = f(yᵢ₊₁, p, mesh[j + 1])
    end

    # Load interpolation residual
    for jj in 1:stage
        K[1:length_dz, jj] = cache.y[ctr_y + jj].du
    end

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, @view(K[1:length_dz, :])) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    return dS_interpolate!(dz, τ, S_coeffs)
end

@inline __build_interpolation(
    cache::FIRKCacheExpand,
    u::AbstractVector
) = FIRKExpandInterpolation(cache.mesh, u, cache)
@inline __build_interpolation(
    cache::FIRKCacheNested,
    u::AbstractVector
) = FIRKNestedInterpolation(cache.mesh, u, cache)

# Intermediate solution for evaluating boundary conditions
# basically simplified version of the interpolation for FIRK
# Expanded FIRK
function (s::EvalSol{C})(tval::Number) where {C <: FIRKCacheExpand}
    (; t, u, cache) = s
    (; f, alg, ITU, mesh_dt, p) = cache
    (; q_coeff) = ITU
    stage = alg_stage(alg)
    # Quick handle for the case where tval is at the boundary
    (tval == t[1]) && return first(u)
    (tval == t[end]) && return last(u)
    K = safe_similar(first(u), length(first(u)), stage)
    j = interval(t, tval)
    ctr_y = (j - 1) * (stage + 1) + 1

    yᵢ = u[ctr_y]
    yᵢ₊₁ = u[ctr_y + stage + 1]

    if SciMLBase.isinplace(cache.prob)
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, p, t[j])
        f(dyᵢ₊₁, yᵢ₊₁, p, t[j + 1])
    else
        dyᵢ = f(yᵢ, p, t[j])
        dyᵢ₊₁ = f(yᵢ₊₁, p, t[j + 1])
    end

    # Load interpolation residual
    for jj in 1:stage
        K[:, jj] = u[ctr_y + jj]
    end
    h = mesh_dt[j]
    τ = tval - t[j]

    M = size(K, 1)
    z₁, z₁′ = similar(yᵢ), similar(yᵢ₊₁)
    for i in 1:M
        ki = @view K[i, :]
        coeffs = get_q_coeffs_interp(q_coeff, ki, h)
        z₁[i] = yᵢ[i] + sum(coeffs[ii] * (τ * h)^(ii) for ii in axes(coeffs, 1))
        z₁′[i] = sum(ii * coeffs[ii] * (τ * h)^(ii - 1) for ii in axes(coeffs, 1))
    end

    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)

    z = similar(yᵢ)

    S_interpolate!(z, τ, S_coeffs)
    return z
end

function get_q_coeffs_interp(A, ki, h)
    coeffs = A * ki
    for i in axes(coeffs, 1)
        coeffs[i] = coeffs[i] / (h^(i - 1))
    end
    return coeffs
end


# Nested FIRK
function (s::EvalSol{C})(tval::Number) where {C <: FIRKCacheNested}
    (; t, u, cache) = s
    (; f, nest_prob, alg, mesh_dt, p, ITU) = cache
    (; q_coeff) = ITU
    stage = alg_stage(alg)
    # Quick handle for the case where tval is at the boundary
    (tval == t[1]) && return first(u)
    (tval == t[end]) && return last(u)
    j = interval(t, tval)
    h = mesh_dt[j]
    τ = tval - t[j]

    nest_nlsolve_alg = __concrete_solve_algorithm(nest_prob, alg.nlsolve)
    nestprob_p = zeros(cache.M + 2)

    yᵢ = u[j]
    yᵢ₊₁ = u[j + 1]

    if SciMLBase.isinplace(cache.prob)
        dyᵢ = similar(yᵢ)
        dyᵢ₊₁ = similar(yᵢ₊₁)

        f(dyᵢ, yᵢ, p, t[j])
        f(dyᵢ₊₁, yᵢ₊₁, p, t[j + 1])
    else
        dyᵢ = f(yᵢ, p, t[j])
        dyᵢ₊₁ = f(yᵢ₊₁, p, t[j + 1])
    end

    nestprob_p[1] = t[j]
    nestprob_p[2] = mesh_dt[j]
    nestprob_p[3:end] .= nodual_value(yᵢ)

    # TODO: Better initial guess or nestprob
    _nestprob = remake(nest_prob, p = nestprob_p, u0 = zeros(length(first(u)), stage))
    nestsol = __solve(_nestprob, nest_nlsolve_alg; alg.nested_nlsolve_kwargs...)
    K = nestsol.u

    z₁, z₁′ = eval_q(yᵢ, 0.5, h, q_coeff, K) # Evaluate q(x) at midpoints
    S_coeffs = get_S_coeffs(h, yᵢ, yᵢ₊₁, z₁, dyᵢ, dyᵢ₊₁, z₁′)
    z = similar(yᵢ)
    S_interpolate!(z, τ, S_coeffs)
    return z
end

# Normalized quintic Hermite interpolation with values and derivatives at both
# ends and at the collocation polynomial's midpoint. A fixed isbits matrix is
# usable in GPU kernels, including with ForwardDiff dual numbers.
const FIRK_QUINTIC_INVERSE = round.(inv(s_constraints_block(1.0)))

@inline function __firk_q_component(y, A, j, ctr, stage, h, τ)
    value, deriv = zero(eltype(y)), zero(eltype(y))
    for k in axes(A, 1)
        coefficient = zero(eltype(y))
        for r in 1:stage
            coefficient += A[k, r] * y[j, ctr + r]
        end
        value += coefficient * τ^k
        deriv += k * coefficient * τ^(k - 1)
    end
    return y[j, ctr] + h * value, deriv
end

@kernel function __firk_interp_setup_kernel!(
        coefficients, endpoints, y, f, p, mesh, mesh_dt, A, stage, in_size, iip, singular_term, mass_matrix
    )
    i = @index(Global, Linear)
    @inbounds begin
        ctr = (i - 1) * (stage + 1) + 1
        h = mesh_dt[i]
        left, right = view(y, :, ctr), view(y, :, ctr + stage + 1)
        dl, dr = view(endpoints, :, 1, i), view(endpoints, :, 2, i)
        if mass_matrix isa LinearAlgebra.UniformScaling
            __firk_device_rhs!(dl, f, left, p, mesh[i], in_size, iip, singular_term)
            __firk_device_rhs!(dr, f, right, p, mesh[i + 1], in_size, iip, singular_term)
            for j in axes(y, 1)
                dl[j] /= mass_matrix.λ
                dr[j] /= mass_matrix.λ
            end
        else
            # The collocation stages are derivatives even for singular mass
            # matrices; use their polynomial to avoid inverting the mass matrix.
            for j in axes(y, 1)
                _, dl[j] = __firk_q_component(y, A, j, ctr, stage, h, zero(h))
                _, dr[j] = __firk_q_component(y, A, j, ctr, stage, h, one(h))
            end
        end
        for j in axes(y, 1)
            mid, dmid = __firk_q_component(y, A, j, ctr, stage, h, one(h) / 2)
            # Time and state types may differ (e.g. Float64 times with Float32
            # states). Keep the small tuple homogeneous for device indexing.
            T = eltype(coefficients)
            rhs = (T(left[j]), T(right[j]), T(mid), T(h * dl[j]), T(h * dr[j]), T(h * dmid))
            for k in 1:6
                v = zero(eltype(coefficients))
                for r in 1:6
                    v += convert(eltype(A), FIRK_QUINTIC_INVERSE[k, r]) * rhs[r]
                end
                coefficients[j, k, i] = v
            end
        end
    end
end
function __firk_device_interp_setup!(cache::Union{FIRKCacheExpand{iip}, FIRKCacheNested{iip}}, y, work) where {iip}
    platform = cache.alg.platform
    __firk_interp_setup_kernel!(platform)(
        work.coefficients, work.endpoints, y,
        cache.f, cache.p, cache.mesh, cache.mesh_dt, cache.ITU.q_coeff,
        cache.TU.s, cache.in_size, Val(iip), cache.singular_term, cache.mass_matrix;
        ndrange = length(cache.host_mesh) - 1
    )
    synchronize(platform)
    return work.coefficients
end

@concrete struct FIRKDeviceInterpolation <: AbstractDiffEqInterpolation
    t
    u
    mesh_dt
    coefficients
    in_size
    stage::Int
    platform
end
SciMLBase.interp_summary(::FIRKDeviceInterpolation) = "FIRK quintic device interpolation"

struct FIRKDeviceMeshValues{I, D}
    interp::I
    deriv::D
end
Base.length(u::FIRKDeviceMeshValues) = length(u.interp.t)
Base.size(u::FIRKDeviceMeshValues) = (length(u),)
Base.firstindex(::FIRKDeviceMeshValues) = 1
Base.lastindex(u::FIRKDeviceMeshValues) = length(u)
Base.eachindex(u::FIRKDeviceMeshValues) = Base.OneTo(length(u))
Base.@propagate_inbounds Base.getindex(u::FIRKDeviceMeshValues{I, Val{0}}, i::Int) where {I} =
    __device_reshape(view(u.interp.u, :, (i - 1) * (u.interp.stage + 1) + 1), u.interp.in_size)
Base.@propagate_inbounds Base.getindex(u::FIRKDeviceMeshValues{I, Val{1}}, i::Int) where {I} =
    __firk_interpolated(u.interp, u.interp.t[i], Val(1))
Base.first(u::FIRKDeviceMeshValues) = u[1]
Base.last(u::FIRKDeviceMeshValues) = u[length(u)]
@inline Base.iterate(u::FIRKDeviceMeshValues, i::Int = 1) = i > length(u) ? nothing : (u[i], i + 1)

@inline function __firk_eval_sol(y, mesh, mesh_dt, coefficients, in_size, stage)
    interp = FIRKDeviceInterpolation(mesh, y, mesh_dt, coefficients, in_size, stage, nothing)
    return EvalSol(FIRKDeviceMeshValues(interp, Val(0)), mesh, interp)
end
@inline function Base.getproperty(sol::EvalSol{<:FIRKDeviceInterpolation}, name::Symbol)
    name === :du && return FIRKDeviceMeshValues(getfield(sol, :cache), Val(1))
    return getfield(sol, name)
end
Base.size(sol::EvalSol{<:FIRKDeviceInterpolation}) = (sol.cache.in_size..., length(sol.t))
Base.firstindex(::EvalSol{<:FIRKDeviceInterpolation}, d::Int) = 1
Base.lastindex(sol::EvalSol{<:FIRKDeviceInterpolation}, d::Int) = size(sol, d)
Base.@propagate_inbounds Base.getindex(sol::EvalSol{<:FIRKDeviceInterpolation}, i::Int) = sol.u[i]
Base.@propagate_inbounds Base.getindex(sol::EvalSol{<:FIRKDeviceInterpolation}, ::Colon, i::Int) = sol.u[i]
Base.@propagate_inbounds Base.getindex(sol::EvalSol{<:FIRKDeviceInterpolation}, i::Int, j::Int) = sol.u[j][i]
Base.@propagate_inbounds function Base.getindex(
        sol::EvalSol{<:FIRKDeviceInterpolation}, indices::Vararg{Int, N}
    ) where {N}
    dims = sol.cache.in_size
    @boundscheck N == length(dims) + 1 || throw(BoundsError(sol, indices))
    row, stride = indices[1], dims[1]
    for d in 2:(N - 1)
        row += (indices[d] - 1) * stride
        stride *= dims[d]
    end
    return sol.cache.u[row, (indices[N] - 1) * (sol.cache.stage + 1) + 1]
end
Base.@propagate_inbounds Base.getindex(sol::EvalSol{<:FIRKDeviceInterpolation}, ::Colon, ::Colon, node::Int) =
    sol.u[node]
@inline Base.iterate(sol::EvalSol{<:FIRKDeviceInterpolation}, i::Int = 1) =
    i > length(sol.u) ? nothing : (sol.u[i], i + 1)

# Binary search preserves the CPU interpolant's left continuity.
@inline function __firk_device_interval(mesh, t)
    lo, hi = 1, length(mesh)
    @inbounds increasing = mesh[hi] >= mesh[lo]
    while lo < hi
        mid = (lo + hi) >>> 1
        @inbounds before = increasing ? mesh[mid] < t : mesh[mid] > t
        if before
            lo = mid + 1
        else
            hi = mid
        end
    end
    return min(max(lo - 1, 1), length(mesh) - 1)
end
struct FIRKDeviceInterpolatedArray{T, N, I, S, D} <: AbstractArray{T, N}
    interp::I
    τ::S
    interval::Int
    endpoint::Int
    deriv::D
end
Base.size(u::FIRKDeviceInterpolatedArray) = u.interp.in_size
Base.IndexStyle(::Type{<:FIRKDeviceInterpolatedArray}) = IndexLinear()
@inline function __firk_interpolated(id, t, deriv::Union{Val{0}, Val{1}})
    i = __firk_device_interval(id.t, t)
    @inbounds τ = (t - id.t[i]) / id.mesh_dt[i]
    @inbounds endpoint = deriv isa Val{0} ?
        (t == id.t[i] ? i : (t == id.t[i + 1] ? i + 1 : 0)) : 0
    return FIRKDeviceInterpolatedArray{eltype(id.u), length(id.in_size), typeof(id), typeof(τ), typeof(deriv)}(
        id, τ, i, endpoint, deriv
    )
end
Base.@propagate_inbounds function Base.getindex(u::FIRKDeviceInterpolatedArray, j::Int)
    id, i = u.interp, u.interval
    u.endpoint != 0 && return id.u[j, (u.endpoint - 1) * (id.stage + 1) + 1]
    result = zero(eltype(u))
    if u.deriv isa Val{0}
        for k in 6:-1:1
            result = result * u.τ + id.coefficients[j, k, i]
        end
    else
        for k in 6:-1:2
            result = result * u.τ + (k - 1) * id.coefficients[j, k, i]
        end
        result /= id.mesh_dt[i]
    end
    return result
end
@inline (sol::EvalSol{<:FIRKDeviceInterpolation})(t::Number, d::Union{Val{0}, Val{1}} = Val(0)) =
    __firk_interpolated(sol.cache, t, d)
@inline (sol::EvalSol{<:FIRKDeviceInterpolation})(t::Number, ::Type{Val{D}}) where {D} = sol(t, Val(D))

@kernel function __firk_interpolate_kernel!(out, y, mesh, mesh_dt, coefficients, in_size, stage, t, deriv, idxs)
    j = @index(Global, Linear)
    @inbounds begin
        id = FIRKDeviceInterpolation(mesh, y, mesh_dt, coefficients, in_size, stage, nothing)
        row = idxs === nothing ? j : (idxs isa Integer ? idxs : idxs[j])
        out[j] = __firk_interpolated(id, t, deriv)[row]
    end
end
function (id::FIRKDeviceInterpolation)(out, t::Number, idxs::Union{Nothing, Integer, AbstractArray, Tuple}, ::Union{Type{Val{D}}, Val{D}}, p, continuity::Symbol = :left) where {D}
    D in (0, 1) || throw(ArgumentError("FIRK interpolation supports derivatives of order zero or one."))
    expected = idxs === nothing ? prod(id.in_size) : (idxs isa Integer ? 1 : length(idxs))
    length(out) == expected || throw(DimensionMismatch("Interpolation output has the wrong length."))
    if idxs !== nothing
        indices = idxs isa Integer ? (idxs,) : idxs
        all(i -> i isa Integer && 1 <= i <= prod(id.in_size), indices) || throw(BoundsError(id, idxs))
    end
    typeof(__device_initial_backend(out)) === typeof(id.platform) ||
        throw(ArgumentError("Interpolation output must use the solution backend."))
    indices = idxs isa AbstractArray ? __device_parameter(id.platform, idxs) : idxs
    __firk_interpolate_kernel!(id.platform)(
        out, id.u, id.t, id.mesh_dt, id.coefficients,
        id.in_size, id.stage, t, Val(D), indices; ndrange = length(out)
    )
    synchronize(id.platform)
    return out
end
function (id::FIRKDeviceInterpolation)(t::Number, idxs, deriv::Union{Type{<:Val}, Val}, p, continuity::Symbol = :left)
    dims = idxs === nothing ? id.in_size : (idxs isa Integer ? (1,) : (length(idxs),))
    out = similar(id.u, eltype(id.u), dims)
    id(out, t, idxs, deriv, p, continuity)
    return idxs isa Integer ? sum(out) : out
end
function (id::FIRKDeviceInterpolation)(times, idxs, deriv::Union{Type{<:Val}, Val}, p, continuity::Symbol = :left)
    ts = collect(times)
    return DiffEqArray([id(t, idxs, deriv, p, continuity) for t in ts], ts)
end
function (id::FIRKDeviceInterpolation)(out, times, idxs::Union{Nothing, Integer, AbstractArray, Tuple}, deriv::Union{Type{<:Val}, Val}, p, continuity::Symbol = :left)
    for (i, t) in enumerate(times)
        id(out[i], t, idxs, deriv, p, continuity)
    end
    return out
end
