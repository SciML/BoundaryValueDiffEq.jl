# MIRK Interpolation
@concrete struct MIRKInterpolation{C} <: AbstractDiffEqInterpolation
    t
    u
    cache::C
end

function SciMLBase.interp_summary(interp::MIRKInterpolation{<:MIRKCache})
    return "MIRK Order $(interp.cache.order) Interpolation"
end

__has_control_variables(cache::MIRKCache, length_z) = !isnothing(cache.f_prototype) &&
    length(cache.f_prototype) < length_z
__state_variable_count(cache::MIRKCache, length_z) = __has_control_variables(cache, length_z) ?
    length(cache.f_prototype) : length_z

@inline __mirk_interp_values(value, ::DiffCacheNeeded) = value.du
@inline __mirk_interp_values(value, ::NoDiffCacheNeeded) = value

function (id::MIRKInterpolation)(
        tvals, idxs, deriv::Union{Type{<:Val}, Val}, p, continuity::Symbol = :left
    )
    return interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::MIRKInterpolation)(
        val, tvals, idxs, deriv::Union{Type{<:Val}, Val}, p, continuity::Symbol = :left
    )
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
    return
end

# A Val parameter and explicit continuity overlap with the in-place call's
# derivative and parameter positions; this intersection is an out-of-place call.
function (id::MIRKInterpolation)(
        tvals, idxs, deriv::Union{Type{<:Val}, Val}, p::Union{Type{<:Val}, Val},
        continuity::Symbol
    )
    return interpolation(tvals, id, idxs, deriv, p, continuity)
end

@inline function interpolation(
        tvals, id::MIRKInterpolation{<:MIRKCache}, idxs, deriv,
        p, continuity::Symbol = :left
    )
    (; t, u, cache) = id
    (; mesh, mesh_dt) = cache
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
        interpolant!(z, id, cache, tvals[j], mesh, mesh_dt, deriv)
        vals[j] = idxs !== nothing ? z[idxs] : z
    end
    return DiffEqArray(vals, tvals)
end

@inline function interpolation!(
        vals, tvals, id::MIRKInterpolation{<:MIRKCache}, idxs,
        deriv, p, continuity::Symbol = :left
    )
    (; t, cache) = id
    (; mesh, mesh_dt) = cache
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    for j in idx
        z = similar(id.u[1])
        interpolant!(z, id, cache, tvals[j], mesh, mesh_dt, deriv)
        vals[j] = z
    end
    return
end

@inline function interpolation(
        tval::Number, id::MIRKInterpolation{<:MIRKCache}, idxs,
        deriv, p, continuity::Symbol = :left
    )
    z = similar(id.u[1])
    interpolant!(z, id, id.cache, tval, id.cache.mesh, id.cache.mesh_dt, deriv)
    return idxs !== nothing ? z[idxs] : z
end

@inline function interpolant!(
        z::AbstractArray, id::MIRKInterpolation{<:MIRKCache}, cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{0}}
    )
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    w, _ = interp_weights(τ, cache.alg)
    return sum_stages!(z, id, cache, w, i, τ, T)
end

@inline function interpolant!(
        dz::AbstractArray, id::MIRKInterpolation{<:MIRKCache},
        cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{1}}
    )
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    _, w′ = interp_weights(τ, cache.alg)
    return sum_stages!(dz, id, cache, w′, i, τ, T)
end

@views function sum_stages!(
        z::AbstractArray, id::MIRKInterpolation{<:MIRKCache},
        cache::MIRKCache{iip, T, use_both, diffcache},
        w, i::Int, τ, ::Type{Val{0}}
    ) where {iip, T, use_both, diffcache}
    (; stage, k_discrete, k_interp, M) = cache
    (; s_star) = cache.ITU
    dt = cache.mesh_dt[i]

    has_control = __has_control_variables(cache, length(z))

    # state variables have their interpolation polynomials
    length_z = __state_variable_count(cache, length(z))
    z .= zero(z)
    __maybe_matmul!(
        z[1:length_z], __mirk_interp_values(k_discrete[i], diffcache())[1:length_z, 1:stage],
        w[1:stage]
    )
    __maybe_matmul!(
        z[1:length_z], k_interp.u[i][1:length_z, 1:(s_star - stage)],
        w[(stage + 1):s_star], true, true
    )

    # control variable just use linear interpolation
    if has_control
        inc = τ / dt .* (id.u[i + 1] .- id.u[i])
        copyto!(z, (length_z + 1):M, inc, (length_z + 1):M)
    end
    z .= z .* dt .+ id.u[i]

    return nothing
end
@views function sum_stages!(
        z′, id::MIRKInterpolation{<:MIRKCache}, cache::MIRKCache{iip, T, use_both, diffcache},
        w′, i::Int, τ, ::Type{Val{1}}
    ) where {iip, T, use_both, diffcache}
    (; stage, k_discrete, k_interp, M) = cache
    (; s_star) = cache.ITU
    has_control = __has_control_variables(cache, length(z′))
    length_z = __state_variable_count(cache, length(z′))

    z′ .= zero(z′)
    __maybe_matmul!(
        z′[1:length_z], __mirk_interp_values(k_discrete[i], diffcache())[1:length_z, 1:stage],
        w′[1:stage]
    )
    __maybe_matmul!(
        z′[1:length_z], k_interp.u[i][1:length_z, 1:(s_star - stage)],
        w′[(stage + 1):s_star], true, true
    )

    # control variable just use linear interpolation
    if has_control
        inc = (id.u[i + 1] .- id.u[i]) ./ cache.mesh_dt[i]
        copyto!(z′, (length_z + 1):M, inc, (length_z + 1):M)
    end

    return nothing
end

@inline __build_interpolation(cache::MIRKCache, u::AbstractVector) = MIRKInterpolation(cache.mesh, u, cache)

"""
    EvalSol

Intermediate solution for evaluating boundary conditions.
It contains the discrete solution, discrete stages and new stages for interpolation.
"""
function (s::EvalSol{C})(tval::Number) where {C <: MIRKCache}
    (; t, u, cache) = s
    (; alg, stage, k_discrete, k_interp, M) = cache
    # Quick handle for the case where tval is at the boundary
    (tval == t[1]) && return first(u)
    (tval == t[end]) && return last(u)
    z = zero(last(u))
    has_control = __has_control_variables(cache, length(z))
    length_z = __state_variable_count(cache, length(z))
    ii = interval(t, tval)
    dt = cache.mesh_dt[ii]
    τ = (tval - t[ii]) / dt
    w, _ = interp_weights(τ, alg)
    K = __needs_diffcache(alg.jac_alg) ? @view(k_discrete[ii].du[:, 1:stage]) :
        @view(k_discrete[ii][:, 1:stage])
    KI = @view(k_interp.u[ii][1:length_z, 1:(cache.ITU.s_star - stage)])
    __maybe_matmul!(@view(z[1:length_z]), K, @view(w[1:stage]))
    __maybe_matmul!(@view(z[1:length_z]), KI, @view(w[(stage + 1):cache.ITU.s_star]), true, true)

    # control variable just use linear interpolation
    if has_control
        inc = τ / dt .* (u[ii + 1] .- u[ii])
        copyto!(z, (length_z + 1):M, inc, (length_z + 1):M)
    end

    z .= z .* dt .+ u[ii]

    return z
end

# Interpolate intermediate solution at multiple points
function (s::EvalSol{C})(tvals::AbstractArray{<:Number}) where {C <: MIRKCache}
    (; t, u, cache) = s
    (; alg, stage, k_discrete, k_interp, mesh_dt, M) = cache
    # Quick handle for the case where tval is at the boundary
    zvals = [zero(last(u)) for _ in tvals]
    has_control = __has_control_variables(cache, length(first(zvals)))
    length_z = __state_variable_count(cache, length(first(zvals)))
    for (i, tval) in enumerate(tvals)
        (tval == t[1]) && return first(u)
        (tval == t[end]) && return last(u)
        ii = interval(t, tval)
        dt = mesh_dt[ii]
        τ = (tval - t[ii]) / dt
        w, _ = interp_weights(τ, alg)
        K = __needs_diffcache(alg.jac_alg) ? @view(k_discrete[ii].du[:, 1:stage]) :
            @view(k_discrete[ii][:, 1:stage])
        KI = @view(k_interp.u[ii][1:length_z, 1:(cache.ITU.s_star - stage)])
        __maybe_matmul!(@view(zvals[i][1:length_z]), K, @view(w[1:stage]))
        __maybe_matmul!(
            @view(zvals[i][1:length_z]), KI, @view(w[(stage + 1):cache.ITU.s_star]), true, true
        )

        # control variable just use linear interpolation
        if has_control
            inc = τ / dt .* (u[ii + 1] .- u[ii])
            copyto!(zvals[i], (length_z + 1):M, inc, (length_z + 1):M)
        end
        zvals[i] .= zvals[i] .* dt .+ u[ii]
    end
    return zvals
end

# Intermediate derivative solution for evaluating derivative boundary conditions
function (s::EvalSol{C})(tval::Number, ::Type{Val{1}}) where {C <: MIRKCache}
    (; t, cache) = s
    (; alg, stage, k_discrete, k_interp, mesh_dt) = cache
    z′ = zeros(typeof(tval), cache.M)
    ii = interval(t, tval)
    dt = mesh_dt[ii]
    τ = (tval - t[ii]) / dt
    _, w′ = interp_weights(τ, alg)
    __maybe_matmul!(z′, @view(k_discrete[ii].du[:, 1:stage]), @view(w′[1:stage]))
    __maybe_matmul!(
        z′, @view(k_interp.u[ii][:, 1:(cache.ITU.s_star - stage)]), @view(w′[(stage + 1):cache.ITU.s_star]),
        true, true
    )
    return z′
end

"""
    interp_setup!(cache::MIRKCache)

Prepare the extra stages in `k_interp` for interpolant construction.
"""
function interp_setup!(
        cache::MIRKCache{iip, T, use_both, diffcache}
    ) where {iip, T, use_both, diffcache}
    (; x_star, c_star, v_star) = cache.ITU
    (; k_interp, k_discrete, f, new_stages, y, p, mesh, mesh_dt) = cache
    trait = diffcache()
    # Preserve the CPU interpolation RHS, including its existing wrappers.
    for i in eachindex(new_stages.u)
        __mirk_interp_interval_values!(
            k_interp.u[i], new_stages.u[i], __mirk_interp_values(k_discrete[i], trait),
            vec(__mirk_interp_values(y[i], trait)),
            vec(__mirk_interp_values(y[i + 1], trait)),
            p, mesh[i], mesh_dt[i], c_star, v_star, x_star, f, Val(iip), nothing, nothing
        )
    end
    return k_interp
end

# Extra interpolation stages on one interval, shared by nested and packed storage.
# The RHS adapter accounts for the state's shape and any singular contribution.
@inline function __mirk_interp_interval_values!(
        KI, tmp, K, y_left, y_right, p, t_left, h, c, v, x, f::F, iip, singular_term, metadata
    ) where {F}
    stage = size(K, 2)
    extra = size(KI, 2)
    nstates = size(K, 1)
    @inbounds for r in 1:extra
        for j in eachindex(tmp)
            value = zero(eltype(tmp))
            # Control components, when present, use linear interpolation.
            if j <= nstates
                for s in 1:stage
                    value += K[j, s] * x[(s - 1) * extra + r]
                end
                for s in 1:(r - 1)
                    value += KI[j, s] * x[(stage + s - 1) * extra + r]
                end
            end
            tmp[j] = (1 - v[r]) * y_left[j] + v[r] * y_right[j] + h * value
        end
        __mirk_rhs!(view(KI, :, r), f, tmp, p, t_left + c[r] * h, iip, singular_term, metadata)
    end
    return nothing
end

# Interpolant stages
@kernel function __mirk_device_interp_setup_kernel!(
        KI, tmp, K, y, f, p, mesh, mesh_dt, c, v, x, in_size, iip, singular_term
    )
    i = @index(Global, Linear)
    metadata = (; in_size, f_size = in_size, nparameters = 0, tune_parameters = Val(false))
    @inbounds __mirk_interp_interval_values!(
        view(KI, :, :, i), view(tmp, :, i), view(K, :, :, i),
        view(y, :, i), view(y, :, i + 1), p, mesh[i], mesh_dt[i], c, v, x,
        f, iip, singular_term, metadata
    )
end

function __mirk_device_interp_setup!(
        platform, KI, tmp, K, y, f, p, mesh, mesh_dt,
        itableau::MIRKInterpTableau, in_size, iip, singular_term
    )
    __mirk_device_interp_setup_kernel!(platform)(
        KI, tmp, K, y, f, p, mesh, mesh_dt, itableau.c_star, itableau.v_star,
        itableau.x_star, in_size, iip, singular_term; ndrange = length(mesh_dt)
    )
    synchronize(platform)
    return KI
end

"""
    update_eval_sol!(eval_sol::EvalSol, y_, cache::MIRKCache)

Update the intermediate solution `eval_sol` with the new flattened solution `y_` and the cache.
When evaluating boundary conditions with new solution during nonlinear solving, we should
always update the intermediate solution with discrete solution + discrete stages + new stages
(Continuous MIRK: u(meshᵢ + τ*dt) = yᵢ + dt sum br(τ)*kr).
"""
@views function update_eval_sol!(eval_sol::EvalSol, y_, cache::MIRKCache)
    restructured = __restructure_sol(y_, cache.in_size)
    eval_sol.cache.k_discrete[1:end] .= cache.k_discrete
    eval_sol.cache.k_interp.u[1:end] .= cache.k_interp.u
    interp_setup!(eval_sol.cache)
    # On an ordinary residual evaluation `y_` matches `eval_sol.u`'s element type, so the
    # preallocated buffer is updated in place (works for e.g. StaticArray states too).
    if promote_type(eltype(eltype(restructured)), eltype(eltype(eval_sol.u))) ===
            eltype(eltype(eval_sol.u))
        eval_sol.u[1:end] .= restructured
        return eval_sol
    end
    # When the residual is differentiated directly (e.g. a line-search directional
    # derivative), `y_` carries ForwardDiff.Duals that cannot be written into the Float64
    # buffer. The lazy cache hands back a matching-eltype buffer, allocated once per eltype
    # and reused afterwards. On this path `y_` is DiffCache-backed (plain Arrays), so the
    # VectorOfArray buffer is safe to allocate via `similar`.
    voa = VectorOfArray(restructured)
    u = get_tmp(cache.eval_sol_cache, voa)
    u .= voa
    return EvalSol(u.u, eval_sol.t, cache)
end

"""
Construct n root-finding problems and solve them to find the critical points with continuous derivative polynomials
"""
function __construct_then_solve_root_problem(sol::EvalSol{C}, tspan::Tuple) where {
        C <:
        MIRKCache,
    }
    n = first(size(sol))
    nlprobs = Vector{SciMLBase.NonlinearProblem}(undef, n)
    nlsols = Vector{SciMLBase.NonlinearSolution}(undef, length(nlprobs))
    nlsolve_alg = __FastShortcutNonlinearPolyalg(eltype(sol.cache))
    for i in 1:n
        f = @closure (t, p) -> sol(t, Val{1})[i]
        nlprob = NonlinearProblem(f, sol.cache.prob.u0[i], tspan)
        nlsols[i] = solve(nlprob, nlsolve_alg)
    end
    return nlsols
end

# It turns out the critical points can't cover all possible maximum/minimum values
# especially when the solution are monotonic, we still need to compare the extremes with
# value at critical points to find the maximum/minimum

"""
    maxsol(sol::EvalSol, tspan::Tuple)

Find the maximum of the solution over the time span `tspan`.
"""
function maxsol(sol::EvalSol{C}, tspan::Tuple) where {C <: MIRKCache}
    nlsols = __construct_then_solve_root_problem(sol, tspan)
    tvals = map(nlsol -> (SciMLBase.successful_retcode(nlsol); return nlsol.u), nlsols)
    u = sol(tvals)
    return max(maximum(sol), maximum(Iterators.flatten(u)))
end

"""
    minsol(sol::EvalSol, tspan::Tuple)

Find the minimum of the solution over the time span `tspan`.
"""
function minsol(sol::EvalSol{C}, tspan::Tuple) where {C <: MIRKCache}
    nlsols = __construct_then_solve_root_problem(sol, tspan)
    tvals = map(nlsol -> (SciMLBase.successful_retcode(nlsol); return nlsol.u), nlsols)
    u = sol(tvals)
    return min(minimum(sol), minimum(Iterators.flatten(u)))
end

"""
    interp_weights(τ, alg)

interp_weights: solver-specified interpolation weights and its first derivative
"""
function interp_weights end

# Tuple weights keep the identical polynomials available inside device kernels.
# The CPU interface still returns vectors for its matrix multiplication helpers.
for order in (2, 3, 4, 5, 6)
    alg = Symbol("MIRK$(order)")
    @eval begin
        function interp_weights(τ::T, ::$(alg)) where {T}
            w, wp = __mirk_interp_weights(τ, Val($(QuoteNode(alg))))
            return collect(w), collect(wp)
        end
        @inline function __mirk_interp_weights(τ::T, ::Val{$(QuoteNode(alg))}) where {T}
            if $(order == 2)
                w = (0, τ * (1 - τ / 2), τ^2 / 2)

                #     Derivative polynomials.

                wp = (0, 1 - τ, τ)
            elseif $(order == 3)
                w = (
                    τ / 4.0 * (2.0 * τ^2 - 5.0 * τ + 4.0),
                    -3.0 / 4.0 * τ^2 * (2.0 * τ - 3.0), τ^2 * (τ - 1.0),
                )

                #     Derivative polynomials.

                wp = (
                    3.0 / 2.0 * (τ - 2.0 / 3.0) * (τ - 1.0),
                    -9.0 / 2.0 * τ * (τ - 1.0), 3.0 * τ * (τ - 2.0 / 3.0),
                )
            elseif $(order == 4)
                t2 = τ * τ
                tm1 = τ - 1.0
                t4m3 = τ * 4.0 - 3.0
                t2m1 = τ * 2.0 - 1.0

                w = (
                    -τ * (2.0 * τ - 3.0) * (2.0 * t2 - 3.0 * τ + 2.0) / 6.0,
                    t2 * (12.0 * t2 - 20.0 * τ + 9.0) / 6.0,
                    2.0 * t2 * (6.0 * t2 - 14.0 * τ + 9.0) / 3.0,
                    -16.0 * t2 * tm1 * tm1 / 3.0,
                )

                #   Derivative polynomials

                wp = (
                    -tm1 * t4m3 * t2m1 / 3.0, τ * t2m1 * t4m3,
                    4.0 * τ * t4m3 * tm1, -32.0 * τ * t2m1 * tm1 / 3.0,
                )
            elseif $(order == 5)
                w = (
                    τ * (
                        22464.0 - 83910.0 * τ + 143041.0 * τ^2 - 113808.0 * τ^3 +
                            33256.0 * τ^4
                    ) / 22464.0,
                    τ^2 * (-2418.0 + 12303.0 * τ - 19512.0 * τ^2 + 10904.0 * τ^3) / 3360.0,
                    -8 / 81 * τ^2 * (-78.0 + 209.0 * τ - 204.0 * τ^2 + 8.0 * τ^3),
                    -25 / 1134 * τ^2 * (-390.0 + 1045.0 * τ - 1020.0 * τ^2 + 328.0 * τ^3),
                    -25 / 5184 * τ^2 * (390.0 + 255.0 * τ - 1680.0 * τ^2 + 2072.0 * τ^3),
                    279841 / 168480 * τ^2 * (-6.0 + 21.0 * τ - 24.0 * τ^2 + 8.0 * τ^3),
                )

                #   Derivative polynomials

                wp = (
                    1.0 - 13985 // 1872 * τ + 143041 // 7488 * τ^2 - 2371 // 117 * τ^3 +
                        20785 // 2808 * τ^4,
                    -403 // 280 * τ + 12303 // 1120 * τ^2 - 813 // 35 * τ^3 +
                        1363 // 84 * τ^4,
                    416 // 27 * τ - 1672 // 27 * τ^2 + 2176 // 27 * τ^3 - 320 // 81 * τ^4,
                    3250 // 189 * τ - 26125 // 378 * τ^2 + 17000 // 189 * τ^3 -
                        20500 // 567 * τ^4,
                    -1625 // 432 * τ - 2125 // 576 * τ^2 + 875 // 27 * τ^3 -
                        32375 // 648 * τ^4,
                    -279841 // 14040 * τ + 1958887 // 18720 * τ^2 - 279841 // 1755 * τ^3 +
                        279841 // 4212 * τ^4,
                )
            elseif $(order == 6)
                w = (
                    τ - 28607 // 7434 * τ^2 - 166210 // 33453 * τ^3 +
                        334780 // 11151 * τ^4 - 1911296 // 55755 * τ^5 + 406528 // 33453 * τ^6,
                    777 // 590 * τ^2 - 2534158 // 234171 * τ^3 + 2088580 // 78057 * τ^4 -
                        10479104 // 390285 * τ^5 + 11328512 // 1170855 * τ^6,
                    -1008 // 59 * τ^2 + 222176 // 1593 * τ^3 - 180032 // 531 * τ^4 +
                        876544 // 2655 * τ^5 - 180224 // 1593 * τ^6,
                    -1008 // 59 * τ^2 + 222176 // 1593 * τ^3 - 180032 // 531 * τ^4 +
                        876544 // 2655 * τ^5 - 180224 // 1593 * τ^6,
                    -378 // 59 * τ^2 + 27772 // 531 * τ^3 - 22504 // 177 * τ^4 +
                        109568 // 885 * τ^5 - 22528 // 531 * τ^6,
                    -95232 // 413 * τ^2 + 62384128 // 33453 * τ^3 -
                        49429504 // 11151 * τ^4 + 46759936 // 11151 * τ^5 -
                        46661632 // 33453 * τ^6,
                    896 // 5 * τ^2 - 4352 // 3 * τ^3 + 3456 * τ^4 - 16384 // 5 * τ^5 +
                        16384 // 15 * τ^6,
                    50176 // 531 * τ^2 - 179554304 // 234171 * τ^3 +
                        143363072 // 78057 * τ^4 - 136675328 // 78057 * τ^5 +
                        137363456 // 234171 * τ^6,
                    16384 // 441 * τ^3 - 16384 // 147 * τ^4 + 16384 // 147 * τ^5 -
                        16384 // 441 * τ^6,
                )

                #     Derivative polynomials.

                wp = (
                    1 - 28607 // 3717 * τ - 166210 // 11151 * τ^2 + 1339120 // 11151 * τ^3 -
                        1911296 // 11151 * τ^4 + 813056 // 11151 * τ^5,
                    777 // 295 * τ - 2534158 // 78057 * τ^2 + 8354320 // 78057 * τ^3 -
                        10479104 // 78057 * τ^4 + 22657024 // 390285 * τ^5,
                    -2016 // 59 * τ + 222176 // 531 * τ^2 - 720128 // 531 * τ^3 +
                        876544 // 531 * τ^4 - 360448 // 531 * τ^5,
                    -2016 // 59 * τ + 222176 // 531 * τ^2 - 720128 // 531 * τ^3 +
                        876544 // 531 * τ^4 - 360448 // 531 * τ^5,
                    -756 // 59 * τ + 27772 // 177 * τ^2 - 90016 // 177 * τ^3 +
                        109568 // 177 * τ^4 - 45056 // 177 * τ^5,
                    -190464 // 413 * τ + 62384128 // 11151 * τ^2 -
                        197718016 // 11151 * τ^3 + 233799680 // 11151 * τ^4 -
                        93323264 // 11151 * τ^5,
                    1792 // 5 * τ - 4352 * τ^2 + 13824 * τ^3 - 16384 * τ^4 +
                        32768 // 5 * τ^5,
                    100352 // 531 * τ - 179554304 // 78057 * τ^2 +
                        573452288 // 78057 * τ^3 - 683376640 // 78057 * τ^4 +
                        274726912 // 78057 * τ^5,
                    16384 // 147 * τ^2 - 65536 // 147 * τ^3 + 81920 // 147 * τ^4 -
                        32768 // 147 * τ^5,
                )
            end
            return T.(w), T.(wp)
        end
    end
end

for order in (6,)
    alg = Symbol("MIRK$(order)I")
    @eval begin
        function interp_weights(τ::T, ::$(alg)) where {T}
            w, wp = __mirk_interp_weights(τ, Val($(QuoteNode(alg))))
            return collect(w), collect(wp)
        end
        @inline function __mirk_interp_weights(τ::T, ::Val{$(QuoteNode(alg))}) where {T}
            if $(order == 6)
                w = (
                    -(12233 + 1450 * sqrt(7)) *
                        (
                        800086000 * τ^5 + 63579600 * sqrt(7) * τ^4 - 2936650584 * τ^4 +
                            4235152620 * τ^3 - 201404565 * sqrt(7) * τ^3 +
                            232506630 * sqrt(7) * τ^2 - 3033109390 * τ^2 + 1116511695 * τ -
                            116253315 * sqrt(7) * τ + 22707000 * sqrt(7) - 191568780
                    ) *
                        τ / 2112984835740,
                    -(-10799 + 650 * sqrt(7)) *
                        (
                        24962000 * τ^4 + 473200 * sqrt(7) * τ^3 - 67024328 * τ^3 -
                            751855 * sqrt(7) * τ^2 + 66629600 * τ^2 - 29507250 * τ +
                            236210 * sqrt(7) * τ +
                            5080365 +
                            50895 * sqrt(7)
                    ) *
                        τ^2 / 29551834260,
                    7 / 1274940 *
                        (259 + 50 * sqrt(7)) *
                        (
                        14000 * τ^4 - 48216 * τ^3 + 1200 * sqrt(7) * τ^3 -
                            3555 * sqrt(7) * τ^2 +
                            62790 * τ^2 +
                            3610 * sqrt(7) * τ - 37450 * τ + 9135 - 1305 * sqrt(7)
                    ) *
                        τ^2,
                    7 / 1274940 *
                        (259 + 50 * sqrt(7)) *
                        (
                        14000 * τ^4 - 48216 * τ^3 + 1200 * sqrt(7) * τ^3 -
                            3555 * sqrt(7) * τ^2 +
                            62790 * τ^2 +
                            3610 * sqrt(7) * τ - 37450 * τ + 9135 - 1305 * sqrt(7)
                    ) *
                        τ^2,
                    16 / 2231145 *
                        (259 + 50 * sqrt(7)) *
                        (
                        14000 * τ^4 - 48216 * τ^3 + 1200 * sqrt(7) * τ^3 -
                            3555 * sqrt(7) * τ^2 +
                            62790 * τ^2 +
                            3610 * sqrt(7) * τ - 37450 * τ + 9135 - 1305 * sqrt(7)
                    ) *
                        τ^2,
                    4 / 1227278493 *
                        (740 * sqrt(7) - 6083) *
                        (
                        1561000 * τ^2 - 2461284 * τ - 109520 * sqrt(7) * τ +
                            979272 +
                            86913 * sqrt(7)
                    ) *
                        (τ - 1)^2 *
                        τ^2,
                    -49 / 63747 *
                        sqrt(7) *
                        (20000 * τ^2 - 20000 * τ + 3393) *
                        (τ - 1)^2 *
                        τ^2,
                    -1250000000 / 889206903 * (28 * τ^2 - 28 * τ + 9) * (τ - 1)^2 * τ^2,
                )

                #     Derivative polynomials.

                wp = (
                    (1450 * sqrt(7) + 12233) *
                        (14 * τ - 7 + sqrt(7)) *
                        (τ - 1) *
                        (-400043 * τ + 75481 + 2083 * sqrt(7)) *
                        (100 * τ - 87) *
                        (2 * τ - 1) / 493029795006,
                    -(650 * sqrt(7) - 10799) *
                        (14 * τ - 7 + sqrt(7)) *
                        (37443 * τ - 13762 - 2083 * sqrt(7)) *
                        (100 * τ - 87) *
                        (2 * τ - 1) *
                        τ / 20686283982,
                    7 / 42498 *
                        (259 + 50 * sqrt(7)) *
                        (14 * τ - 7 + sqrt(7)) *
                        (τ - 1) *
                        (100 * τ - 87) *
                        (2 * τ - 1) *
                        τ,
                    7 / 42498 *
                        (259 + 50 * sqrt(7)) *
                        (14 * τ - 7 + sqrt(7)) *
                        (τ - 1) *
                        (100 * τ - 87) *
                        (2 * τ - 1) *
                        τ,
                    32 / 148743 *
                        (259 + 50 * sqrt(7)) *
                        (14 * τ - 7 + sqrt(7)) *
                        (τ - 1) *
                        (100 * τ - 87) *
                        (2 * τ - 1) *
                        τ,
                    4 / 1227278493 *
                        (740 * sqrt(7) - 6083) *
                        (14 * τ - 7 + sqrt(7)) *
                        (τ - 1) *
                        (100 * τ - 87) *
                        (6690 * τ - 4085 - 869 * sqrt(7)) *
                        τ,
                    -98 / 21249 *
                        sqrt(7) *
                        (τ - 1) *
                        (100 * τ - 13) *
                        (100 * τ - 87) *
                        (2 * τ - 1) *
                        τ,
                    -1250000000 / 2074816107 *
                        (14 * τ - 7 + sqrt(7)) *
                        (τ - 1) *
                        (14 * τ - 7 - sqrt(7)) *
                        (2 * τ - 1) *
                        τ,
                )
            end
            return T.(w), T.(wp)
        end
    end
end

# Packed storage reuses the CPU interpolation and intermediate solution wrappers.
# The cache is a concrete NamedTuple that can be constructed from adapted arrays
# inside a kernel; no host solver cache enters device code.
@inline function __build_interpolation(
        y::AbstractMatrix, k, ki, mesh, mesh_dt, algid, in_size, platform = nothing
    )
    return MIRKInterpolation(mesh, y, (; k, ki, mesh_dt, algid, in_size, platform))
end

struct MIRKMeshValues{I, D}
    interp::I
    deriv::D
end

Base.length(u::MIRKMeshValues) = size(u.interp.u, 2)
Base.size(u::MIRKMeshValues) = (length(u),)
Base.firstindex(::MIRKMeshValues) = 1
Base.lastindex(u::MIRKMeshValues) = length(u)
Base.eachindex(u::MIRKMeshValues) = Base.OneTo(length(u))
Base.@propagate_inbounds Base.getindex(u::MIRKMeshValues{I, Val{0}}, i::Int) where {I} =
    __device_reshape(view(u.interp.u, :, i), u.interp.cache.in_size)
Base.@propagate_inbounds Base.getindex(u::MIRKMeshValues{I, Val{1}}, i::Int) where {I} =
    interpolation(u.interp.t[i], u.interp, Val(1))
Base.first(u::MIRKMeshValues) = u[1]
Base.last(u::MIRKMeshValues) = u[length(u)]
@inline Base.iterate(u::MIRKMeshValues, i::Int = 1) =
    i > length(u) ? nothing : (u[i], i + 1)

BoundaryValueDiffEqCore.EvalSol(id::MIRKInterpolation{<:NamedTuple}) =
    EvalSol(MIRKMeshValues(id, Val(0)), id.t, id)

@inline function Base.getproperty(sol::EvalSol{<:MIRKInterpolation}, name::Symbol)
    name === :du && return MIRKMeshValues(getfield(sol, :cache), Val(1))
    return getfield(sol, name)
end

# Core's generic indexing materializes a VectorOfArray, so packed interpolation
# dispatches directly to views of its matrix instead.
Base.size(sol::EvalSol{<:MIRKInterpolation}) =
    (sol.cache.cache.in_size..., size(sol.cache.u, 2))
Base.firstindex(::EvalSol{<:MIRKInterpolation}, d::Int) = 1
Base.lastindex(sol::EvalSol{<:MIRKInterpolation}, d::Int) = size(sol, d)
Base.@propagate_inbounds Base.getindex(sol::EvalSol{<:MIRKInterpolation}, i::Int) = sol.u[i]
Base.@propagate_inbounds Base.getindex(sol::EvalSol{<:MIRKInterpolation}, ::Colon, i::Int) =
    view(sol.cache.u, :, i)
Base.@propagate_inbounds Base.getindex(sol::EvalSol{<:MIRKInterpolation}, i::Int, j::Int) =
    sol.cache.u[i, j]
Base.@propagate_inbounds function Base.getindex(
        sol::EvalSol{<:MIRKInterpolation}, indices::Vararg{Int, N}
    ) where {N}
    in_size = sol.cache.cache.in_size
    @boundscheck N == length(in_size) + 1 || throw(BoundsError(sol, indices))
    row = indices[1]
    stride = in_size[1]
    for d in 2:(N - 1)
        row += (indices[d] - 1) * stride
        stride *= in_size[d]
    end
    return sol.cache.u[row, indices[N]]
end
Base.@propagate_inbounds Base.getindex(sol::EvalSol{<:MIRKInterpolation}, ::Colon, ::Colon, node::Int) =
    sol.u[node]
@inline Base.iterate(sol::EvalSol{<:MIRKInterpolation}, i::Int = 1) =
    i > length(sol) ? nothing : (sol[i], i + 1)

# Match the CPU interpolant's left continuity for either mesh direction.
@inline function __mirk_device_interval(mesh, t)
    lo = 1
    hi = length(mesh)
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

struct MIRKDeviceInterpolatedArray{T, N, S, W, D} <: AbstractArray{T, N}
    sol::S
    weights::W
    interval::Int
    endpoint::Int
    deriv::D
end

Base.size(u::MIRKDeviceInterpolatedArray) = u.sol.cache.in_size
Base.IndexStyle(::Type{<:MIRKDeviceInterpolatedArray}) = IndexLinear()

@inline function interpolation(
        t::Number, sol::MIRKInterpolation{<:NamedTuple}, deriv::Union{Val{0}, Val{1}}
    )
    (; u, cache) = sol
    (; k, ki, mesh_dt, algid, in_size) = cache
    D = deriv isa Val{0} ? 0 : 1
    i = __mirk_device_interval(sol.t, t)
    @inbounds τ = (t - sol.t[i]) / mesh_dt[i]
    w, wp = __mirk_interp_weights(τ, algid)
    weights = D == 0 ? w : wp
    endpoint = 0
    if D == 0
        @inbounds endpoint = t == sol.t[1] ? 1 :
            (t == sol.t[end] ? length(sol.t) : 0)
    end
    return MIRKDeviceInterpolatedArray{
        eltype(u), length(in_size), typeof(sol), typeof(weights), typeof(deriv),
    }(sol, weights, i, endpoint, deriv)
end

@inline (sol::EvalSol{<:MIRKInterpolation})(
    t::Number, deriv::Union{Val{0}, Val{1}} = Val(0)
) = interpolation(t, sol.cache, deriv)
@inline (sol::EvalSol{<:MIRKInterpolation})(t::Number, ::Type{Val{D}}) where {D} =
    sol(t, Val(D))

Base.@propagate_inbounds function Base.getindex(u::MIRKDeviceInterpolatedArray, j::Int)
    sol = u.sol
    (; k, ki, mesh_dt) = sol.cache
    i = u.interval
    if u.endpoint != 0
        return sol.u[j, u.endpoint]
    end
    stage = size(k, 2)
    value = zero(eltype(u))
    for r in 1:stage
        value += k[j, r, i] * u.weights[r]
    end
    for r in 1:size(ki, 2)
        value += ki[j, r, i] * u.weights[stage + r]
    end
    result = u.deriv isa Val{0} ? sol.u[j, i] + mesh_dt[i] * value : value
    return convert(eltype(u), result)
end

SciMLBase.interp_summary(id::MIRKInterpolation{<:NamedTuple}) =
    "$(typeof(id.cache.algid).parameters[1]) device interpolation"

@inline __mirk_device_interpolation_index(::Nothing, j) = j
@inline __mirk_device_interpolation_index(idxs::Integer, j) = idxs
@inline __mirk_device_interpolation_index(idxs, j) = @inbounds idxs[j]

@kernel function __mirk_device_interpolate_kernel!(
        out, y, K, KI, mesh, mesh_dt, algid, in_size, t, deriv, idxs
    )
    j = @index(Global, Linear)
    @inbounds begin
        id = __build_interpolation(y, K, KI, mesh, mesh_dt, algid, in_size)
        row = __mirk_device_interpolation_index(idxs, j)
        out[j] = interpolation(t, id, deriv)[row]
    end
end

function interpolation!(
        out, t::Number, id::MIRKInterpolation{<:NamedTuple}, idxs,
        ::Type{Val{D}}, p, continuity::Symbol = :left
    ) where {D}
    return interpolation!(out, t, id, idxs, Val(D), p, continuity)
end

function interpolation!(
        out, t::Number, id::MIRKInterpolation{<:NamedTuple}, idxs,
        deriv::Val{D}, p, continuity::Symbol = :left
    ) where {D}
    D in (0, 1) || throw(ArgumentError("MIRK interpolation supports derivatives of order zero or one."))
    (; u, cache) = id
    (; k, ki, mesh_dt, algid, in_size, platform) = cache
    nstates = size(u, 1)
    if idxs isa Integer
        1 <= idxs <= nstates || throw(BoundsError(Base.OneTo(nstates), idxs))
    elseif idxs !== nothing
        idxs isa Union{AbstractArray, Tuple} || throw(
            ArgumentError("MIRK interpolation indices must be integers or an integer collection.")
        )
        all(i -> i isa Integer && 1 <= i <= nstates, idxs) ||
            throw(BoundsError(Base.OneTo(nstates), idxs))
    end
    expected_length = idxs === nothing ? nstates : (idxs isa Integer ? 1 : length(idxs))
    length(out) == expected_length || throw(
        DimensionMismatch("MIRK interpolation output length must be $expected_length.")
    )
    typeof(__device_initial_backend(out)) === typeof(platform) || throw(
        ArgumentError("MIRK interpolation output must use the solution backend.")
    )
    device_idxs = idxs isa AbstractArray ? __device_parameter(platform, idxs) : idxs
    __mirk_device_interpolate_kernel!(platform)(
        out, u, k, ki, id.t, mesh_dt, algid,
        in_size, t, deriv, device_idxs; ndrange = length(out)
    )
    synchronize(platform)
    return nothing
end

function interpolation(
        t::Number, id::MIRKInterpolation{<:NamedTuple}, idxs,
        deriv::Union{Type{<:Val}, Val}, p, continuity::Symbol = :left
    )
    dims = idxs === nothing ? id.cache.in_size : (idxs isa Integer ? (1,) : (length(idxs),))
    out = similar(id.u, eltype(id.u), dims)
    interpolation!(out, t, id, idxs, deriv, p, continuity)
    return idxs isa Integer ? sum(out) : out
end

function interpolation(
        tvals, id::MIRKInterpolation{<:NamedTuple}, idxs,
        deriv::Union{Type{<:Val}, Val}, p, continuity::Symbol = :left
    )
    # Keep time metadata on the host.
    times = collect(tvals)
    values = [id(t, idxs, deriv, p, continuity) for t in times]
    return DiffEqArray(values, times)
end

function interpolation!(
        out, tvals, id::MIRKInterpolation{<:NamedTuple}, idxs,
        deriv::Union{Type{<:Val}, Val}, p, continuity::Symbol = :left
    )
    for (i, t) in enumerate(tvals)
        interpolation!(out[i], t, id, idxs, deriv, p, continuity)
    end
    return nothing
end
