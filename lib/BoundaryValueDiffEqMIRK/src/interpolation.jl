# MIRK Interpolation
@concrete struct MIRKInterpolation <: AbstractDiffEqInterpolation
    t
    u
    cache
end

function SciMLBase.interp_summary(interp::MIRKInterpolation)
    return "MIRK Order $(interp.cache.order) Interpolation"
end

__has_control_variables(cache::MIRKCache, length_z) = !isnothing(cache.f_prototype) &&
    length(cache.f_prototype) < length_z
__state_variable_count(cache::MIRKCache, length_z) = __has_control_variables(cache, length_z) ?
    length(cache.f_prototype) : length_z

function (id::MIRKInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::MIRKInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
    return
end

@inline function interpolation(
        tvals, id::MIRKInterpolation, idxs, deriv,
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
        vals, tvals, id::MIRKInterpolation, idxs,
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
        tval::Number, id::MIRKInterpolation, idxs,
        deriv, p, continuity::Symbol = :left
    )
    z = similar(id.u[1])
    interpolant!(z, id, id.cache, tval, id.cache.mesh, id.cache.mesh_dt, deriv)
    return idxs !== nothing ? z[idxs] : z
end

@inline function interpolant!(
        z::AbstractArray, id::MIRKInterpolation, cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{0}}
    )
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    w, _ = interp_weights(τ, cache.alg)
    return sum_stages!(z, id, cache, w, i, τ, T)
end

@inline function interpolant!(
        dz::AbstractArray, id::MIRKInterpolation,
        cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{1}}
    )
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    _, w′ = interp_weights(τ, cache.alg)
    return sum_stages!(dz, id, cache, w′, i, τ, T)
end

@views function sum_stages!(
        z::AbstractArray, id::MIRKInterpolation,
        cache::MIRKCache{iip, T, use_both, DiffCacheNeeded},
        w, i::Int, τ, ::Type{Val{0}}
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp, M) = cache
    (; s_star) = cache.ITU
    dt = cache.mesh_dt[i]

    has_control = __has_control_variables(cache, length(z))

    # state variables have their interpolation polynomials
    length_z = __state_variable_count(cache, length(z))
    z .= zero(z)
    __maybe_matmul!(z[1:length_z], k_discrete[i].du[1:length_z, 1:stage], w[1:stage])
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
        z::AbstractArray, id::MIRKInterpolation,
        cache::MIRKCache{iip, T, use_both, NoDiffCacheNeeded},
        w, i::Int, τ, ::Type{Val{0}}
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp, M) = cache
    (; s_star) = cache.ITU
    dt = cache.mesh_dt[i]

    has_control = __has_control_variables(cache, length(z))
    length_z = __state_variable_count(cache, length(z))

    z .= zero(z)
    __maybe_matmul!(z[1:length_z], k_discrete[i][1:length_z, 1:stage], w[1:stage])
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
        z′, id::MIRKInterpolation, cache::MIRKCache{iip, T, use_both, DiffCacheNeeded},
        w′, i::Int, τ, ::Type{Val{1}}
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp, M) = cache
    (; s_star) = cache.ITU
    has_control = __has_control_variables(cache, length(z′))
    length_z = __state_variable_count(cache, length(z′))

    z′ .= zero(z′)
    __maybe_matmul!(z′[1:length_z], k_discrete[i].du[1:length_z, 1:stage], w′[1:stage])
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
@views function sum_stages!(
        z′, id::MIRKInterpolation, cache::MIRKCache{iip, T, use_both, NoDiffCacheNeeded},
        w′, i::Int, τ, ::Type{Val{1}}
    ) where {iip, T, use_both}
    (; stage, k_discrete, k_interp, M) = cache
    (; s_star) = cache.ITU
    has_control = __has_control_variables(cache, length(z′))
    length_z = __state_variable_count(cache, length(z′))

    z′ .= zero(z′)
    __maybe_matmul!(z′[1:length_z], k_discrete[i][1:length_z, 1:stage], w′[1:stage])
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
    ii = interval(s.t, tval)
    return __interp_derivative(s, ii, (tval - s.t[ii]) / s.cache.mesh_dt[ii])
end

# Derivative of the continuous MIRK solution on mesh interval `ii` at local coordinate `τ`.
function __interp_derivative(s::EvalSol{C}, ii::Int, τ) where {C <: MIRKCache}
    (; alg, stage, k_discrete, k_interp) = s.cache
    z′ = zeros(typeof(τ), s.cache.M)
    _, w′ = interp_weights(τ, alg)
    K = __needs_diffcache(alg.jac_alg) ? @view(k_discrete[ii].du[:, 1:stage]) :
        @view(k_discrete[ii][:, 1:stage])
    __maybe_matmul!(z′, K, @view(w′[1:stage]))
    __maybe_matmul!(
        z′, @view(k_interp.u[ii][:, 1:(s.cache.ITU.s_star - stage)]),
        @view(w′[(stage + 1):s.cache.ITU.s_star]), true, true
    )
    return z′
end

"""
    interp_setup!(cache::MIRKCache)

`interp_setup!` prepare the extra stages in `ki_interp`` for interpolant construction.
Here, the `ki_interp`` is the stages in one subinterval.
"""
@views function interp_setup!(
        cache::MIRKCache{
            iip, T, use_both, DiffCacheNeeded,
        }
    ) where {iip, T, use_both}
    (; x_star, s_star, c_star, v_star) = cache.ITU
    (; k_interp, k_discrete, f, stage, new_stages, y, p, mesh, mesh_dt) = cache
    for r in 1:(s_star - stage)
        idx₁ = ((1:stage) .- 1) .* (s_star - stage) .+ r
        idx₂ = ((1:(r - 1)) .+ stage .- 1) .* (s_star - stage) .+ r
        for j in eachindex(k_discrete)
            __maybe_matmul!(new_stages.u[j], k_discrete[j].du[:, 1:stage], x_star[idx₁])
        end
        if r > 1
            for j in eachindex(k_interp.u)
                __maybe_matmul!(
                    new_stages.u[j], k_interp.u[j][:, 1:(r - 1)], x_star[idx₂], T(1), T(1)
                )
            end
        end
        for i in eachindex(new_stages.u)
            new_stages.u[i] .= new_stages.u[i] .* mesh_dt[i] .+
                (1 - v_star[r]) .* vec(y[i].du) .+
                v_star[r] .* vec(y[i + 1].du)
            if iip
                f(k_interp.u[i][:, r], new_stages.u[i], p, mesh[i] + c_star[r] * mesh_dt[i])
            else
                k_interp.u[i][:, r] .= f(
                    new_stages.u[i], p, mesh[i] +
                        c_star[r] * mesh_dt[i]
                )
            end
        end
    end

    return k_interp
end
@views function interp_setup!(
        cache::MIRKCache{
            iip, T, use_both, NoDiffCacheNeeded,
        }
    ) where {iip, T, use_both}
    (; x_star, s_star, c_star, v_star) = cache.ITU
    (; k_interp, k_discrete, f, stage, new_stages, y, p, mesh, mesh_dt) = cache
    for r in 1:(s_star - stage)
        idx₁ = ((1:stage) .- 1) .* (s_star - stage) .+ r
        idx₂ = ((1:(r - 1)) .+ stage .- 1) .* (s_star - stage) .+ r
        for j in eachindex(k_discrete)
            __maybe_matmul!(new_stages.u[j], k_discrete[j][:, 1:stage], x_star[idx₁])
        end
        if r > 1
            for j in eachindex(k_interp.u)
                __maybe_matmul!(
                    new_stages.u[j], k_interp.u[j][:, 1:(r - 1)], x_star[idx₂], T(1), T(1)
                )
            end
        end
        for i in eachindex(new_stages.u)
            new_stages.u[i] .= new_stages.u[i] .* mesh_dt[i] .+
                (1 - v_star[r]) .* vec(y[i]) .+ v_star[r] .* vec(y[i + 1])
            if iip
                f(k_interp.u[i][:, r], new_stages.u[i], p, mesh[i] + c_star[r] * mesh_dt[i])
            else
                k_interp.u[i][:, r] .= f(
                    new_stages.u[i], p, mesh[i] +
                        c_star[r] * mesh_dt[i]
                )
            end
        end
    end

    return k_interp
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

# An extremum of a component over `tspan` is attained at an end of `tspan`, at a mesh point, or
# at a zero of the component's derivative. Those zeros are bracketed by a sign change of the
# derivative over a mesh interval and located by bisection. The candidates are sorted in time.
function __extremum_candidates(sol::EvalSol{C}, tspan::Tuple) where {C <: MIRKCache}
    (; t) = sol
    mesh_dt = sol.cache.mesh_dt
    lo, hi = minmax(tspan...)
    tvals = [lo, hi]
    for ii in 1:(length(t) - 1)
        lo < t[ii] < hi && push!(tvals, t[ii])
        a, b = max(t[ii], lo), min(t[ii + 1], hi)
        a < b || continue
        τa, τb = (a - t[ii]) / mesh_dt[ii], (b - t[ii]) / mesh_dt[ii]
        da, db = __interp_derivative(sol, ii, τa), __interp_derivative(sol, ii, τb)
        for i in eachindex(da, db)
            (iszero(da[i]) || iszero(db[i]) || signbit(da[i]) == signbit(db[i])) && continue
            τ = __bisect(τ -> __interp_derivative(sol, ii, τ)[i], τa, τb, da[i])
            push!(tvals, t[ii] + τ * mesh_dt[ii])
        end
    end
    return sort!(tvals)
end

function __bisect(f, a, b, fa)
    m = (a + b) / 2
    while a < m < b
        fm = f(m)
        iszero(fm) && return m
        if signbit(fm) == signbit(fa)
            a, fa = m, fm
        else
            b = m
        end
        m = (a + b) / 2
    end
    return m
end

# Ties go to the earliest candidate (and lowest component) for both the maximum and the
# minimum, so that the boundary-condition Jacobian does not depend on an asymmetric
# tie-breaking rule. With Base's `max`/`min`, a flat initial guess pairs the last point for the
# maximum with the first point for the minimum, which can make the Newton system singular.
function __extremum(isbetter::F, sol::EvalSol{C}, tspan::Tuple) where {F, C <: MIRKCache}
    best = first(sol(minimum(tspan)))
    for t in __extremum_candidates(sol, tspan)
        k = searchsortedfirst(sol.t, t)
        u = k ≤ length(sol.t) && sol.t[k] == t ? sol.u[k] : sol(t)
        for x in u
            isbetter(x, best) && (best = x)
        end
    end
    return best
end

"""
    maxsol(sol::EvalSol, tspan::Tuple)

Find the maximum over all components of the solution over the time span `tspan`.
"""
maxsol(sol::EvalSol{C}, tspan::Tuple) where {C <: MIRKCache} = __extremum(>, sol, tspan)

"""
    minsol(sol::EvalSol, tspan::Tuple)

Find the minimum over all components of the solution over the time span `tspan`.
"""
minsol(sol::EvalSol{C}, tspan::Tuple) where {C <: MIRKCache} = __extremum(<, sol, tspan)

"""
    interp_weights(τ, alg)

interp_weights: solver-specified interpolation weights and its first derivative
"""
function interp_weights end

for order in (2, 3, 4, 5, 6)
    alg = Symbol("MIRK$(order)")
    @eval begin
        function interp_weights(τ::T, ::$(alg)) where {T}
            if $(order == 2)
                w = [0, τ * (1 - τ / 2), τ^2 / 2]

                #     Derivative polynomials.

                wp = [0, 1 - τ, τ]
            elseif $(order == 3)
                w = [
                    τ / 4.0 * (2.0 * τ^2 - 5.0 * τ + 4.0),
                    -3.0 / 4.0 * τ^2 * (2.0 * τ - 3.0), τ^2 * (τ - 1.0),
                ]

                #     Derivative polynomials.

                wp = [
                    3.0 / 2.0 * (τ - 2.0 / 3.0) * (τ - 1.0),
                    -9.0 / 2.0 * τ * (τ - 1.0), 3.0 * τ * (τ - 2.0 / 3.0),
                ]
            elseif $(order == 4)
                t2 = τ * τ
                tm1 = τ - 1.0
                t4m3 = τ * 4.0 - 3.0
                t2m1 = τ * 2.0 - 1.0

                w = [
                    -τ * (2.0 * τ - 3.0) * (2.0 * t2 - 3.0 * τ + 2.0) / 6.0,
                    t2 * (12.0 * t2 - 20.0 * τ + 9.0) / 6.0,
                    2.0 * t2 * (6.0 * t2 - 14.0 * τ + 9.0) / 3.0,
                    -16.0 * t2 * tm1 * tm1 / 3.0,
                ]

                #   Derivative polynomials

                wp = [
                    -tm1 * t4m3 * t2m1 / 3.0, τ * t2m1 * t4m3,
                    4.0 * τ * t4m3 * tm1, -32.0 * τ * t2m1 * tm1 / 3.0,
                ]
            elseif $(order == 5)
                w = [
                    τ * (
                        22464.0 - 83910.0 * τ + 143041.0 * τ^2 - 113808.0 * τ^3 +
                            33256.0 * τ^4
                    ) / 22464.0,
                    τ^2 * (-2418.0 + 12303.0 * τ - 19512.0 * τ^2 + 10904.0 * τ^3) / 3360.0,
                    -8 / 81 * τ^2 * (-78.0 + 209.0 * τ - 204.0 * τ^2 + 8.0 * τ^3),
                    -25 / 1134 * τ^2 * (-390.0 + 1045.0 * τ - 1020.0 * τ^2 + 328.0 * τ^3),
                    -25 / 5184 * τ^2 * (390.0 + 255.0 * τ - 1680.0 * τ^2 + 2072.0 * τ^3),
                    279841 / 168480 * τ^2 * (-6.0 + 21.0 * τ - 24.0 * τ^2 + 8.0 * τ^3),
                ]

                #   Derivative polynomials

                wp = [
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
                ]
            elseif $(order == 6)
                w = [
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
                ]

                #     Derivative polynomials.

                wp = [
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
                ]
            end
            return T.(w), T.(wp)
        end
    end
end

for order in (6,)
    alg = Symbol("MIRK$(order)I")
    @eval begin
        function interp_weights(τ::T, ::$(alg)) where {T}
            if $(order == 6)
                w = [
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
                ]

                #     Derivative polynomials.

                wp = [
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
                ]
            end
            return T.(w), T.(wp)
        end
    end
end
