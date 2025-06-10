# MIRK Interpolation
@concrete struct MIRKInterpolation <: AbstractDiffEqInterpolation
    t
    u
    cache
end

function DiffEqBase.interp_summary(interp::MIRKInterpolation)
    return "MIRK Order $(interp.cache.order) Interpolation"
end

function (id::MIRKInterpolation)(tvals, idxs, deriv, p, continuity::Symbol = :left)
    return interpolation(tvals, id, idxs, deriv, p, continuity)
end

function (id::MIRKInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol = :left)
    interpolation!(val, tvals, id, idxs, deriv, p, continuity)
    return
end

@inline function interpolation(tvals, id::MIRKInterpolation, idxs, deriv::D,
        p, continuity::Symbol = :left) where {D}
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

@inline function interpolation!(vals, tvals, id::MIRKInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
    (; t, cache) = id
    (; mesh, mesh_dt) = cache
    tdir = sign(t[end] - t[1])
    idx = sortperm(tvals, rev = tdir < 0)

    for j in idx
        z = similar(id.u[1])
        interpolant!(z, id, cache, tvals[j], mesh, mesh_dt, deriv)
        vals[j] = z
    end
end

@inline function interpolation(tval::Number, id::MIRKInterpolation, idxs,
        deriv::D, p, continuity::Symbol = :left) where {D}
    z = similar(id.u[1])
    interpolant!(z, id, id.cache, tval, id.cache.mesh, id.cache.mesh_dt, deriv)
    return idxs !== nothing ? z[idxs] : z
end

@inline function interpolant!(
        z::AbstractArray, id, cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{0}})
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    w, _ = interp_weights(τ, cache.alg)
    sum_stages!(z, id, cache, w, i, T)
end

@inline function interpolant!(dz::AbstractArray, id::MIRKInterpolation,
        cache::MIRKCache, t, mesh, mesh_dt, T::Type{Val{1}})
    i = interval(mesh, t)
    dt = mesh_dt[i]
    τ = (t - mesh[i]) / dt
    _, w′ = interp_weights(τ, cache.alg)
    sum_stages!(dz, id, cache, w′, i, T)
end

@views function sum_stages!(z::AbstractArray, id::MIRKInterpolation,
        cache::MIRKCache{iip, T, use_both, DiffCacheNeeded, fit_parameters},
        w, i::Int, ::Type{Val{0}}) where {iip, T, use_both, fit_parameters}
    (; stage, k_discrete, k_interp) = cache
    (; s_star) = cache.ITU
    dt = cache.mesh_dt[i]
    length_z = length(z)
    z .= zero(z)
    __maybe_matmul!(z, k_discrete[i].du[1:length_z, 1:stage], w[1:stage])
    __maybe_matmul!(
        z, k_interp.u[i][1:length_z, 1:(s_star - stage)], w[(stage + 1):s_star], true, true)
    z .= z .* dt .+ id.u[i]

    return nothing
end
@views function sum_stages!(z::AbstractArray, id::MIRKInterpolation,
        cache::MIRKCache{iip, T, use_both, NoDiffCacheNeeded, fit_parameters},
        w, i::Int, ::Type{Val{0}}) where {iip, T, use_both, fit_parameters}
    (; stage, k_discrete, k_interp) = cache
    (; s_star) = cache.ITU
    dt = cache.mesh_dt[i]
    length_z = length(z)
    z .= zero(z)
    __maybe_matmul!(z, k_discrete[i][1:length_z, 1:stage], w[1:stage])
    __maybe_matmul!(
        z, k_interp.u[i][1:length_z, 1:(s_star - stage)], w[(stage + 1):s_star], true, true)
    z .= z .* dt .+ id.u[i]

    return nothing
end

@views function sum_stages!(z′, id::MIRKInterpolation,
        cache::MIRKCache{iip, T, use_both, DiffCacheNeeded, fit_parameters},
        w′, i::Int, ::Type{Val{1}}) where {iip, T, use_both, fit_parameters}
    (; stage, k_discrete, k_interp) = cache
    (; s_star) = cache.ITU
    length_z = length(z)
    z′ .= zero(z′)
    __maybe_matmul!(z′, k_discrete[i].du[1:length_z, 1:stage], w′[1:stage])
    __maybe_matmul!(z′, k_interp.u[i][1:length_z, 1:(s_star - stage)],
        w′[(stage + 1):s_star], true, true)

    return nothing
end
@views function sum_stages!(z′, id::MIRKInterpolation,
        cache::MIRKCache{iip, T, use_both, NoDiffCacheNeeded, fit_parameters},
        w′, i::Int, ::Type{Val{1}}) where {iip, T, use_both, fit_parameters}
    (; stage, k_discrete, k_interp) = cache
    (; s_star) = cache.ITU
    length_z = length(z)

    z′ .= zero(z′)
    __maybe_matmul!(z′, k_discrete[i][1:length_z, 1:stage], w′[1:stage])
    __maybe_matmul!(z′, k_interp.u[i][1:length_z, 1:(s_star - stage)],
        w′[(stage + 1):s_star], true, true)

    return nothing
end

@inline __build_interpolation(
    cache::MIRKCache, u::AbstractVector) = MIRKInterpolation(cache.mesh, u, cache)

# Intermidiate solution for evaluating boundry conditions
# basically simplified version of the interpolation for MIRK
function (s::EvalSol{C})(tval::Number) where {C <: MIRKCache}
    (; t, u, cache) = s
    (; alg, stage, k_discrete) = cache
    # Quick handle for the case where tval is at the boundary
    (tval == t[1]) && return first(u)
    (tval == t[end]) && return last(u)
    z = zero(last(u))
    ii = interval(t, tval)
    dt = cache.mesh_dt[ii]
    τ = (tval - t[ii]) / dt
    w, _ = evalsol_interp_weights(τ, alg)
    K = __needs_diffcache(alg.jac_alg) ? @view(k_discrete[ii].du[:, 1:stage]) :
        @view(k_discrete[ii][:, 1:stage])
    __maybe_matmul!(z, K, @view(w[1:stage]))
    z .= z .* dt .+ u[ii]
    return z
end

# Interpolate intermidiate solution at multiple points
function (s::EvalSol{C})(tvals::AbstractArray{<:Number}) where {C <: MIRKCache}
    (; t, u, cache) = s
    (; alg, stage, k_discrete, mesh_dt) = cache
    # Quick handle for the case where tval is at the boundary
    zvals = [zero(last(u)) for _ in tvals]
    for (i, tval) in enumerate(tvals)
        (tval == t[1]) && return first(u)
        (tval == t[end]) && return last(u)
        ii = interval(t, tval)
        dt = mesh_dt[ii]
        τ = (tval - t[ii]) / dt
        w, _ = evalsol_interp_weights(τ, alg)
        K = __needs_diffcache(alg.jac_alg) ? @view(k_discrete[ii].du[:, 1:stage]) :
            @view(k_discrete[ii][:, 1:stage])
        __maybe_matmul!(zvals[i], K, @view(w[1:stage]))
        zvals[i] .= zvals[i] .* dt .+ u[ii]
    end
    return zvals
end

# Intermidiate derivative solution for evaluating boundry conditions
function (s::EvalSol{C})(tval::Number, ::Type{Val{1}}) where {C <: MIRKCache}
    (; t, u, cache) = s
    (; alg, stage, k_discrete, mesh_dt) = cache
    z′ = zeros(typeof(tval), 2)
    ii = interval(t, tval)
    dt = mesh_dt[ii]
    τ = (tval - t[ii]) / dt
    _, w′ = interp_weights(τ, alg)
    __maybe_matmul!(z′, @view(k_discrete[ii].du[:, 1:stage]), @view(w′[1:stage]))
    return z′
end

"""
Construct n root-finding problems and solve them to find the critical points with continuous derivative polynomials
"""
function __construct_then_solve_root_problem(
        sol::EvalSol{C}, tspan::Tuple) where {C <: MIRKCache}
    (; alg) = sol.cache
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

@inline function evalsol_interp_weights(τ::T, ::MIRK2) where {T}
    w = [0, τ * (1 - τ / 2), τ^2 / 2]

    #     Derivative polynomials.

    wp = [0, 1 - τ, τ]
    return T.(w), T.(wp)
end
@inline function evalsol_interp_weights(τ::T, ::MIRK3) where {T}
    w = [τ / 4.0 * (2.0 * τ^2 - 5.0 * τ + 4.0),
        -3.0 / 4.0 * τ^2 * (2.0 * τ - 3.0), τ^2 * (τ - 1.0)]

    #     Derivative polynomials.

    wp = [3.0 / 2.0 * (τ - 2.0 / 3.0) * (τ - 1.0),
        -9.0 / 2.0 * τ * (τ - 1.0), 3.0 * τ * (τ - 2.0 / 3.0)]
    return T.(w), T.(wp)
end
@inline function evalsol_interp_weights(τ::T, ::MIRK4) where {T}
    t2 = τ * τ
    tm1 = τ - 1.0
    t4m3 = τ * 4.0 - 3.0
    t2m1 = τ * 2.0 - 1.0

    w = [-τ * (2.0 * τ - 3.0) * (2.0 * t2 - 3.0 * τ + 2.0) / 6.0,
        t2 * (12.0 * t2 - 20.0 * τ + 9.0) / 6.0,
        2.0 * t2 * (6.0 * t2 - 14.0 * τ + 9.0) / 3.0, -16.0 * t2 * tm1 * tm1 / 3.0]

    #   Derivative polynomials

    wp = [-tm1 * t4m3 * t2m1 / 3.0, τ * t2m1 * t4m3,
        4.0 * τ * t4m3 * tm1, -32.0 * τ * t2m1 * tm1 / 3.0]
    return T.(w), T.(wp)
end
@inline function evalsol_interp_weights(τ::T, ::MIRK5) where {T}
    w = [
        τ * (22464.0 - 83910.0 * τ + 143041.0 * τ^2 - 113808.0 * τ^3 + 33256.0 * τ^4) /
        22464.0,
        τ^2 * (-2418.0 + 12303.0 * τ - 19512.0 * τ^2 + 10904.0 * τ^3) / 3360.0,
        -8 / 81 * τ^2 * (-78.0 + 209.0 * τ - 204.0 * τ^2 + 8.0 * τ^3),
        -25 / 1134 * τ^2 * (-390.0 + 1045.0 * τ - 1020.0 * τ^2 + 328.0 * τ^3),
        -25 / 5184 * τ^2 * (390.0 + 255.0 * τ - 1680.0 * τ^2 + 2072.0 * τ^3),
        279841 / 168480 * τ^2 * (-6.0 + 21.0 * τ - 24.0 * τ^2 + 8.0 * τ^3)]

    #   Derivative polynomials

    wp = [
        1.0 - 13985 // 1872 * τ + 143041 // 7488 * τ^2 - 2371 // 117 * τ^3 +
        20785 // 2808 * τ^4,
        -403 // 280 * τ + 12303 // 1120 * τ^2 - 813 // 35 * τ^3 + 1363 // 84 * τ^4,
        416 // 27 * τ - 1672 // 27 * τ^2 + 2176 // 27 * τ^3 - 320 // 81 * τ^4,
        3250 // 189 * τ - 26125 // 378 * τ^2 + 17000 // 189 * τ^3 - 20500 // 567 * τ^4,
        -1625 // 432 * τ - 2125 // 576 * τ^2 + 875 // 27 * τ^3 - 32375 // 648 * τ^4,
        -279841 // 14040 * τ + 1958887 // 18720 * τ^2 - 279841 // 1755 * τ^3 +
        279841 // 4212 * τ^4]
    return T.(w), T.(wp)
end
@inline function evalsol_interp_weights(τ::T, ::MIRK6) where {T}
    w = [
        τ - 28607 // 7434 * τ^2 - 166210 // 33453 * τ^3 + 334780 // 11151 * τ^4 -
        1911296 // 55755 * τ^5 + 406528 // 33453 * τ^6,
        777 // 590 * τ^2 - 2534158 // 234171 * τ^3 + 2088580 // 78057 * τ^4 -
        10479104 // 390285 * τ^5 + 11328512 // 1170855 * τ^6,
        -1008 // 59 * τ^2 + 222176 // 1593 * τ^3 - 180032 // 531 * τ^4 +
        876544 // 2655 * τ^5 - 180224 // 1593 * τ^6,
        -1008 // 59 * τ^2 + 222176 // 1593 * τ^3 - 180032 // 531 * τ^4 +
        876544 // 2655 * τ^5 - 180224 // 1593 * τ^6,
        -378 // 59 * τ^2 + 27772 // 531 * τ^3 - 22504 // 177 * τ^4 + 109568 // 885 * τ^5 -
        22528 // 531 * τ^6,
        -95232 // 413 * τ^2 + 62384128 // 33453 * τ^3 - 49429504 // 11151 * τ^4 +
        46759936 // 11151 * τ^5 - 46661632 // 33453 * τ^6,
        896 // 5 * τ^2 - 4352 // 3 * τ^3 + 3456 * τ^4 - 16384 // 5 * τ^5 +
        16384 // 15 * τ^6,
        50176 // 531 * τ^2 - 179554304 // 234171 * τ^3 + 143363072 // 78057 * τ^4 -
        136675328 // 78057 * τ^5 + 137363456 // 234171 * τ^6,
        16384 // 441 * τ^3 - 16384 // 147 * τ^4 + 16384 // 147 * τ^5 - 16384 // 441 * τ^6]

    #     Derivative polynomials.

    wp = [
        1 - 28607 // 3717 * τ - 166210 // 11151 * τ^2 + 1339120 // 11151 * τ^3 -
        1911296 // 11151 * τ^4 + 813056 // 11151 * τ^5,
        777 // 295 * τ - 2534158 // 78057 * τ^2 + 8354320 // 78057 * τ^3 -
        10479104 // 78057 * τ^4 + 22657024 // 390285 * τ^5,
        -2016 // 59 * τ + 222176 // 531 * τ^2 - 720128 // 531 * τ^3 + 876544 // 531 * τ^4 -
        360448 // 531 * τ^5,
        -2016 // 59 * τ + 222176 // 531 * τ^2 - 720128 // 531 * τ^3 + 876544 // 531 * τ^4 -
        360448 // 531 * τ^5,
        -756 // 59 * τ + 27772 // 177 * τ^2 - 90016 // 177 * τ^3 + 109568 // 177 * τ^4 -
        45056 // 177 * τ^5,
        -190464 // 413 * τ + 62384128 // 11151 * τ^2 - 197718016 // 11151 * τ^3 +
        233799680 // 11151 * τ^4 - 93323264 // 11151 * τ^5,
        1792 // 5 * τ - 4352 * τ^2 + 13824 * τ^3 - 16384 * τ^4 + 32768 // 5 * τ^5,
        100352 // 531 * τ - 179554304 // 78057 * τ^2 + 573452288 // 78057 * τ^3 -
        683376640 // 78057 * τ^4 + 274726912 // 78057 * τ^5,
        16384 // 147 * τ^2 - 65536 // 147 * τ^3 + 81920 // 147 * τ^4 - 32768 // 147 * τ^5]
    return T.(w), T.(wp)
end

@inline function evalsol_interp_weights(τ::T, ::MIRK6I) where {T}
    w = [
        -(12233 + 1450 * sqrt(7)) *
        (800086000 * τ^5 + 63579600 * sqrt(7) * τ^4 - 2936650584 * τ^4 + 4235152620 * τ^3 -
         201404565 * sqrt(7) * τ^3 + 232506630 * sqrt(7) * τ^2 - 3033109390 * τ^2 +
         1116511695 * τ - 116253315 * sqrt(7) * τ + 22707000 * sqrt(7) - 191568780) *
        τ / 2112984835740,
        -(-10799 + 650 * sqrt(7)) *
        (24962000 * τ^4 + 473200 * sqrt(7) * τ^3 - 67024328 * τ^3 - 751855 * sqrt(7) * τ^2 +
         66629600 * τ^2 - 29507250 * τ +
         236210 * sqrt(7) * τ +
         5080365 +
         50895 * sqrt(7)) *
        τ^2 / 29551834260,
        7 / 1274940 *
        (259 + 50 * sqrt(7)) *
        (14000 * τ^4 - 48216 * τ^3 + 1200 * sqrt(7) * τ^3 - 3555 * sqrt(7) * τ^2 +
         62790 * τ^2 +
         3610 * sqrt(7) * τ - 37450 * τ + 9135 - 1305 * sqrt(7)) *
        τ^2,
        7 / 1274940 *
        (259 + 50 * sqrt(7)) *
        (14000 * τ^4 - 48216 * τ^3 + 1200 * sqrt(7) * τ^3 - 3555 * sqrt(7) * τ^2 +
         62790 * τ^2 +
         3610 * sqrt(7) * τ - 37450 * τ + 9135 - 1305 * sqrt(7)) *
        τ^2,
        16 / 2231145 *
        (259 + 50 * sqrt(7)) *
        (14000 * τ^4 - 48216 * τ^3 + 1200 * sqrt(7) * τ^3 - 3555 * sqrt(7) * τ^2 +
         62790 * τ^2 +
         3610 * sqrt(7) * τ - 37450 * τ + 9135 - 1305 * sqrt(7)) *
        τ^2,
        4 / 1227278493 *
        (740 * sqrt(7) - 6083) *
        (1561000 * τ^2 - 2461284 * τ - 109520 * sqrt(7) * τ + 979272 + 86913 * sqrt(7)) *
        (τ - 1)^2 *
        τ^2,
        -49 / 63747 * sqrt(7) * (20000 * τ^2 - 20000 * τ + 3393) * (τ - 1)^2 * τ^2,
        -1250000000 / 889206903 * (28 * τ^2 - 28 * τ + 9) * (τ - 1)^2 * τ^2]

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
        -98 / 21249 * sqrt(7) * (τ - 1) * (100 * τ - 13) * (100 * τ - 87) * (2 * τ - 1) * τ,
        -1250000000 / 2074816107 *
        (14 * τ - 7 + sqrt(7)) *
        (τ - 1) *
        (14 * τ - 7 - sqrt(7)) *
        (2 * τ - 1) *
        τ]
    return T.(w), T.(wp)
end
