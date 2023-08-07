"""
    interp_eval(S::BVPSystem, cache::AbstractMIRKCache,
        alg::Union{GeneralMIRK, MIRK}, ITU::MIRKInterpTableau, t, k_interp, mesh, y, dt_)

After we construct an interpolant, we use interp_eval to evaluate it.
"""
@views function interp_eval(S::BVPSystem, cache::AbstractMIRKCache,
    alg::Union{GeneralMIRK, MIRK}, ITU::MIRKInterpTableau, t, k_interp, mesh, y, dt_)
    @unpack k_discrete = cache
    i = interval(mesh, t)
    dt = dt_[i]
    τ = (t - mesh[i]) / dt
    w, w′ = interp_weights(τ, alg)
    z, z′ = sum_stages(S, ITU, w, w′, k_discrete[:, :, i:i], k_interp[:, :, i:i],
        mesh[i:(i + 1)], y[:, i:i], dt_[i:i])
    return dropdims(z, dims = 2), dropdims(z′, dims = 2)
end

"""
    interval(mesh, t)

Find the interval that `t` belongs to in `mesh`. Assumes that `mesh` is sorted.
"""
function interval(mesh, t)
    t == first(mesh) && return 1
    t == last(mesh) && return length(mesh) - 1
    return searchsortedfirst(mesh, t) - 1
end

"""
    mesh_selector(S::BVPSystem, alg::Union{GeneralMIRK, MIRK}, defect, abstol)

Generate new mesh based on the defect.
"""
@views function mesh_selector(S::BVPSystem, alg::Union{GeneralMIRK, MIRK}, defect,
    abstol, mesh, dt)
    T = eltype(S)
    #exports: mesh_new, Nsub_star, info
    @unpack M, N = S

    #TODO: Need users to manually specify, here, we set it as 3000 by default.
    MxNsub = 3000

    safety_factor = T(1.3)
    ρ = T(1.0) # Set rho=1 means mesh distribution will take place everytime.
    Nsub_star = 0
    Nsub_star_ub = 4 * (N - 1)
    Nsub_star_lb = N ÷ 2

    info = ReturnCode.Success
    order = alg_order(alg)
    ŝ = similar(defect, N - 1)

    ŝ = vec(amax(defect; dims = 1))
    ŝ .= (ŝ ./ abstol) .^ (T(1) / (order + 1))
    r₁ = maximum(ŝ)
    r₂ = sum(ŝ)
    r₃ = r₂ / (N - 1)

    n_predict = round(Int, (safety_factor * r₂) + 1)
    n = N - 1
    n_ = T(0.1) * n
    n_predict = ifelse(abs((n_predict - n)) < n_, round(Int, n + n_), n_predict)

    if r₁ ≤ ρ * r₂
        Nsub_star = 2 * (N - 1)
        if Nsub_star > MxNsub # Need to determine the too large threshold
            # println("New mesh would be too large")
            info = ReturnCode.Failure
            mesh_new = mesh  ## Return the current mesh to preserve type stability
        else
            # println("Half the current mesh")
            mesh_new = half_mesh(mesh)
        end
    else
        Nsub_star = clamp(n_predict, Nsub_star_lb, Nsub_star_ub)
        if Nsub_star > MxNsub
            # Mesh redistribution fails
            # println("New mesh would be too large")
            info = ReturnCode.Failure
            mesh_new = mesh  ## Return the current mesh to preserve type stability
        else
            # println("Mesh redistributing")
            ŝ ./= dt
            mesh_new = redistribute(mesh, Nsub_star, ŝ, dt)
        end
    end
    return mesh_new, Nsub_star, info
end

"""
    redistribute(mesh, Nsub_star, s_hat)

Generate a new mesh based on the .
"""
function redistribute(mesh, Nsub_star::Int, ŝ::AbstractVector{T}, dt) where {T}
    N = length(mesh)
    ζ = sum(ŝ .* dt) / Nsub_star
    k, i = 1, 0
    mesh_new = similar(mesh, Nsub_star + 1)
    mesh_new[1] = mesh[1]
    t = mesh[1]
    integral = T(0)
    while k ≤ N - 1
        next_piece = ŝ[k] * (mesh[k + 1] - t)
        _int_next = integral + next_piece
        if _int_next > ζ
            mesh_new[i + 2] = (ζ - integral) / ŝ[k] + t
            t = mesh_new[i + 2]
            i += 1
            integral = T(0)
        else
            integral = _int_next
            t = mesh[k + 1]
            k += 1
        end
    end
    mesh_new[end] = mesh[end]
    return mesh_new
end

"""
    half_mesh(mesh)

The input mesh has length of n+1

Divide the original subinterval into two equal length subinterval.
"""
@views function half_mesh(mesh::AbstractVector{T}) where {T}
    n = length(mesh) - 1
    mesh_new = similar(mesh, 2n + 1)
    mesh_new[begin] = mesh[begin]
    for i in eachindex(mesh)[begin:(end - 1)]
        mesh_new[2i + 1] = mesh[i + 1]
        mesh_new[2i] = (mesh[i + 1] + mesh[i]) / T(2)
    end
    return mesh_new
end

amax(x; kwargs...) = first(findmax(abs, x; kwargs...))

"""
    defect_estimate(S::BVPSystem, cache::AbstractMIRKCache, alg::Union{GeneralMIRK, MIRK},
        ITU::MIRKInterpTableau)

defect_estimate use the discrete solution approximation Y, plus stages of 
the RK method in 'k_discrete', plus some new stages in 'k_interp' to construct 
an interpolant
"""
@views function defect_estimate(S::BVPSystem, cache::AbstractMIRKCache,
    alg::Union{GeneralMIRK, MIRK}, ITU::MIRKInterpTableau, y, p, mesh, dt)
    @unpack M, N, stage, f! = S
    @unpack k_discrete = cache
    T = eltype(y)
    @unpack s_star, τ_star = ITU

    # Evaluate at the first sample point
    w₁, w₁′ = interp_weights(τ_star, alg)
    # Evaluate at the second sample point
    w₂, w₂′ = interp_weights(T(1) - τ_star, alg)

    k_interp = interp_setup(S, mesh, y, ITU, k_discrete, p, dt)

    # Sample Point 1
    tmp₁ = similar(y, M, N - 1)
    z₁, z₁′ = sum_stages(S, ITU, w₁, w₁′, k_discrete, k_interp, mesh, y, dt)

    # Sample Point 2
    tmp₂ = similar(y, M, N - 1)
    z₂, z₂′ = sum_stages(S, ITU, w₂, w₂′, k_discrete, k_interp, mesh, y, dt)

    defect = similar(y, M, N - 1)
    foreach(1:(N - 1)) do i
        dt = mesh[i + 1] - mesh[i]

        f!(tmp₁[:, i], z₁[:, i], p, mesh[i] + τ_star * dt)
        tmp₁ .= (z₁′[:, i] .- tmp₁) ./ (abs.(tmp₁) .+ T(1))
        est₁ = maximum(abs, tmp₁)

        f!(tmp₂[:, i], z₂[:, i], p, mesh[i] + (T(1) - τ_star) * dt)
        tmp₂ .= (z₂′[:, i] .- tmp₂) ./ (abs.(tmp₂) .+ T(1))
        est₂ = maximum(abs, tmp₂)

        defect[:, i] .= est₁ > est₂ ? tmp₁[:, i] : tmp₂[:, i]
    end
    defect_norm = maximum(abs, defect)

    return defect, defect_norm, k_interp
end

"""
    interp_setup

interp_setup prepare the extra stages in ki_interp for interpolant construction.
Here, the ki_interp is the stages in one subinterval.
"""
@views function interp_setup(S::BVPSystem, mesh, y, ITU::MIRKInterpTableau,
    k_discrete::AbstractArray{T, 3}, p, dt_) where {T}
    @unpack M, N, f!, stage = S
    @unpack x_star, s_star, c_star, v_star = ITU

    k_interp = similar(k_discrete, M, s_star - stage, N - 1)
    new_stages = similar(k_discrete, M, N - 1)

    dt = reshape(dt_, 1, N - 1)

    for r in 1:(s_star - stage)
        # M × (N - 1)
        idx₁ = ((1:stage) .- 1) .* (s_star - stage) .+ r
        idx₂ = ((1:(r - 1)) .+ stage .- 1) .* (s_star - stage) .+ r
        new_stages_ = reshape(new_stages, M, 1, N - 1)
        sum!(new_stages_, reshape(x_star[idx₁], 1, :) .* k_discrete[:, 1:stage, :])
        if r > 1
            sum!(new_stages_, reshape(x_star[idx₂], 1, :) .* k_interp[:, 1:(r - 1), :])
        end
        new_stages .= new_stages .* reshape(dt, 1, :) .+
                      (1 - v_star[r]) .* y[:, 1:(N - 1)] .+ v_star[r] .* y[:, 2:N]

        foreach(1:(N - 1)) do i
            f!(k_interp[:, r, i], new_stages[:, i], p, mesh[i] + c_star[r] * dt[i])
        end
    end

    return k_interp
end

"""
    sum_stages(weights, weights_prime, ki_discrete, ki_interp, len, dt, y)

sum_stages add the discrete solution, RK method stages and extra stages to construct interpolant.

Here, ki_discrete is a matrix stored with discrete RK stages in the ith interval, ki_discrete has legnth of s*neqns
Here, ki_interp is a matrix stored with interpolation coefficients in the ith interval, ki_interp has length of (s_star-s)*neqns
"""
@views function sum_stages(S::BVPSystem, ITU::MIRKInterpTableau,
    weights::AbstractVector{T₁}, weights′::AbstractVector{T₂},
    k_discrete::AbstractArray{T₃, 3}, k_interp::AbstractArray{T₄, 3},
    mesh::AbstractVector, y::AbstractMatrix{T₅}, dt) where {T₁, T₂, T₃, T₄, T₅}
    @unpack M, stage = S
    @unpack s_star = ITU
    N = length(mesh)

    z = similar(k_discrete, promote_type(T₁, T₃, T₄, T₅), M, N - 1)
    z′ = similar(k_discrete, promote_type(T₂, T₃, T₄), M, N - 1)

    foreach(1:(N - 1)) do i
        mul!(z[:, i:i], k_discrete[:, 1:stage, i], weights[1:stage])
        mul!(z′[:, i:i], k_discrete[:, 1:stage, i], weights′[1:stage])
        mul!(z[:, i:i], k_interp[:, 1:(s_star - stage), i], weights[(stage + 1):s_star],
            true, true)
        mul!(z′[:, i:i], k_interp[:, 1:(s_star - stage), i], weights′[(stage + 1):s_star],
            true, true)
    end

    z .= z .* reshape(dt, 1, :) .+ y[:, 1:(N - 1)]

    return z, z′
end

for order in (3, 4, 5, 6), alg in (Symbol("GeneralMIRK$(order)"), Symbol("MIRK$(order)"))
    @eval begin
        """
            interp_weights(τ, alg)

        interp_weights: solver-specified interpolation weights and its first derivative
        """
        function interp_weights(τ::T, ::$(alg)) where {T}
            if $(order == 3)
                w = [τ / 4.0 * (2.0 * τ^2 - 5.0 * τ + 4.0),
                    -3.0 / 4.0 * τ^2 * (2.0 * τ - 3.0),
                    τ^2 * (τ - 1.0)]

                #     Derivative polynomials.

                wp = [3.0 / 2.0 * (τ - 2.0 / 3.0) * (τ - 1.0),
                    -9.0 / 2.0 * τ * (τ - 1.0),
                    3.0 * τ * (τ - 2.0 / 3.0)]
            elseif $(order == 4)
                t2 = τ * τ
                tm1 = τ - 1.0
                t4m3 = τ * 4.0 - 3.0
                t2m1 = τ * 2.0 - 1.0

                w = [-τ * (2.0 * τ - 3.0) * (2.0 * t2 - 3.0 * τ + 2.0) / 6.0,
                    t2 * (12.0 * t2 - 20.0 * τ + 9.0) / 6.0,
                    2.0 * t2 * (6.0 * t2 - 14.0 * τ + 9.0) / 3.0,
                    -16.0 * t2 * tm1 * tm1 / 3.0]

                #   Derivative polynomials

                wp = [-tm1 * t4m3 * t2m1 / 3.0,
                    τ * t2m1 * t4m3,
                    4.0 * τ * t4m3 * tm1,
                    -32.0 * τ * t2m1 * tm1 / 3.0]
            elseif $(order == 5)
                w = [
                    τ * (22464.0 - 83910.0 * τ + 143041.0 * τ^2 - 113808.0 * τ^3 +
                     33256.0 * τ^4) / 22464.0,
                    τ^2 * (-2418.0 + 12303.0 * τ - 19512.0 * τ^2 + 10904.0 * τ^3) /
                    3360.0,
                    -8 / 81 * τ^2 * (-78.0 + 209.0 * τ - 204.0 * τ^2 + 8.0 * τ^3),
                    -25 / 1134 * τ^2 *
                    (-390.0 + 1045.0 * τ - 1020.0 * τ^2 + 328.0 * τ^3),
                    -25 / 5184 * τ^2 *
                    (390.0 + 255.0 * τ - 1680.0 * τ^2 + 2072.0 * τ^3),
                    279841 / 168480 * τ^2 *
                    (-6.0 + 21.0 * τ - 24.0 * τ^2 + 8.0 * τ^3)]

                #   Derivative polynomials

                wp = [
                    1.0 - 13985 // 1872 * τ + 143041 // 7488 * τ^2 -
                    2371 // 117 * τ^3 +
                    20785 // 2808 * τ^4,
                    -403 // 280 * τ + 12303 // 1120 * τ^2 - 813 // 35 * τ^3 +
                    1363 // 84 * τ^4,
                    416 // 27 * τ - 1672 // 27 * τ^2 + 2176 // 27 * τ^3 -
                    320 // 81 * τ^4,
                    3250 // 189 * τ - 26125 // 378 * τ^2 + 17000 // 189 * τ^3 -
                    20500 // 567 * τ^4,
                    -1625 // 432 * τ - 2125 // 576 * τ^2 + 875 // 27 * τ^3 -
                    32375 // 648 * τ^4,
                    -279841 // 14040 * τ + 1958887 // 18720 * τ^2 -
                    279841 // 1755 * τ^3 +
                    279841 // 4212 * τ^4]
            elseif $(order == 6)
                w = [
                    τ - 28607 // 7434 * τ^2 - 166210 // 33453 * τ^3 +
                    334780 // 11151 * τ^4 -
                    1911296 // 55755 * τ^5 + 406528 // 33453 * τ^6,
                    777 // 590 * τ^2 - 2534158 // 234171 * τ^3 +
                    2088580 // 78057 * τ^4 -
                    10479104 // 390285 * τ^5 + 11328512 // 1170855 * τ^6,
                    -1008 // 59 * τ^2 + 222176 // 1593 * τ^3 - 180032 // 531 * τ^4 +
                    876544 // 2655 * τ^5 - 180224 // 1593 * τ^6,
                    -1008 // 59 * τ^2 + 222176 // 1593 * τ^3 - 180032 // 531 * τ^4 +
                    876544 // 2655 * τ^5 - 180224 // 1593 * τ^6,
                    -378 // 59 * τ^2 + 27772 // 531 * τ^3 - 22504 // 177 * τ^4 +
                    109568 // 885 * τ^5 - 22528 // 531 * τ^6,
                    -95232 // 413 * τ^2 + 62384128 // 33453 * τ^3 -
                    49429504 // 11151 * τ^4 +
                    46759936 // 11151 * τ^5 - 46661632 // 33453 * τ^6,
                    896 // 5 * τ^2 - 4352 // 3 * τ^3 + 3456 * τ^4 -
                    16384 // 5 * τ^5 +
                    16384 // 15 * τ^6,
                    50176 // 531 * τ^2 - 179554304 // 234171 * τ^3 +
                    143363072 // 78057 * τ^4 -
                    136675328 // 78057 * τ^5 + 137363456 // 234171 * τ^6,
                    16384 // 441 * τ^3 - 16384 // 147 * τ^4 + 16384 // 147 * τ^5 -
                    16384 // 441 * τ^6]

                #     Derivative polynomials.

                wp = [
                    1 - 28607 // 3717 * τ - 166210 // 11151 * τ^2 +
                    1339120 // 11151 * τ^3 -
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
                    197718016 // 11151 * τ^3 +
                    233799680 // 11151 * τ^4 - 93323264 // 11151 * τ^5,
                    1792 // 5 * τ - 4352 * τ^2 + 13824 * τ^3 - 16384 * τ^4 +
                    32768 // 5 * τ^5,
                    100352 // 531 * τ - 179554304 // 78057 * τ^2 +
                    573452288 // 78057 * τ^3 -
                    683376640 // 78057 * τ^4 + 274726912 // 78057 * τ^5,
                    16384 // 147 * τ^2 - 65536 // 147 * τ^3 + 81920 // 147 * τ^4 -
                    32768 // 147 * τ^5]
            end
            return T.(w), T.(wp)
        end
    end
end
