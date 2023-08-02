"""
    interp_eval(mesh, Y, t, k_discrete, k_interp)

After we construct an interpolant, we use interp_eval to evaluate it.
"""
@views function interp_eval(S::BVPSystem,
    cache::AbstractMIRKCache,
    alg::Union{GeneralMIRK, MIRK},
    ITU::MIRKInterpTableau,
    t,
    k_interp)
    mesh, Y = S.x, S.y
    k_discrete = cache.k_discrete
    # EXPORTS: z, z_prime
    i = interval(mesh, t)
    dt = mesh[i + 1] - mesh[i]
    tau = (t - mesh[i]) / dt
    weights, weights_prime = interp_weights(tau, alg)
    z, z_prime = sum_stages(S,
        ITU,
        weights,
        weights_prime,
        k_discrete[i, :],
        k_interp[i, :],
        Y[i])
    return z, z_prime
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
@views function mesh_selector(S::BVPSystem{T}, alg::Union{GeneralMIRK, MIRK}, defect, abstol) where {T}
    #exports: mesh_new, Nsub_star, info
    mesh_current, n = S.x, S.N - 1

    #TODO: Need users to manually specify, here, we set it as 3000 by default.
    MxNsub = 3000

    safety_factor = T(1.3)
    rho = T(1.0) # Set rho=1 means mesh distribution will take place everytime.
    r1 = T(0.0)
    r2 = T(0.0)
    Nsub_star = 0
    Nsub_star_ub = 4 * n
    Nsub_star_lb = n ÷ 2

    info = ReturnCode.Success
    p = alg_order(alg)
    s_hat = similar(S.x, n)

    for i in eachindex(s_hat)
        h = mesh_current[i + 1] - mesh_current[i]
        norm = amax(defect[i, :])
        s_hat[i] = (norm / abstol)^(T(1) / (p + 1))
        r1 = max(s_hat[i], r1)
        r2 += s_hat[i]
        s_hat[i] /= h
    end

    r3 = r2 / n
    n_predict = round(Int, (safety_factor * r2) + 1)
    n_ = T(0.1) * n
    n_predict = ifelse(abs((n_predict - n)) < n_, round(Int, n + n_), n)

    if r1 ≤ rho * r3
        Nsub_star = 2 * n
        if Nsub_star > MxNsub # Need to determine the too large threshold
            #("New mesh would be too large")
            info = ReturnCode.Failure
            mesh_new = mesh_current  ## Return the current mesh to preserve type stability
        else
            #println("Half the current mesh")
            mesh_new = half_mesh(mesh_current)
        end
    else
        Nsub_star = clamp(n_predict, Nsub_star_lb, Nsub_star_ub)
        if Nsub_star > MxNsub
            # Mesh redistribution fails
            #println("New mesh would be too large")
            info = ReturnCode.Failure
            mesh_new = mesh_current  ## Return the current mesh to preserve type stability
        else
            #println("Mesh redistributing")
            mesh_new = redistribute(mesh_current, Nsub_star, s_hat)
        end
    end
    return mesh_new, Nsub_star, info
end

"""
    redistribute(mesh_current, Nsub_star, s_hat)

Generate a new mesh based on the .
"""
function redistribute(mesh_current::AbstractVector,
    Nsub_star::Int,
    s_hat::AbstractVector)
    n = length(mesh_current)
    mesh_new = similar(mesh_current, Nsub_star + 1)
    ζ = sum(k -> s_hat[k] * (mesh_current[k + 1] - mesh_current[k]), 1:(n - 1)) / Nsub_star
    T = typeof(ζ)
    k = 1
    i = 0
    mesh_new[1] = mesh_current[1]
    t = mesh_current[1]
    integral = T(0)
    while k ≤ n - 1
        next_piece = s_hat[k] * (mesh_current[k + 1] - t)
        _int_next = integral + next_piece
        if _int_next > ζ
            mesh_new[i + 2] = (ζ - integral) / s_hat[k] + t
            t = mesh_new[i + 2]
            i += 1
            integral = T(0)
        else
            integral = _int_next
            t = mesh_current[k + 1]
            k += 1
        end
    end
    mesh_new[Nsub_star + 1] = mesh_current[n]
    return mesh_new
end

"""
    half_mesh(mesh_current)

The input mesh_current has length of n+1

Divide the original subinterval into two equal length subinterval.
"""
@views function half_mesh(mesh_current::AbstractVector{T}) where {T}
    n = length(mesh_current) - 1
    mesh_new = similar(mesh_current, 2n + 1)
    mesh_new[begin] = mesh_current[begin]
    for i in eachindex(mesh_current)[begin:end - 1]
        mesh_new[2i + 1] = mesh_current[i + 1]
        mesh_new[2i] = (mesh_current[i + 1] + mesh_current[i]) / T(2)
    end
    return mesh_new
end

idamax(x) = last(findmax(abs, x))
amax(x) = first(findmax(abs, x))

"""
    defect_estimate(prob, Y, alg, n, dt, mesh, k_discrete)

defect_estimate use the discrete solution approximation Y, plus stages of 
the RK method in 'k_discrete', plus some new stages in 'k_interp' to construct 
an interpolant
"""
function defect_estimate(S::BVPSystem,
    cache::AbstractMIRKCache,
    alg::Union{GeneralMIRK, MIRK},
    ITU::MIRKInterpTableau)
    n, len, Y, p, k_discrete, mesh, f = S.N - 1,
    S.M,
    S.y,
    S.p,
    cache.k_discrete,
    S.x,
    S.fun!
    s, s_star, tau_star = S.s, ITU.s_star, ITU.τ_star

    # Initialization
    defect = zeros(Float64, n, len)
    #s, s_star, tau_star, x_star, v_star, c_star = setup_coeff(alg)

    f_sample_1, f_sample_2 = zeros(Float64, len), zeros(Float64, len)
    def_1, def_2 = zeros(Float64, len), zeros(Float64, len)
    temp_1, temp_2 = zeros(Float64, len), zeros(Float64, len)
    estimate_1, estimate_2 = zeros(Float64), zeros(Float64)

    # Evaluate at the first sample point
    weights_1, weights_1_prime = interp_weights(tau_star, alg)
    # Evaluate at the second sample point
    weights_2, weights_2_prime = interp_weights(1.0 - tau_star, alg)

    k_interp = similar([S.y[1]], S.N - 1, (s_star - s))
    for i in 1:n
        dt = mesh[i + 1] - mesh[i]

        k_interp[i, :] = interp_setup(S, mesh[i], dt, Y[i], Y[i + 1], ITU, k_discrete[i, :])

        # Sample point 1
        z, z_prime = sum_stages(S, ITU, weights_1, weights_1_prime, k_discrete[i, :],
            k_interp[i, :], Y[i])
        f(f_sample_1, z, p, mesh[i] + tau_star * dt)
        z_prime .= z_prime .- f_sample_1
        def_1 = copy(z_prime)
        for j in 1:len
            temp_1[j] = def_1[j] / (abs(f_sample_1[j]) + 1.0)
        end
        estimate_1 = maximum(abs.(temp_1))

        # Sample point 2
        z, z_prime = sum_stages(S, ITU, weights_2, weights_2_prime, k_discrete[i, :],
            k_interp[i, :], Y[i])
        f(f_sample_2, z, p, mesh[i] + (1.0 - tau_star) * dt)
        z_prime .= z_prime .- f_sample_2
        def_2 .= copy(z_prime)
        for j in 1:len
            temp_2[j] = def_2[j] / (abs(f_sample_2[j]) + 1.0)
        end
        estimate_2 = maximum(abs.(temp_2))

        # Compare defect estimates for the above two sample points
        if estimate_1 > estimate_2
            defect[i, :] = temp_1
        else
            defect[i, :] = temp_2
        end
    end
    defect_norm = maximum(abs.(defect))
    return defect, defect_norm, k_interp
end

"""
    interp_setup

interp_setup prepare the extra stages in ki_interp for interpolant construction.
Here, the ki_interp is the stages in one subinterval.
"""
function interp_setup(S::BVPSystem,
    tim1,
    dt,
    y_left,
    y_right,
    ITU::MIRKInterpTableau,
    ki_discrete)
    len, f, p = S.M, S.fun!, S.p
    #TODO: Temporary, only debuging
    s, s_star, c_star, v_star, x_star = S.s,
    ITU.s_star,
    ITU.c_star,
    ITU.v_star,
    ITU.x_star
    x_star = x_star[:]
    # EXPORTS: ki_interp
    ki_interp = similar([zeros(Float64, len)], s_star - s)
    for r in 1:(s_star - s)
        new_stages = zeros(Float64, len)
        for j in 1:s
            new_stages .= new_stages .+
                          x_star[(j - 1) * (s_star - s) + r] .*
                          ki_discrete[j]
        end
        for j in 1:(r - 1)
            new_stages .= new_stages .+
                          x_star[(j + s - 1) * (s_star - s) + r] .*
                          ki_interp[j]
        end
        new_stages .= new_stages .* dt
        new_stages .= new_stages .+ (1 - v_star[r]) .* y_left
        new_stages .= new_stages .+ v_star[r] .* y_right

        temp = zeros(Float64, len)
        f(temp, new_stages, p, tim1 + c_star[r] * dt)
        ki_interp[r] = temp
    end
    return ki_interp
end

"""
    sum_stages(weights, weights_prime, ki_discrete, ki_interp, len, dt, y)

sum_stages add the discrete solution, RK method stages and extra stages to construct interpolant.

Here, ki_discrete is a matrix stored with discrete RK stages in the ith interval, ki_discrete has legnth of s*neqns
Here, ki_interp is a matrix stored with interpolation coefficients in the ith interval, ki_interp has length of (s_star-s)*neqns
"""
function sum_stages(S::BVPSystem,
    ITU::MIRKInterpTableau,
    weights,
    weights_prime,
    ki_discrete::AbstractVector{T1},
    ki_interp::AbstractVector{T2},
    y) where {T1, T2}
    len, mesh = S.M, S.x
    dt = mesh[end] - mesh[end - 1]
    s, s_star = S.s, ITU.s_star
    elType_z = promote_type(eltype(weights), eltype(T1), eltype(T2))
    elType_zprime = promote_type(eltype(weights_prime), eltype(T1), eltype(T2))
    z = similar(first(ki_discrete), elType_z)
    z_prime = similar(first(ki_discrete), elType_zprime)

    foreach(1:s) do i
        z .+= weights[i] .* ki_discrete[i]
        z_prime .+= weights_prime[i] .* ki_discrete[i]
    end
    foreach(1:(s_star - s)) do i
        z .+= weights[i + s] .* ki_interp[i]
        z_prime .+= weights_prime[i + s] .* ki_interp[i]
    end
    @. z = z * dt + y

    return z, z_prime
end

"""
    interp_weights(tau, alg)

interp_weights: solver-specified interpolation weights and its first derivative
"""
function interp_weights(tau, alg::Union{GeneralMIRK, MIRK})
    if alg_order(alg) == 3
        w = [tau / 4.0 * (2.0 * tau^2 - 5.0 * tau + 4.0),
            -3.0 / 4.0 * tau^2 * (2.0 * tau - 3.0),
            tau^2 * (tau - 1.0)]

        #     Derivative polynomials.

        wp = [3.0 / 2.0 * (tau - 2.0 / 3.0) * (tau - 1.0),
            -9.0 / 2.0 * tau * (tau - 1.0),
            3.0 * tau * (tau - 2.0 / 3.0)]
        return w, wp
    elseif alg_order(alg) == 4
        t2 = tau * tau
        tm1 = tau - 1.0
        t4m3 = tau * 4.0 - 3.0
        t2m1 = tau * 2.0 - 1.0

        w = [-tau * (2.0 * tau - 3.0) * (2.0 * t2 - 3.0 * tau + 2.0) / 6.0,
            t2 * (12.0 * t2 - 20.0 * tau + 9.0) / 6.0,
            2.0 * t2 * (6.0 * t2 - 14.0 * tau + 9.0) / 3.0,
            -16.0 * t2 * tm1 * tm1 / 3.0]

        #   Derivative polynomials

        wp = [-tm1 * t4m3 * t2m1 / 3.0,
            tau * t2m1 * t4m3,
            4.0 * tau * t4m3 * tm1,
            -32.0 * tau * t2m1 * tm1 / 3.0]

        return w, wp
    elseif alg_order(alg) == 5
        w = [
            tau * (22464.0 - 83910.0 * tau + 143041.0 * tau^2 - 113808.0 * tau^3 +
             33256.0 * tau^4) / 22464.0,
            tau^2 * (-2418.0 + 12303.0 * tau - 19512.0 * tau^2 + 10904.0 * tau^3) / 3360.0,
            -8 / 81 * tau^2 * (-78.0 + 209.0 * tau - 204.0 * tau^2 + 8.0 * tau^3),
            -25 / 1134 * tau^2 * (-390.0 + 1045.0 * tau - 1020.0 * tau^2 + 328.0 * tau^3),
            -25 / 5184 * tau^2 * (390.0 + 255.0 * tau - 1680.0 * tau^2 + 2072.0 * tau^3),
            279841 / 168480 * tau^2 * (-6.0 + 21.0 * tau - 24.0 * tau^2 + 8.0 * tau^3)]

        #     Derivative polynomials.

        wp = [
            1.0 - 13985 // 1872 * tau + 143041 // 7488 * tau^2 - 2371 // 117 * tau^3 +
            20785 // 2808 * tau^4,
            -403 // 280 * tau + 12303 // 1120 * tau^2 - 813 // 35 * tau^3 +
            1363 // 84 * tau^4,
            416 // 27 * tau - 1672 // 27 * tau^2 + 2176 // 27 * tau^3 - 320 // 81 * tau^4,
            3250 // 189 * tau - 26125 // 378 * tau^2 + 17000 // 189 * tau^3 -
            20500 // 567 * tau^4,
            -1625 // 432 * tau - 2125 // 576 * tau^2 + 875 // 27 * tau^3 -
            32375 // 648 * tau^4,
            -279841 // 14040 * tau + 1958887 // 18720 * tau^2 - 279841 // 1755 * tau^3 +
            279841 // 4212 * tau^4]

        return w, wp

    elseif alg_order(alg) == 6
        w = [
            tau - 28607 // 7434 * tau^2 - 166210 // 33453 * tau^3 +
            334780 // 11151 * tau^4 -
            1911296 // 55755 * tau^5 + 406528 // 33453 * tau^6,
            777 // 590 * tau^2 - 2534158 // 234171 * tau^3 + 2088580 // 78057 * tau^4 -
            10479104 // 390285 * tau^5 + 11328512 // 1170855 * tau^6,
            -1008 // 59 * tau^2 + 222176 // 1593 * tau^3 - 180032 // 531 * tau^4 +
            876544 // 2655 * tau^5 - 180224 // 1593 * tau^6,
            -1008 // 59 * tau^2 + 222176 // 1593 * tau^3 - 180032 // 531 * tau^4 +
            876544 // 2655 * tau^5 - 180224 // 1593 * tau^6,
            -378 // 59 * tau^2 + 27772 // 531 * tau^3 - 22504 // 177 * tau^4 +
            109568 // 885 * tau^5 - 22528 // 531 * tau^6,
            -95232 // 413 * tau^2 + 62384128 // 33453 * tau^3 - 49429504 // 11151 * tau^4 +
            46759936 // 11151 * tau^5 - 46661632 // 33453 * tau^6,
            896 // 5 * tau^2 - 4352 // 3 * tau^3 + 3456 * tau^4 - 16384 // 5 * tau^5 +
            16384 // 15 * tau^6,
            50176 // 531 * tau^2 - 179554304 // 234171 * tau^3 +
            143363072 // 78057 * tau^4 -
            136675328 // 78057 * tau^5 + 137363456 // 234171 * tau^6,
            16384 // 441 * tau^3 - 16384 // 147 * tau^4 + 16384 // 147 * tau^5 -
            16384 // 441 * tau^6]

        #     Derivative polynomials.

        wp = [
            1 - 28607 // 3717 * tau - 166210 // 11151 * tau^2 + 1339120 // 11151 * tau^3 -
            1911296 // 11151 * tau^4 + 813056 // 11151 * tau^5,
            777 // 295 * tau - 2534158 // 78057 * tau^2 + 8354320 // 78057 * tau^3 -
            10479104 // 78057 * tau^4 + 22657024 // 390285 * tau^5,
            -2016 // 59 * tau + 222176 // 531 * tau^2 - 720128 // 531 * tau^3 +
            876544 // 531 * tau^4 - 360448 // 531 * tau^5,
            -2016 // 59 * tau + 222176 // 531 * tau^2 - 720128 // 531 * tau^3 +
            876544 // 531 * tau^4 - 360448 // 531 * tau^5,
            -756 // 59 * tau + 27772 // 177 * tau^2 - 90016 // 177 * tau^3 +
            109568 // 177 * tau^4 - 45056 // 177 * tau^5,
            -190464 // 413 * tau + 62384128 // 11151 * tau^2 - 197718016 // 11151 * tau^3 +
            233799680 // 11151 * tau^4 - 93323264 // 11151 * tau^5,
            1792 // 5 * tau - 4352 * tau^2 + 13824 * tau^3 - 16384 * tau^4 +
            32768 // 5 * tau^5,
            100352 // 531 * tau - 179554304 // 78057 * tau^2 + 573452288 // 78057 * tau^3 -
            683376640 // 78057 * tau^4 + 274726912 // 78057 * tau^5,
            16384 // 147 * tau^2 - 65536 // 147 * tau^3 + 81920 // 147 * tau^4 -
            32768 // 147 * tau^5]

        return w, wp
    end
end
