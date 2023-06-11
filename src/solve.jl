using BandedMatrices

# The Solve Function
function DiffEqBase.__solve(prob::BVProblem, alg::Shooting; kwargs...)
    bc = prob.bc
    u0 = deepcopy(prob.u0)
    # Form a root finding function.
    loss = function (resid, minimizer)
        uEltype = eltype(minimizer)
        tmp_prob = remake(prob, u0 = minimizer)
        sol = solve(tmp_prob, alg.ode_alg; kwargs...)
        bc(resid, sol, sol.prob.p, sol.t)
        nothing
    end
    opt = alg.nlsolve(loss, u0)
    sol_prob = remake(prob, u0 = opt[1])
    sol = solve(sol_prob, alg.ode_alg; kwargs...)
    if sol.retcode == opt[2]
        DiffEqBase.solution_new_retcode(sol, ReturnCode.Success)
    else
        DiffEqBase.solution_new_retcode(sol, ReturnCode.Failure)
    end
    sol
end

function DiffEqBase.__solve(prob::BVProblem, alg::Union{GeneralMIRK, MIRK}; dt = 0.0,
    tol = 1e-6,
    kwargs...)
    if dt <= 0
        error("dt must be positive")
    end
    n = Int(cld((prob.tspan[2] - prob.tspan[1]), dt))
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
    defect_threshold = 0.1
    info = 0
    defect_norm = 10
    MxNsub = 3000
    while info == 0 && defect_norm > tol
        S = BVPSystem(prob, mesh, alg_order(alg))
        len = S.M
        tableau = constructMIRK(S)
        cache = alg_cache(alg, S)
        # Upper-level iteration
        vec_y = Array{eltype(S.y[1])}(undef, S.M * S.N)              # Vector
        reorder! = function (resid)
            # reorder the Jacobian matrix such that it is banded
            tmp_last = resid[end]
            for i in (length(resid) - 1):-1:1
                resid[i + 1] = resid[i]
            end
            resid[1], resid[end] = resid[end], tmp_last
        end
        loss = function (resid, minimizer)
            nest_vector!(S.y, minimizer)
            Φ!(S, tableau, cache)
            isa(prob.problem_type, TwoPointBVProblem) ? eval_bc_residual!(S) :
            general_eval_bc_residual!(S)
            flatten_vector!(resid, S.residual)
            reorder!(resid)
            nothing
        end

        jac_wrapper = BVPJacobianWrapper(loss)

        flatten_vector!(vec_y, S.y)
        opt = isa(prob.problem_type, TwoPointBVProblem) ?
              alg.nlsolve(ConstructJacobian(jac_wrapper, vec_y), vec_y) :
              alg.nlsolve(ConstructJacobian(jac_wrapper, S, vec_y), vec_y) # Sparse matrix is broken
        nest_vector!(S.y, opt[1])

        k_discrete = copy(cache.k_discrete)

        if opt[2] == ReturnCode.Success
            info = 0
        elseif opt[2] == ReturnCode.Failure
            info = 1
        end

        # Change the original solution Vector{Vector{Float64}} to a matrix
        # has size of (number of subintervals, length of equation)
        global Y = transpose(reduce(hcat, S.y))

        if info == 0
            defect, defect_norm, k_interp = defect_estimate(prob, Y, alg, n, dt, len, mesh,
                k_discrete)
            if defect_norm > defect_threshold
                info = 4
                println("Defect norm is ", defect_norm)
                println("Newton iteration was successful, but")
                println("the defect is greater than 10%, the solution is not acceptable")
            end
        end

        s, s_star = setup_coeff(alg)

        if info == 0
            println("Newton interation was successful")
            if defect_norm > tol
                mesh_new, Nsub_star, info = mesh_selector(mesh, defect, tol, n, len, alg)
                if info == 0
                    z, z_prime = zeros(len), zeros(len)
                    new_Y = zeros(Nsub_star + 1, len)
                    for i in 0:Nsub_star
                        z, z_prime = interp_eval(mesh, Y, alg, mesh_new[i + 1], dt, len, s,
                                                 s_star, k_discrete, k_interp)
                        new_Y[i + 1, :] = z
                    end
                    mesh = copy(mesh_new)
                    n = copy(Nsub_star)
                    Y = copy(new_Y)
                end
            end

            if (info == 0) && (defect_norm < tol)
                println("Succesful computation, the user defined tolerance has been satisfied")
            end
        else
            println("Cannot obtain a solution for the current mesh")
            if 2 * n > MxNsub
                println("New mesh would be too large")
                info = -1
            else
                mesh_new = half_mesh(mesh, n)
                Nsub_star = 2 * n
                mesh = copy(mesh_new)
                n = copy(Nsub_star)
                println("New mesh will be of size ", n) # Next computation would be based on length n mesh
                info = 0 # Force a restart
                defect_norm = 2 * tol
            end
        end
    end

    retcode = ReturnCode.Success
    # 
    if info == 0
        retcode = ReturnCode.Success
    elseif info == -4
        retcode = ReturnCode.Terminated
    else
        retcode = ReturnCode.Failure
    end
    DiffEqBase.build_solution(prob, alg, mesh, Y, retcode = retcode)
end

"""
    interp_eval(mesh, Y, t, k_discrete, k_interp)

After we construct an interpolant, we use interp_eval to evaluate it.
"""
function interp_eval(mesh, Y, alg, t, dt, len, s, s_star, k_discrete, k_interp)
    # EXPORTS: z, z_prime
    i = interval(mesh, t)
    hi = mesh[i + 1] - mesh[i]
    tau = (t - mesh[i]) / hi
    weights, weights_prime = interp_weights(tau, alg)
    z, z_prime = sum_stages(weights, weights_prime, k_discrete, k_interp, len, dt, Y, s, s_star)
    return z, z_prime
end

function interval(mesh, t)
    ind = findfirst(x -> x > t, mesh)
    i::Int64 = copy(ind)
    return i
end

"""
    mesh_selector(mesh_current, defect, tol, n, len, alg)

Generate new mesh based on the defect.
"""
function mesh_selector(mesh_current::Vector, defect, tol, n::Int64, len::Int64, alg::Union{GeneralMIRK, MIRK})
    #exports: mesh_new, Nsub_star, info

    #TODO: Need users to manually specify, here, we set it as 3000 by default.
    MxNsub = 3000

    safety_factor = 1.3
    rho = 1.0 # Set rho=1 means mesh distribution will take place everytime.
    upper_new_mesh = 4.0
    lower_new_mesh = 0.5
    r1 = 0.0
    r2 = 0.0
    Nsub_star = 0
    info = 0
    p = alg_order(alg)
    s_hat = zeros(Float64, n)
    for i in 1:n
        h = mesh_current[i + 1] - mesh_current[i]
        norm = abs(defect[i, idamax(defect[i, :])])
        s_hat[i] = (norm / tol)^(1.0 / (p + 1)) / h
        if s_hat[i] * h > r1
            r1 = s_hat[i] * h
        end
        r2 = r2 + s_hat[i] * h
    end
    r3 = r2 / n
    n_predict::Int64 = round(Int, (safety_factor * r2) + 1)
    if abs((n_predict - n) / n) < 0.1
        n_predict = round(Int, 1.1 * n)
    end

    if r1 <= rho * r3
        Nsub_star::Int64 = 2 * n
        if Nsub_star > MxNsub # Need to determine the too large threshold
            println("New mesh would be too large")
            info = -1
        else
            println("Half the current mesh")
            mesh_new = half_mesh(mesh_current, n)
        end
    else
        Nsub_star = copy(n_predict)
        if Nsub_star > upper_new_mesh * n
            Nsub_star = upper_new_mesh * n
        end
        if Nsub_star < lower_new_mesh * n
            Nsub_star = lower_new_mesh * n
        end
        if Nsub_star > MxNsub
            # Mesh redistribution fails
            println("New mesh would be too large")
            info = -1
        else
            println("Mesh redistributing")
            mesh_new = redistribute(mesh_current, n, Nsub_star, s_hat)
        end
    end
    return mesh_new, Nsub_star, info
end

"""
    redistribute(mesh_current, n, Nsub_star, s_hat)

Generate a new mesh.
"""
function redistribute(mesh_current::Vector,
    n::Int64,
    Nsub_star::Int64,
    s_hat::Vector{Float64})
    mesh_new = zeros(Float64, Nsub_star + 1)
    sum = 0.0
    for k in 1:n
        sum += s_hat[k] * (mesh_current[k + 1] - mesh_current[k])
    end
    zeta = sum / Nsub_star
    k::Int64 = 1
    i::Int64 = 0
    mesh_new[1] = mesh_current[1]
    t = mesh_current[1]
    integral::Float64 = 0.0
    while k <= n
        next_piece = s_hat[k] * (mesh_current[k + 1] - t)
        if (integral + next_piece) > zeta
            mesh_new[i + 2] = (zeta - integral) / s_hat[k] + t
            t = mesh_new[i + 2]
            i += 1
            integral = 0
        else
            integral += next_piece
            t = mesh_current[k + 1]
            k += 1
        end
    end
    mesh_new[Nsub_star + 1] = mesh_current[n + 1]
    return mesh_new
end

"""
    half_mesh(mesh_current, n)

The input mesh_current has length of n+1

Divide the original subinterval into two equal length subinterval.
"""
function half_mesh(mesh_current::Vector, n::Int64)
    mesh_new = zeros(Float64, 2 * n + 1)
    mesh_new[1] = mesh_current[1]
    for i in 1:n
        mesh_new[2 * i + 1] = mesh_current[i + 1]
        mesh_new[2 * i] = (mesh_current[i + 1] + mesh_current[i]) / 2.0
    end
    return mesh_new
end

function idamax(x)
    x = abs.(x)
    _, id = findmax(x)
    return id
end

"""
    defect_estimate(prob, Y, alg, n, dt, mesh, k_discrete)

defect_estimate use the discrete solution approximation Y, plus stages of 
the RK method in 'k_discrete', plus some new stages in 'k_interp' to construct 
an interpolant
"""
function defect_estimate(prob::BVProblem, Y, alg::Union{GeneralMIRK, MIRK}, n::Int64, dt, len::Int64, mesh::Vector, k_discrete)
    # Initialization
    defect = zeros(n, len)
    s, s_star, tau_star, x_star, v_star, c_star = setup_coeff(alg)

    f_sample_1, f_sample_2 = zeros(Float64, len), zeros(Float64, len)
    def_1, def_2 = zeros(Float64, len), zeros(Float64, len)
    temp_1, temp_2 = zeros(Float64, len), zeros(Float64, len)
    estimate_1, estimate_2 = zeros(Float64), zeros(Float64)

    # Evaluate at the first sample point
    weights_1, weights_1_prime = interp_weights(tau_star, alg)
    # Evaluate at the second sample point
    weights_2, weights_2_prime = interp_weights(1.0 - tau_star, alg)

    k_interp = zeros(Float64)
    for i in 1:n
        k_interp = interp_setup(mesh[i], dt, Y[i, :], Y[i + 1, :], s, s_star, x_star,
                                v_star, c_star, k_discrete[i, :], prob, len)

        # Sample point 1
        z, z_prime = sum_stages(weights_1, weights_1_prime, k_discrete, k_interp, len, dt,
                                Y, s, s_star)
        prob.f(f_sample_1, z, prob.p, mesh[i] + tau_star * dt)
        z_prime .= z_prime .- f_sample_1
        def_1 = copy(z_prime)
        for j in 1:len
            temp_1[j] = def_1[j] / (abs(f_sample_1[j]) + 1.0)
        end
        estimate_1 = maximum(abs.(temp_1))

        # Sample point 2
        z, z_prime = sum_stages(weights_2, weights_2_prime, k_discrete, k_interp, len, dt,
                                Y, s, s_star)
        prob.f(f_sample_2, z, prob.p, mesh[i] + (1.0 - tau_star) * dt)
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
function setup_coeff(alg::Union{GeneralMIRK, MIRK})
    if alg_order(alg) == 4
        tau_star = 0.226
        s = 3
        s_star = 4
        x_star = [3 / 64, -9 / 64, 0.0, 0.0]
        v_star = 27.0 / 32.0
        c_star = 3.0 / 4.0
        return s, s_star, tau_star, x_star, v_star, c_star
    elseif alg_order(alg) == 6
        tau_star = 0.7156
        s = 3
        s_star = 9
        c_star = [7 / 16, 3 / 8, 9 / 16, 1 / 8]
        v_star = [7 / 16, 3 / 8, 9 / 16, 1 / 8]
        x_star = [1547 / 32768, 83 / 1536, 1225 / 32768, 233 / 3456,
            -1225 / 32768, -13 / 384, -1547 / 32768, -19 / 1152,
            749 / 4096, 283 / 1536, 287 / 2048, 0,
            -287 / 2048, -167 / 1536, -749 / 4096, 0,
            -861 / 16384, -49 / 512, 861 / 16384, 0,
            0, 0, 0, -5 / 72,
            0, 0, 0, 7 / 72,
            0, 0, 0, -17 / 216,
            0, 0, 0, 0]
        return s, s_star, tau_star, x_star, v_star, c_star
    end
end

"""
    interp_setup

interp_setup prepare the extra stages in ki_interp for interpolant construction.
Here, the ki_interp is the stages in one subinterval.
"""
function interp_setup(tim1, dt, y_left, y_right, s, s_star, x_star, v_star, c_star,
                      ki_discrete, prob, len)
    # EXPORTS: ki_interp
    ki_interp = zeros(Float64, (s_star - s) * len)
    for r in 1:(s_star - s)
        new_stages = zeros(Float64, len)
        for j in 1:s
            new_stages .= new_stages .+
                          x_star[j * (s_star - s) + r] .*
                          ki_discrete[((j - 1) * len + 1):((j - 1) * len + len)]
        end
        for j in 1:(r - 1)
            new_stages .= new_stages .+
                          x_star[(j + s - 1) * (s_star - s) + r] .*
                          ki_interp[((j - 1) * len + 1):((j - 1) * len + len)]
        end
        new_stages .= new_stages .* dt
        new_stages .= new_stages .+ (1 - v_star[r]) .* y_left
        new_stages .= new_stages .+ v_star[r] .* y_right

        temp = copy(ki_interp[:, r])
        prob.f(temp, new_stages, prob.p, tim1 + c_star[r])
        ki_interp[:, r] = temp
    end
    return ki_interp
end

"""
    sum_stages(weights, weights_prime, ki_discrete, ki_interp, len, dt, y)

sum_stages add the discrete solution, RK method stages and extra stages to construct interpolant.

Here, ki_discrete is a matrix stored with discrete RK stages in the ith interval, ki_discrete has legnth of s*neqns
Here, ki_interp is a matrix stored with interpolation coefficients in the ith interval, ki_interp has length of (s_star-s)*neqns
"""
function sum_stages(weights, weights_prime, ki_discrete, ki_interp, len, dt, y, s, s_star)
    # EXPORTS: z, z_prime
    z, z_prime = zeros(len), zeros(len)
    ki_discrete = ki_discrete[:]
    for i in 1:s
        z .= z .+ weights[i] .* ki_discrete[((i - 1) * len + 1):((i - 1) * len + len)]
        z_prime .= z_prime .+
                   weights_prime[i] .*
                   ki_discrete[((i - 1) * len + 1):((i - 1) * len + len)]
    end
    for j in 1:(s_star - s)
        z .= z .+ weights[s + j] .* ki_interp[((j - 1) * len + 1):((j - 1) * len + len)]
        z_prime .= z_prime .+
                   weights_prime[j] .* ki_interp[((j - 1) * len + 1):((j - 1) * len + len)]
    end
    z = z .* dt
    z = z .* y
    return z, z_prime
end

"""
    interp_weights(tau, alg)

interp_weights: solver-specified interpolation weights and its first derivative
"""
function interp_weights(tau, alg::Union{GeneralMIRK, MIRK})
    if alg_order(alg) == 4
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

    elseif alg_order(alg) == 6
        w = [
            tau - 28607 / 7434 * tau^2 - 166210 / 33453 * tau^3 + 334780 / 11151 * tau^4 -
            1911296 / 55755 * tau^5 + 406528 / 33453 * tau^6,
            777 / 590 * tau^2 - 2534158 / 234171 * tau^3 + 2088580 / 78057 * tau^4 -
            10479104 / 390285 * tau^5 + 11328512 / 1170855 * tau^6,
            -1008 / 59 * tau^2 + 222176 / 1593 * tau^3 - 180032 / 531 * tau^4 +
            876544 / 2655 * tau^5 - 180224 / 1593 * tau^6,
            -1008 / 59 * tau^2 + 222176 / 1593 * tau^3 - 180032 / 531 * tau^4 +
            876544 / 2655 * tau^5 - 180224 / 1593 * tau^6,
            -378 / 59 * tau^2 + 27772 / 531 * tau^3 - 22504 / 177 * tau^4 +
            109568 / 885 * tau^5 - 22528 / 531 * tau^6,
            -95232 / 413 * tau^2 + 62384128 / 33453 * tau^3 - 49429504 / 11151 * tau^4 +
            46759936 / 11151 * tau^5 - 46661632 / 33453 * tau^6,
            896 / 5 * tau^2 - 4352 / 3 * tau^3 + 3456 * tau^4 - 16384 / 5 * tau^5 +
            16384 / 15 * tau^6,
            50176 / 531 * tau^2 - 179554304 / 234171 * tau^3 + 143363072 / 78057 * tau^4 -
            136675328 / 78057 * tau^5 + 137363456 / 234171 * tau^6,
            16384 / 441 * tau^3 - 16384 / 147 * tau^4 + 16384 / 147 * tau^5 -
            16384 / 441 * tau^6]

        #     Derivative polynomials.

        wp = [
            1 - 28607 / 3717 * tau - 166210 / 11151 * tau^2 + 1339120 / 11151 * tau^3 -
            1911296 / 11151 * tau^4 + 813056 / 11151 * tau^5,
            777 / 295 * tau - 2534158 / 78057 * tau^2 + 8354320 / 78057 * tau^3 -
            10479104 / 78057 * tau^4 + 22657024 / 390285 * tau^5,
            -2016 / 59 * tau + 222176 / 531 * tau^2 - 720128 / 531 * tau^3 +
            876544 / 531 * tau^4 - 360448 / 531 * tau^5,
            -2016 / 59 * tau + 222176 / 531 * tau^2 - 720128 / 531 * tau^3 +
            876544 / 531 * tau^4 - 360448 / 531 * tau^5,
            -756 / 59 * tau + 27772 / 177 * tau^2 - 90016 / 177 * tau^3 +
            109568 / 177 * tau^4 - 45056 / 177 * tau^5,
            -190464 / 413 * tau + 62384128 / 11151 * tau^2 - 197718016 / 11151 * tau^3 +
            233799680 / 11151 * tau^4 - 93323264 / 11151 * tau^5,
            1792 / 5 * tau - 4352 * tau^2 + 13824 * tau^3 - 16384 * tau^4 +
            32768 / 5 * tau^5,
            100352 / 531 * tau - 179554304 / 78057 * tau^2 + 573452288 / 78057 * tau^3 -
            683376640 / 78057 * tau^4 + 274726912 / 78057 * tau^5,
            16384 / 147 * tau^2 - 65536 / 147 * tau^3 + 81920 / 147 * tau^4 -
            32768 / 147 * tau^5]

        return w, wp
    end
end
