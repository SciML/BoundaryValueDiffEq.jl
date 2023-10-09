function SciMLBase.__solve(prob::BVProblem, alg::MultipleShooting; odesolve_kwargs = (;),
    nlsolve_kwargs = (;), ensemblealg = EnsembleThreads(), verbose = true, kwargs...)
    @unpack f, tspan = prob
    bc = prob.f.bc
    has_initial_guess = prob.u0 isa AbstractVector{<:AbstractArray}
    _u0 = has_initial_guess ? first(prob.u0) : prob.u0
    N, u0_size, nshoots, iip = length(_u0), size(_u0), alg.nshoots, isinplace(prob)
    bcresid_prototype = prob.f.bcresid_prototype === nothing ? similar(_u0) :
                        prob.f.bcresid_prototype

    if has_initial_guess && length(prob.u0) != nshoots + 1
        nshoots = length(prob.u0) - 1
        verbose &&
            @warn "Initial guess length != `nshoots + 1`! Adapting to `nshoots = $(nshoots)`"
    end

    @views function loss!(resid::ArrayPartition, us, p, cur_nshoots, nodes)
        ts_ = Vector{Vector{typeof(first(tspan))}}(undef, cur_nshoots)
        us_ = Vector{Vector{typeof(us)}}(undef, cur_nshoots)

        resid_bc, resid_nodes = resid.x[1], resid.x[2]

        function prob_func(probᵢ, i, repeat)
            return remake(probᵢ; u0 = reshape(us[((i - 1) * N + 1):(i * N)], u0_size),
                tspan = (nodes[i], nodes[i + 1]))
        end

        function reduction(u, data, I)
            for i in I
                u.us[i] = data[i].u
                u.ts[i] = data[i].t
                u.resid[((i - 1) * N + 1):(i * N)] .= vec(us[(i * N + 1):((i + 1) * N)]) .-
                                                      vec(data[i].u[end])
            end
            return (u, false)
        end

        odeprob = ODEProblem{iip}(f, reshape(us[1:N], u0_size), tspan, p)

        ensemble_prob = EnsembleProblem(odeprob; prob_func, reduction, safetycopy = false,
            u_init = (; us = us_, ts = ts_, resid = resid_nodes))
        ensemble_sol = solve(ensemble_prob, alg.ode_alg, ensemblealg; odesolve_kwargs...,
            verbose, kwargs..., save_end = true, save_everystep = false,
            trajectories = cur_nshoots)

        _us = reduce(vcat, ensemble_sol.u.us)
        _ts = reduce(vcat, ensemble_sol.u.ts)

        # Boundary conditions
        # Builds an ODESolution object to keep the framework for bc(,,) consistent
        total_solution = SciMLBase.build_solution(odeprob, alg.ode_alg, _ts, _us)

        if iip
            eval_bc_residual!(resid_bc, prob.problem_type, bc, total_solution, p)
        else
            resid_bc .= eval_bc_residual(prob.problem_type, bc, total_solution, p)
        end

        return resid
    end

    @views function jac!(J::AbstractMatrix, us, p, cur_nshoots, nodes, resid_bc)
        J_bc = J[1:N, :]
        J_c = J[(N + 1):end, :]

        # Threads.@threads :static
        # FIXME: Threading here leads to segfaults
        for i in 1:cur_nshoots
            uᵢ = us[((i - 1) * N + 1):(i * N)]
            idx = ((i - 1) * N + 1):(i * N)
            probᵢ = ODEProblem{iip}(f, reshape(uᵢ, u0_size), (nodes[i], nodes[i + 1]), p)
            function solve_func(u₀)
                sJ = solve(probᵢ, alg.ode_alg; u0 = u₀, odesolve_kwargs...,
                    kwargs..., save_end = true, save_everystep = false, saveat = ())
                return -last(sJ)
            end
            # @show sum(J_c[idx, idx]), sum(J_c[idx, idx .+ N])
            ForwardDiff.jacobian!(J_c[idx, idx], solve_func, uᵢ)
            J_c′ = J_c[idx, idx .+ N]
            J_c′[diagind(J_c′)] .= 1
            # @show sum(J_c[idx, idx]), sum(J_c[idx, idx .+ N])
        end

        function evaluate_boundary_condition(us)
            ts_ = Vector{Vector{typeof(first(tspan))}}(undef, cur_nshoots)
            us_ = Vector{Vector{typeof(us)}}(undef, cur_nshoots)

            function prob_func(probᵢ, i, repeat)
                return remake(probᵢ; u0 = reshape(us[((i - 1) * N + 1):(i * N)], u0_size),
                    tspan = (nodes[i], nodes[i + 1]))
            end

            function reduction(u, data, I)
                for i in I
                    u.us[i] = data[i].u
                    u.ts[i] = data[i].t
                end
                return (u, false)
            end

            odeprob = ODEProblem{iip}(f, reshape(us[1:N], u0_size), tspan, p)

            ensemble_prob = EnsembleProblem(odeprob; prob_func, reduction,
                safetycopy = false, u_init = (; us = us_, ts = ts_))
            ensemble_sol = solve(ensemble_prob, alg.ode_alg, ensemblealg;
                odesolve_kwargs..., kwargs..., save_end = true, save_everystep = false,
                trajectories = cur_nshoots)

            _us = reduce(vcat, ensemble_sol.u.us)
            _ts = reduce(vcat, ensemble_sol.u.ts)

            # Boundary conditions
            # Builds an ODESolution object to keep the framework for bc(,,) consistent
            total_solution = SciMLBase.build_solution(odeprob, alg.ode_alg, _ts, _us)

            if iip
                _resid_bc = get_tmp(resid_bc, us)
                eval_bc_residual!(_resid_bc, prob.problem_type, bc, total_solution, p)
                return _resid_bc
            else
                return eval_bc_residual(prob.problem_type, bc, total_solution, p)
            end
        end

        ForwardDiff.jacobian!(J_bc, evaluate_boundary_condition, us)

        return nothing
    end

    # This gets all the nshoots except the final SingleShooting case
    all_nshoots = get_all_nshoots(alg.grid_coarsening, nshoots)
    u_at_nodes, nodes = nothing, nothing

    for (i, cur_nshoot) in enumerate(all_nshoots)
        if i == 1
            nodes, u_at_nodes = multiple_shooting_initialize(prob, alg, has_initial_guess,
                nshoots; odesolve_kwargs, verbose, kwargs...)
        else
            nodes, u_at_nodes = multiple_shooting_initialize(u_at_nodes, prob, alg, nodes,
                cur_nshoot, all_nshoots[i - 1], has_initial_guess; odesolve_kwargs, verbose,
                kwargs...)
        end

        resid_prototype = ArrayPartition(bcresid_prototype,
            similar(u_at_nodes, cur_nshoot * N))
        residbc_prototype = DiffCache(bcresid_prototype,
            pickchunksize((cur_nshoot + 1) * N))

        J_bc = similar(bcresid_prototype, length(bcresid_prototype), N * (cur_nshoot + 1))
        J_c, col_colorvec, row_colorvec = __generate_sparse_jacobian_prototype(alg, _u0, N,
            cur_nshoot)
        jac_prototype = vcat(J_bc, J_c)

        loss_function! = NonlinearFunction{true}((args...) -> loss!(args..., cur_nshoot,
                nodes); resid_prototype,
            jac = (args...) -> jac!(args..., cur_nshoot, nodes, residbc_prototype),
            jac_prototype)
        nlprob = NonlinearProblem(loss_function!, u_at_nodes, prob.p)
        sol_nlsolve = solve(nlprob, alg.nlsolve; nlsolve_kwargs..., verbose, kwargs...)
        u_at_nodes = sol_nlsolve.u
    end

    single_shooting_prob = remake(prob; u0 = reshape(u_at_nodes[1:N], u0_size))
    return SciMLBase.__solve(single_shooting_prob, Shooting(alg.ode_alg; alg.nlsolve);
        odesolve_kwargs, nlsolve_kwargs, verbose, kwargs...)
end

function multiple_shooting_initialize(prob, alg::MultipleShooting, has_initial_guess,
    nshoots; odesolve_kwargs = (;), verbose = true, kwargs...)
    @unpack f, u0, tspan, p = prob
    @unpack ode_alg = alg

    nodes = range(tspan[1], tspan[2]; length = nshoots + 1)
    N = has_initial_guess ? length(first(u0)) : length(u0)

    if has_initial_guess
        u_at_nodes = similar(first(u0), (nshoots + 1) * N)
        recursive_flatten!(u_at_nodes, u0)
        return nodes, u_at_nodes
    end

    # Ensures type stability in case the parameters are dual numbers
    if !(typeof(p) <: SciMLBase.NullParameters)
        if !isconcretetype(eltype(p)) && verbose
            @warn "Type inference will fail if eltype(p) is not a concrete type"
        end
        u_at_nodes = similar(u0, promote_type(eltype(u0), eltype(p)), (nshoots + 1) * N)
    else
        u_at_nodes = similar(u0, (nshoots + 1) * N)
    end

    # Assumes no initial guess for now
    start_prob = ODEProblem{isinplace(prob)}(f, u0, tspan, p)
    sol = solve(start_prob, ode_alg; odesolve_kwargs..., verbose, kwargs..., saveat = nodes)

    if SciMLBase.successful_retcode(sol)
        u_at_nodes[1:N] .= sol.u[1]
        for i in 2:(nshoots + 1)
            u_at_nodes[(N + (i - 2) * N) .+ (1:N)] .= sol.u[i]
        end
    else
        @warn "Initialization using odesolve failed. Initializing using 0s. It is \
               recommended to provide an `initial_guess` function in this case."
        fill!(u_at_nodes, 0)
    end

    return nodes, u_at_nodes
end

@views @inline function multiple_shooting_initialize(u_at_nodes_prev, prob, alg,
    prev_nodes, nshoots, old_nshoots, has_initial_guess; odesolve_kwargs = (;), kwargs...)
    @unpack f, u0, tspan, p = prob
    nodes = range(tspan[1], tspan[2]; length = nshoots + 1)
    N = has_initial_guess ? length(first(u0)) : length(u0)

    u_at_nodes = similar(u_at_nodes_prev, N + nshoots * N)
    u_at_nodes[1:N] .= u_at_nodes_prev[1:N]
    u_at_nodes[(end - N + 1):end] .= u_at_nodes_prev[(end - N + 1):end]

    skipsize = old_nshoots / nshoots
    for i in 2:nshoots
        pos = skipsize * (i - 1) + 1
        idxs = (N + (i - 2) * N) .+ (1:N)
        if isinteger(pos)
            # If the current node is also a node of the finer grid
            ind = trunc(Int, pos)
            idxs_prev = (N + (ind - 2) * N .+ (1:N))
            u_at_nodes[idxs] .= u_at_nodes_prev[idxs_prev]
        else
            # If the current node is not a node of the finer grid simulate from closest
            # previous node and take result from simulation
            fpos = floor(Int, pos)
            r = pos - fpos

            t0 = prev_nodes[fpos]
            tf = prev_nodes[fpos + 1]
            tstop = t0 + r * (tf - t0)

            idxs_prev = (N + (fpos - 2) * N .+ (1:N))
            ustart = u_at_nodes_prev[idxs_prev]

            odeprob = ODEProblem(f, ustart, (t0, tstop), p)
            odesol = solve(odeprob, alg.ode_alg; odesolve_kwargs..., kwargs..., saveat = (),
                save_end = true)

            u_at_nodes[idxs] .= odesol.u[end]
        end
    end

    return nodes, u_at_nodes
end

@inline function get_all_nshoots(grid_coarsening, nshoots)
    if grid_coarsening isa Bool
        !grid_coarsening && return [nshoots]
        update_fn = Base.Fix2(÷, 2)
    elseif grid_coarsening isa Function
        update_fn = grid_coarsening
    else
        grid_coarsening[1] == nshoots && return grid_coarsening
        return vcat(nshoots, grid_coarsening)
    end
    nshoots_vec = Int[nshoots]
    next = update_fn(nshoots)
    while next > 1
        push!(nshoots_vec, next)
        next = update_fn(last(nshoots_vec))
    end
    @assert !(1 in nshoots_vec)
    return nshoots_vec
end

function __generate_sparse_jacobian_prototype(::MultipleShooting, u0, N::Int, nshoots::Int)
    # Sparse for Stitching solution together
    Is = Vector{UInt32}(undef, (N^2 + N) * nshoots)
    Js = Vector{UInt32}(undef, (N^2 + N) * nshoots)

    idx = 1
    for i in 1:nshoots
        for (i₁, i₂) in Iterators.product(1:N, 1:N)
            Is[idx] = i₁ + ((i - 1) * N)
            Js[idx] = i₂ + ((i - 1) * N)
            idx += 1
        end
        Is[idx:(idx + N - 1)] .= (1:N) .+ ((i - 1) * N)
        Js[idx:(idx + N - 1)] .= (1:N) .+ (i * N)
        idx += N
    end

    J_c = sparse(adapt(parameterless_type(u0), Is), adapt(parameterless_type(u0), Js),
        similar(u0, length(Is)))

    col_colorvec = Vector{Int}(undef, N * (nshoots + 1))
    for i in eachindex(col_colorvec)
        col_colorvec[i] = mod1(i, 2 * N)
    end
    row_colorvec = Vector{Int}(undef, N * nshoots)
    for i in eachindex(row_colorvec)
        row_colorvec[i] = mod1(i, 2 * N)
    end

    return J_c, col_colorvec, row_colorvec
end
