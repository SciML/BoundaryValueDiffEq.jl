# TODO: incorporate `initial_guess` similar to MIRK methods
function SciMLBase.__solve(prob::BVProblem, alg::MultipleShooting; odesolve_kwargs = (;),
    nlsolve_kwargs = (;), kwargs...)
    @unpack f, bc, tspan = prob
    bcresid_prototype = prob.f.bcresid_prototype === nothing ? similar(prob.u0) :
                        prob.f.bcresid_prototype
    N, u0_size, nshoots, iip = length(prob.u0), size(prob.u0), alg.nshoots, isinplace(prob)

    @views function loss!(resid::ArrayPartition, us, p, cur_nshoots, nodes)
        ts_ = Vector{Vector{typeof(first(tspan))}}(undef, cur_nshoots)
        us_ = Vector{Vector{typeof(us)}}(undef, cur_nshoots)

        resid_bc, resid_nodes = resid.x[1], resid.x[2]

        for i in 1:cur_nshoots
            local odeprob = ODEProblem{iip}(f,
                reshape(us[((i - 1) * N + 1):(i * N)], u0_size), (nodes[i], nodes[i + 1]),
                prob.p)
            sol = solve(odeprob, alg.ode_alg; odesolve_kwargs..., kwargs...,
                save_end = true, save_everystep = false)

            ts_[i] = sol.t
            us_[i] = sol.u

            resid_nodes[((i - 1) * N + 1):(i * N)] .= vec(us[(i * N + 1):((i + 1) * N)]) .-
                                                      vec(sol.u[end])
        end

        _ts = foldl(vcat, ts_)
        _us = foldl(vcat, us_)

        # Boundary conditions
        # Builds an ODESolution object to keep the framework for bc(,,) consistent
        odeprob = ODEProblem{iip}(f, reshape(us[1:N], u0_size), tspan, p)
        total_solution = SciMLBase.build_solution(odeprob, alg.ode_alg, _ts, _us)

        if iip
            eval_bc_residual!(resid_bc, prob.problem_type, bc, total_solution, p)
        else
            resid_bc .= eval_bc_residual(prob.problem_type, bc, total_solution, p)
        end

        return resid
    end

    # This gets all the nshoots except the final SingleShooting case
    all_nshoots = get_all_nshoots(alg)
    u_at_nodes, nodes = nothing, nothing

    for (i, cur_nshoot) in enumerate(all_nshoots)
        if i == 1
            nodes, u_at_nodes = multiple_shooting_initialize(prob, alg; odesolve_kwargs,
                kwargs...)
        else
            nodes, u_at_nodes = multiple_shooting_initialize(u_at_nodes, prob, alg, nodes,
                cur_nshoot, all_nshoots[i - 1]; odesolve_kwargs, kwargs...)
        end

        resid_prototype = ArrayPartition(bcresid_prototype,
            similar(u_at_nodes, cur_nshoot * N))
        loss_function! = NonlinearFunction{true}((args...) -> loss!(args...,
                cur_nshoot, nodes); resid_prototype)
        nlprob = NonlinearProblem(loss_function!, u_at_nodes, prob.p)
        sol_nlsolve = solve(nlprob, alg.nlsolve; nlsolve_kwargs..., kwargs...)
        u_at_nodes = sol_nlsolve.u
    end

    single_shooting_prob = remake(prob; u0 = reshape(u_at_nodes[1:N], u0_size))
    return SciMLBase.__solve(single_shooting_prob, Shooting(alg.ode_alg; alg.nlsolve);
        odesolve_kwargs, nlsolve_kwargs, kwargs...)
end

function multiple_shooting_initialize(prob, alg::MultipleShooting; odesolve_kwargs = (;),
    kwargs...)
    @unpack f, bc, u0, tspan, p = prob
    @unpack ode_alg, nshoots = alg

    N = length(u0)
    nodes = range(tspan[1], tspan[2]; length = nshoots + 1)

    # Ensures type stability in case the parameters are dual numbers
    if !(typeof(p) <: SciMLBase.NullParameters)
        if !isconcretetype(eltype(p))
            @warn "Type inference will fail if eltype(p) is not a concrete type"
        end
        u_at_nodes = similar(u0, promote_type(eltype(u0), eltype(p)), (nshoots + 1) * N)
    else
        u_at_nodes = similar(u0, (nshoots + 1) * N)
    end

    # Assumes no initial guess for now
    start_prob = ODEProblem{isinplace(prob)}(f, u0, tspan, p)
    sol = solve(start_prob, ode_alg; odesolve_kwargs..., kwargs..., saveat = nodes)

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
    prev_nodes, nshoots, old_nshoots; odesolve_kwargs = (;), kwargs...)
    @unpack f, bc, u0, tspan, p = prob
    nodes = range(tspan[1], tspan[2]; length = nshoots + 1)
    N = length(u0)

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

@inline function get_all_nshoots(alg::MultipleShooting)
    @unpack nshoots, grid_coarsening = alg
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
