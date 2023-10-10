function SciMLBase.__solve(prob::BVProblem, alg::MultipleShooting; odesolve_kwargs = (;),
    nlsolve_kwargs = (;), ensemblealg = EnsembleThreads(), verbose = true, kwargs...)
    @unpack f, tspan = prob
    bc = prob.f.bc
    has_initial_guess = prob.u0 isa AbstractVector{<:AbstractArray}
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    _u0 = has_initial_guess ? first(prob.u0) : prob.u0
    N, u0_size, nshoots, iip = length(_u0), size(_u0), alg.nshoots, isinplace(prob)
    if prob.f.bcresid_prototype === nothing
        if prob.problem_type isa TwoPointBVProblem
            # This can only happen if the problem is !iip
            bcresid_prototype = ArrayPartition(bc[1](_u0, prob.p), bc[2](_u0, prob.p))
        else
            bcresid_prototype = similar(_u0)
        end
    else
        bcresid_prototype = prob.f.bcresid_prototype
    end

    if prob.problem_type isa TwoPointBVProblem
        resida_len = length(bcresid_prototype.x[1])
        residb_len = length(bcresid_prototype.x[2])
    end

    if has_initial_guess && length(prob.u0) != nshoots + 1
        nshoots = length(prob.u0) - 1
        verbose &&
            @warn "Initial guess length != `nshoots + 1`! Adapting to `nshoots = $(nshoots)`"
    end

    # We will use colored AD for this part!
    @views function solve_internal_odes!(resid_nodes, us, p, cur_nshoots, nodes)
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

        return reduce(vcat, ensemble_sol.u.us), reduce(vcat, ensemble_sol.u.ts)
    end

    compute_bc_residual! = if prob.problem_type isa TwoPointBVProblem
        @views function compute_bc_residual_tp!(resid_bc, us::ArrayPartition, p,
            cur_nshoots, nodes, resid_nodes::Union{Nothing, MaybeDiffCache} = nothing)
            ua, ub0 = us.x
            # Just Recompute the last ODE Solution
            lastodeprob = ODEProblem{iip}(f, reshape(ub0, u0_size),
                (nodes[end - 1], nodes[end]), p)
            sol_ode_last = solve(lastodeprob, alg.ode_alg; odesolve_kwargs..., verbose,
                kwargs..., save_everystep = false, saveat = (), save_end = true)
            ub = vec(sol_ode_last.u[end])

            resid_bc_a, resid_bc_b = if resid_bc isa ArrayPartition
                resid_bc.x
            else
                resid_bc[1:resida_len], resid_bc[(resida_len + 1):end]
            end

            if iip
                bc[1](resid_bc_a, ua, p)
                bc[2](resid_bc_b, ub, p)
            else
                resid_bc_a .= bc[1](ua, p)
                resid_bc_b .= bc[2](ub, p)
            end

            return resid_bc
        end
    else
        @views function compute_bc_residual_mp!(resid_bc, us, p, cur_nshoots, nodes,
            resid_nodes::Union{Nothing, MaybeDiffCache} = nothing)
            if resid_nodes === nothing
                _resid_nodes = similar(us, cur_nshoots * N)  # This might be Dual based on `us`
            else
                _resid_nodes = get_tmp(resid_nodes, us)
            end

            # NOTE: We need to recompute this to correctly propagate the dual numbers / gradients
            _us, _ts = solve_internal_odes!(_resid_nodes, us, p, cur_nshoots, nodes)

            # Boundary conditions
            # Builds an ODESolution object to keep the framework for bc(,,) consistent
            odeprob = ODEProblem{iip}(f, reshape(us[1:N], u0_size), tspan, p)
            total_solution = SciMLBase.build_solution(odeprob, alg.ode_alg, _ts, _us)

            if iip
                eval_bc_residual!(resid_bc, prob.problem_type, bc, total_solution, p)
            else
                resid_bc .= eval_bc_residual(prob.problem_type, bc, total_solution, p)
            end

            return resid_bc
        end
    end

    @views function loss!(resid::ArrayPartition, us, p, cur_nshoots, nodes)
        resid_bc, resid_nodes = resid.x[1], resid.x[2]

        _us, _ts = solve_internal_odes!(resid_nodes, us, p, cur_nshoots, nodes)

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

    jac! = if prob.problem_type isa TwoPointBVProblem
        @views function jac_tp!(J::AbstractMatrix, us, p, resid_bc,
            resid_nodes::MaybeDiffCache, ode_jac_cache, bc_jac_cache::Tuple, ode_fn, bc_fn,
            cur_nshoot, nodes)
            J isa SparseArrays.SparseMatrixCSC || fill!(J, 0)
            J_bc = J[1:N, :]
            J_c = J[(N + 1):end, :]

            sparse_jacobian!(J_c, alg.jac_alg.nonbc_diffmode, ode_jac_cache, ode_fn,
                resid_nodes.du, us)

            # For BC
            bc_jac_cache′, J_bc′ = bc_jac_cache
            sparse_jacobian!(J_bc′, alg.jac_alg.bc_diffmode, bc_jac_cache′, bc_fn,
                resid_bc, ArrayPartition(us[1:N], us[(end - N + 1):end]))
            resida, residb = resid_bc.x
            J_bc[1:length(resida), 1:N] .= J_bc′[1:length(resida), 1:N]
            idxᵢ = (length(resida) + 1):(length(resida) + length(residb))
            J_bc[idxᵢ, (end - N + 1):end] .= J_bc′[idxᵢ, (end - N + 1):end]

            return nothing
        end
    else
        @views function jac_mp!(J::AbstractMatrix, us, p, resid_bc,
            resid_nodes::MaybeDiffCache, ode_jac_cache, bc_jac_cache, ode_fn, bc_fn,
            cur_nshoot, nodes)
            J_bc = J[1:N, :]
            J_c = J[(N + 1):end, :]

            sparse_jacobian!(J_c, alg.jac_alg.nonbc_diffmode, ode_jac_cache, ode_fn,
                resid_nodes.du, us)

            # For BC
            sparse_jacobian!(J_bc, alg.jac_alg.bc_diffmode, bc_jac_cache, bc_fn, resid_bc,
                us)

            return nothing
        end
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
        resid_nodes = maybe_allocate_diffcache(resid_prototype.x[2],
            pickchunksize((cur_nshoot + 1) * N), alg.jac_alg.bc_diffmode)

        if prob.problem_type isa TwoPointBVProblem
            if alg.jac_alg.nonbc_diffmode isa AbstractSparseADType ||
               alg.jac_alg.bc_diffmode isa AbstractSparseADType
                J_full, col_colorvec, row_colorvec, (J_c, J_bc_partial), col_colorvec_bc, row_colorvec_bc, = __generate_sparse_jacobian_prototype(alg,
                    prob.problem_type, bcresid_prototype, _u0, N, cur_nshoot)
            end
        elseif alg.jac_alg.nonbc_diffmode isa AbstractSparseADType
            J_c, col_colorvec, row_colorvec, = __generate_sparse_jacobian_prototype(alg,
                prob.problem_type, bcresid_prototype, _u0, N, cur_nshoot)
        end

        ode_fn = (du, u) -> solve_internal_odes!(du, u, prob.p, cur_nshoot, nodes)
        sd_ode = if alg.jac_alg.nonbc_diffmode isa AbstractSparseADType
            PrecomputedJacobianColorvec(; jac_prototype = J_c, row_colorvec, col_colorvec)
        else
            NoSparsityDetection()
        end
        ode_jac_cache = sparse_jacobian_cache(alg.jac_alg.nonbc_diffmode, sd_ode,
            ode_fn, similar(u_at_nodes, cur_nshoot * N), u_at_nodes)

        bc_fn = (du, u) -> compute_bc_residual!(du, u, prob.p, cur_nshoot, nodes,
            resid_nodes)
        if prob.problem_type isa TwoPointBVProblem
            sd_bc = if alg.jac_alg.bc_diffmode isa AbstractSparseADType
                PrecomputedJacobianColorvec(; jac_prototype = J_bc_partial,
                    row_colorvec = row_colorvec_bc, col_colorvec = col_colorvec_bc)
            else
                NoSparsityDetection()
            end
            bc_jac_cache_partial = sparse_jacobian_cache(alg.jac_alg.bc_diffmode, sd_bc,
                bc_fn, similar(bcresid_prototype),
                ArrayPartition(@view(u_at_nodes[1:N]),
                    @view(u_at_nodes[(end - N + 1):end])))

            bc_jac_cache = (bc_jac_cache_partial, init_jacobian(bc_jac_cache_partial))

            jac_prototype = if alg.jac_alg.bc_diffmode isa AbstractSparseADType ||
                               alg.jac_alg.nonbc_diffmode isa AbstractSparseADType
                J_full
            else
                # Dense AD being used!
                fill!(similar(u_at_nodes, length(resid_prototype), length(u_at_nodes)), 0)
            end
        else
            sd_bc = alg.jac_alg.bc_diffmode isa AbstractSparseADType ?
                    SymbolicsSparsityDetection() : NoSparsityDetection()
            bc_jac_cache = sparse_jacobian_cache(alg.jac_alg.bc_diffmode,
                sd_bc, bc_fn, similar(bcresid_prototype), u_at_nodes)

            jac_prototype = vcat(init_jacobian(bc_jac_cache), init_jacobian(ode_jac_cache))
        end
        jac_fn = (J, us, p) -> jac!(J, us, p, similar(bcresid_prototype), resid_nodes,
            ode_jac_cache, bc_jac_cache, ode_fn, bc_fn, cur_nshoot, nodes)

        loss_function! = NonlinearFunction{true}((args...) -> loss!(args..., cur_nshoot,
                nodes); resid_prototype, jac = jac_fn, jac_prototype)
        nlprob = NonlinearProblem(loss_function!, u_at_nodes, prob.p)
        sol_nlsolve = solve(nlprob, alg.nlsolve; nlsolve_kwargs..., verbose, kwargs...)
        u_at_nodes = sol_nlsolve.u
    end

    single_shooting_prob = remake(prob; u0 = reshape(u_at_nodes[1:N], u0_size))
    return solve(single_shooting_prob, Shooting(alg.ode_alg; alg.nlsolve); odesolve_kwargs,
        nlsolve_kwargs, verbose, kwargs...)
end

@views function multiple_shooting_initialize(prob, alg::MultipleShooting, has_initial_guess,
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

@views function multiple_shooting_initialize(u_at_nodes_prev, prob, alg, prev_nodes,
    nshoots, old_nshoots, has_initial_guess; odesolve_kwargs = (;), kwargs...)
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

"""
    __generate_sparse_jacobian_prototype(::MultipleShooting, _, _, u0, N::Int,
        nshoots::Int)
    __generate_sparse_jacobian_prototype(::MultipleShooting, ::TwoPointBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int)

For a Multi-Point Problem, returns the Jacobian Prototype for the Sparse Part. For a Two-
Point Problem, returns the Jacobian Prototype for the Entire Jacobian.

Also returns the column and row color vectors for the Sparse Non-BC Part Jacobian.

Returns the column and row color vectors for the Sparse BC Part Jacobian (if computed).

Also returns the indices `Is` and `Js` used to construct the Sparse Jacobian.
"""
function __generate_sparse_jacobian_prototype(::MultipleShooting, _, _, u0, N::Int,
    nshoots::Int)
    # Sparse for Stitching solution together
    Is = Vector{Int64}(undef, (N^2 + N) * nshoots)
    Js = Vector{Int64}(undef, (N^2 + N) * nshoots)

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

    return J_c, col_colorvec, row_colorvec, (J_c, nothing), nothing, nothing, Is, Js
end

function __generate_sparse_jacobian_prototype(alg::MultipleShooting, ::TwoPointBVProblem,
    bcresid_prototype, u0, N::Int, nshoots::Int)
    resida, residb = bcresid_prototype.x
    # Sparse for Stitching solution together
    L = N * length(resida) + (N^2 + N) * nshoots + N * length(residb)
    Is = Vector{Int64}(undef, L)
    Js = Vector{Int64}(undef, L)

    idx = 1
    for row in 1:length(resida)
        for j in 1:N
            Is[idx] = row
            Js[idx] = j
            idx += 1
        end
    end
    for row in 1:length(residb)
        for j in 1:N
            Is[idx] = length(resida) + row
            Js[idx] = j + (nshoots * N)
            idx += 1
        end
    end
    J_c, col_colorvec, row_colorvec, _, _, _, Is′, Js′ = __generate_sparse_jacobian_prototype(alg,
        nothing, nothing, u0, N, nshoots)
    for (i, j) in zip(Is′, Js′)
        Is[idx] = length(resida) + length(residb) + i
        Js[idx] = j
        idx += 1
    end

    col_colorvec_bc = Vector{Int}(undef, 2N)
    row_colorvec_bc = Vector{Int}(undef, length(resida) + length(residb))
    col_colorvec_bc[1:N] .= 1:N
    col_colorvec_bc[(N + 1):end] .= 1:N
    for i in 1:max(length(resida), length(residb))
        if i ≤ length(resida)
            row_colorvec_bc[i] = i
        end
        if i ≤ length(residb)
            row_colorvec_bc[i + length(resida)] = i
        end
    end

    J = sparse(adapt(parameterless_type(u0), Is), adapt(parameterless_type(u0), Js),
        similar(u0, length(Is)))

    Is_bc = Vector{Int64}(undef, N^2)
    Js_bc = Vector{Int64}(undef, N^2)
    idx = 1
    for i in 1:length(resida)
        for j in 1:N
            Is_bc[idx] = i
            Js_bc[idx] = j
            idx += 1
        end
    end
    for i in 1:length(residb)
        for j in 1:N
            Is_bc[idx] = i + length(resida)
            Js_bc[idx] = j + N
            idx += 1
        end
    end

    J_bc = sparse(adapt(parameterless_type(u0), Is_bc),
        adapt(parameterless_type(u0), Js_bc),
        similar(u0, length(Is_bc)))

    return (J, col_colorvec, row_colorvec, (J_c, J_bc), col_colorvec_bc, row_colorvec_bc,
        Is, Js)
end
