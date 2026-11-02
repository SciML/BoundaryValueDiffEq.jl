# For Multiple Shooting
"""
    __generate_sparse_jacobian_prototype(
        ::MultipleShooting, ::StandardBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int
    )
    __generate_sparse_jacobian_prototype(
        ::MultipleShooting, ::TwoPointBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int
    )

Generate a prototype of the sparse Jacobian matrix for the BVP problem.
"""
function __generate_sparse_jacobian_prototype(
        ::MultipleShooting, ::StandardBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int
    )
    fast_scalar_indexing(u0) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J₁ = nshoots * N
    J₂ = (nshoots + 1) * N
    J = BandedMatrix(Ones{eltype(u0)}(J₁, J₂), (N - 1, N + 1))

    return J
end

function __generate_sparse_jacobian_prototype(
        ::MultipleShooting, ::TwoPointBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int
    )
    fast_scalar_indexing(u0) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")

    resida, residb = bcresid_prototype
    L₁, L₂ = length(resida), length(residb)

    J₁ = L₁ + L₂ + nshoots * N
    J₂ = (nshoots + 1) * N

    # FIXME: There is a stronger structure than BandedMatrix here.
    #        We should be able to use that particular structure.
    J = BandedMatrix(Ones{eltype(u0)}(J₁, J₂), (max(L₁, L₂) + N - 1, N + 1))

    # for underdetermined systems we don't have banded qr implemented. use sparse
    J₁ < J₂ && return sparse(J)
    return J
end

# Sparsity/coloring metadata is prepared once on the host. Residuals, seeds,
# differentiated integrations, decompression, and matrix values stay on device.
__shooting_host(p::AbstractArray) = Array(p)
__shooting_host(p::Union{Tuple, NamedTuple}) = map(__shooting_host, p)
__shooting_host(p) = p

function __shooting_trace_pattern(f!, nr, nu, T)
    try
        return ADTypes.jacobian_sparsity(f!, zeros(T, nr), zeros(T, nu), TracerSparsityDetector())
    catch err
        err isa InterruptException && rethrow()
        # Global tracing rejects value-dependent branches. Fall back to a safe
        # full pattern instead of observing only the branch at the initial guess.
        return sparse(repeat(1:nr, nu), repeat(1:nu; inner = nr), trues(nr * nu), nr, nu)
    end
end

function __shooting_flow_pattern(prob, n, T)
    p = __shooting_host(prob.p)
    # Trace time as well as the state: sampling f at one time would miss a
    # coupling activated by a later time-dependent branch.
    rhs! = (r, z) -> __shooting_eval!(r, prob.f.f, (view(z, 1:n), p, z[n + 1]), Val(isinplace(prob)))
    graph = __shooting_trace_pattern(rhs!, n, n + 1, T)[:, 1:n]
    rows, cols = Int[], Int[]
    # An integrated state depends on all states reachable through RHS couplings,
    # even when the instantaneous RHS Jacobian is only diagonal/banded.
    for column in 1:n
        seen = falses(n)
        seen[column] = true
        queue = [column]
        cursor = 1
        while cursor <= length(queue) && length(queue) < n
            node = queue[cursor]
            for k in SparseArrays.nzrange(graph, node)
                row = SparseArrays.rowvals(graph)[k]
                if !seen[row]
                    push!(queue, row)
                    seen[row] = true
                end
            end
            cursor += 1
        end
        append!(rows, queue)
        append!(cols, fill(column, length(queue)))
    end
    return sparse(rows, cols, trues(length(rows)), n, n)
end

function __shooting_boundary_pattern(prob, host_mesh, cache, u)
    (; n, intervals, na, nb, twopoint) = cache
    nr, nu = na + n * intervals + nb, length(u)
    if twopoint
        rows, cols = Int[], Int[]
        p = __shooting_host(prob.p)
        for (bc, count, roffset, coffset) in ((cache.bc[1], na, 0, 0), (cache.bc[2], nb, nr - nb, nu - n))
            boundary! = (r, x) -> __shooting_eval!(r, bc, (x, p), cache.iip)
            pattern = __shooting_trace_pattern(boundary!, count, n, eltype(u))
            local_rows, local_cols, _ = SparseArrays.findnz(pattern)
            append!(rows, local_rows .+ roffset)
            append!(cols, local_cols .+ coffset)
        end
        return sparse(rows, cols, trues(length(rows)), nr, nu)
    end
    host_p = __shooting_host(prob.p)
    function boundary!(r, x)
        d = Matrix{eltype(x)}(undef, n, intervals + 1)
        for i in 1:(intervals + 1)
            __shooting_eval!(view(d, :, i), cache.f, (view(x, ((i - 1) * n + 1):(i * n)), host_p, host_mesh[i]), cache.iip)
        end
        __shooting_eval!(r, cache.bc, (ShootingDeviceEvalSol(x, d, host_mesh, n), host_p, host_mesh), cache.iip)
        return nothing
    end
    pattern = __shooting_trace_pattern(boundary!, na, nu, eltype(u))
    rows, cols, _ = SparseArrays.findnz(pattern)
    return sparse(rows, cols, trues(length(rows)), nr, nu)
end

function __shooting_mode(mode)
    dense = get_dense_ad(mode)
    dense === nothing && return AutoForwardDiff()
    dense isa Union{AutoForwardDiff, AutoFiniteDiff} || throw(ArgumentError("Device shooting supports AutoForwardDiff and AutoFiniteDiff, optionally wrapped in AutoSparse."))
    if dense isa AutoFiniteDiff
        dense.fdjtype isa Union{Val{:forward}, Val{:central}} || throw(ArgumentError("Device shooting finite differences support forward and central differences."))
    end
    return dense
end
function __shooting_group(pattern, selected, rows, cols, mode, cache, u, nr)
    subpattern = pattern[selected, :]
    coloring = __default_coloring_algorithm(mode)
    coloring isa ADTypes.NoColoringAlgorithm && (coloring = __default_coloring_algorithm(nothing))
    colors = collect(Int, ADTypes.column_coloring(subpattern, coloring))
    for j in eachindex(colors)
        isempty(SparseArrays.nzrange(subpattern, j)) && (colors[j] = 0)
    end
    # Validate custom colorings before extracting derivatives with shared seeds.
    transposed = sparse(transpose(subpattern))
    seen = zeros(Int, maximum(colors; init = 0))
    for row in axes(subpattern, 1), k in SparseArrays.nzrange(transposed, row)
        col = SparseArrays.rowvals(transposed)[k]
        color = colors[col]
        color > 0 && seen[color] != row || throw(ArgumentError("Invalid shooting column coloring."))
        seen[color] = row
    end
    active = sort!(unique(filter(>(0), colors)))
    mapping = Dict(c => i for (i, c) in enumerate(active))
    colors = [c == 0 ? 0 : mapping[c] for c in colors]
    entries = findall(i -> rows[i] in selected, eachindex(rows))
    dense = __shooting_mode(mode)
    chunk = dense isa AutoForwardDiff ? __shooting_chunk(dense, length(active)) : 1
    buffers = __shooting_group_buffers(
        dense, Val(chunk), cache, u, nr,
        __shooting_copy(cache.platform, colors), length(active),
        __shooting_copy(cache.platform, entries),
        __shooting_copy(cache.platform, rows[entries]), __shooting_copy(cache.platform, cols[entries])
    )
    return (; buffers..., selection = selected isa UnitRange ? :continuity : :boundary)
end
__shooting_chunk(::AutoForwardDiff{C}, n) where {C} = C === nothing ? min(max(n, 1), 8) : min(max(n, 1), C)
function __shooting_group_buffers(mode::AutoForwardDiff, chunk::Val{C}, cache, u, nr, colors, ncolors, entries, rows, cols) where {C}
    C > 0 || throw(ArgumentError("ForwardDiff chunk size must be positive."))
    tag = mode.tag === nothing ? typeof(ForwardDiff.Tag(__shooting_residual!, eltype(u))) : typeof(mode.tag)
    D = ForwardDiff.Dual{tag, eltype(u), C}
    work = __shooting_buffers(u, cache, nr, D)
    return (; mode, chunk, colors, ncolors, entries, rows, cols, work)
end
function __shooting_group_buffers(mode::AutoFiniteDiff, chunk, cache, u, nr, colors, ncolors, entries, rows, cols)
    work = __shooting_buffers(u, cache, nr, eltype(u))
    return (; mode, chunk, colors, ncolors, entries, rows, cols, work, minus = similar(u, nr))
end

function __shooting_jacobian_plan(prob, alg, u, host_mesh, cache, work)
    (; n, intervals, na, nb) = cache
    nr, nu = length(work.residual), length(u)
    rows, cols = Int[], Int[]
    flow = __shooting_flow_pattern(prob, n, eltype(u))
    flow_rows, flow_cols, _ = SparseArrays.findnz(flow)
    for i in 1:intervals
        append!(rows, flow_rows .+ (na + (i - 1) * n))
        append!(cols, flow_cols .+ ((i - 1) * n))
        for j in 1:n
            push!(rows, na + (i - 1) * n + j)
            push!(cols, i * n + j)
        end
    end
    boundary = __shooting_boundary_pattern(prob, host_mesh, cache, u)
    pattern = sparse(rows, cols, trues(length(rows)), nr, nu) .| boundary
    storage = __device_sparse_matrix(u, pattern)
    # Separate the boundary border so dense/nonlocal BCs never increase the
    # number of differentiated integrations for the continuity band.
    ode_mode = cache.twopoint ? alg.jac_alg.diffmode : alg.jac_alg.nonbc_diffmode
    bc_mode = cache.twopoint ? alg.jac_alg.diffmode : alg.jac_alg.bc_diffmode
    groups = (
        __shooting_group(pattern, (na + 1):(nr - nb), storage.rows, storage.cols, ode_mode, cache, u, nr),
        __shooting_group(pattern, vcat(1:na, (nr - nb + 1):nr), storage.rows, storage.cols, bc_mode, cache, u, nr),
    )
    return (; matrix = storage.matrix, groups, flow, rows = storage.rows, cols = storage.cols)
end

@kernel function __shooting_seed_kernel!(dual, u, colors, firstcolor, ::Val{C}) where {C}
    i = @index(Global, Linear)
    @inbounds dual[i] = eltype(dual)(u[i], ForwardDiff.Partials(ntuple(j -> colors[i] == firstcolor + j - 1 ? one(eltype(u)) : zero(eltype(u)), Val(C))))
end
@kernel function __shooting_extract_kernel!(values, residual, entries, rows, cols, colors, firstcolor, ::Val{C}) where {C}
    i = @index(Global, Linear)
    @inbounds part = colors[cols[i]] - firstcolor + 1
    if 1 <= part <= C
        @inbounds values[entries[i]] = ForwardDiff.partials(residual[rows[i]])[part]
    end
end
function __shooting_jacobian_group!(J, u, cache, group, mode::AutoForwardDiff)
    (; work, chunk, colors, ncolors, entries, rows, cols) = group
    ncolors == 0 && return J
    C = __shooting_chunk(mode, ncolors)
    for color in 1:C:ncolors
        __shooting_seed_kernel!(cache.platform)(work.input, u, colors, color, chunk; ndrange = length(u))
        synchronize(cache.platform)
        __shooting_residual!(work.residual, work.input, cache, work, group.selection)
        __shooting_extract_kernel!(cache.platform)(SparseArrays.nonzeros(J), work.residual, entries, rows, cols, colors, color, chunk; ndrange = length(entries))
        synchronize(cache.platform)
    end
    return J
end
@kernel function __shooting_perturb_kernel!(out, u, colors, color, relstep, absstep, sign)
    i = @index(Global, Linear)
    @inbounds out[i] = u[i] + (colors[i] == color ? sign * max(relstep * abs(u[i]), absstep) : zero(eltype(u)))
end
@kernel function __shooting_fd_extract_kernel!(values, plus, minus, u, entries, rows, cols, colors, color, relstep, absstep, direction, scale)
    i = @index(Global, Linear)
    @inbounds begin
        col = cols[i]
        if colors[col] == color
            values[entries[i]] = (plus[rows[i]] - minus[rows[i]]) / (scale * direction * max(relstep * abs(u[col]), absstep))
        end
    end
end
function __shooting_jacobian_group!(J, u, cache, group, mode::AutoFiniteDiff)
    (; work, colors, ncolors, entries, rows, cols, minus) = group
    ncolors == 0 && return J
    central = mode.fdjtype isa Val{:central}
    T = eltype(u)
    step = central ? cbrt(eps(T)) : sqrt(eps(T))
    relstep = mode.relstep === nothing ? step : T(mode.relstep)
    absstep = mode.absstep === nothing ? relstep : T(mode.absstep)
    direction = mode.dir ? one(T) : -one(T)
    relstep > 0 && absstep > 0 || throw(ArgumentError("Finite-difference steps must be positive."))
    central || __shooting_residual!(minus, u, cache, work, group.selection)
    for color in 1:ncolors
        __shooting_perturb_kernel!(cache.platform)(work.input, u, colors, color, relstep, absstep, direction; ndrange = length(u))
        synchronize(cache.platform)
        __shooting_residual!(work.residual, work.input, cache, work, group.selection)
        if central
            __shooting_perturb_kernel!(cache.platform)(work.input, u, colors, color, relstep, absstep, -direction; ndrange = length(u))
            synchronize(cache.platform)
            __shooting_residual!(minus, work.input, cache, work, group.selection)
        end
        __shooting_fd_extract_kernel!(cache.platform)(SparseArrays.nonzeros(J), work.residual, minus, u, entries, rows, cols, colors, color, relstep, absstep, direction, central ? 2 : 1; ndrange = length(entries))
        synchronize(cache.platform)
    end
    return J
end
function __shooting_jacobian!(J, u, cache, plan)
    for group in plan.groups
        __shooting_jacobian_group!(J, u, cache, group, group.mode)
    end
    return J
end
