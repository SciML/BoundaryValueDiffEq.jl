module BoundaryValueDiffEqShootingCUDSSExt

using BoundaryValueDiffEqCore: __device_sparse_matrix
using BoundaryValueDiffEqShooting: BoundaryValueDiffEqShooting
using CUDA: CUDA, CuArray
using CUDSS: CUDSS
using KernelAbstractions: @kernel, @index
using LinearAlgebra: mul!, ldiv!, norm
using LinearSolve: LinearSolve
using SciMLBase: SciMLBase, ReturnCode
using SparseArrays: SparseArrays, sparse, nnz, nonzeros

@static if isdefined(CUDA, :cuSPARSE)
    const CuCSR = CUDA.cuSPARSE.CuSparseMatrixCSR
else
    const CuCSR = CUDA.CUSPARSE.CuSparseMatrixCSR
end

const LinearVerbosity = @static if isdefined(LinearSolve, :LinearVerbosity)
    Union{LinearSolve.LinearVerbosity, Bool}
else
    Bool
end

# Private LinearSolve adapter selected automatically for square CUDA shooting
# problems. Layout and fallback policy belong to this extension, not Core's API.
struct CachedFactorization{L} <: LinearSolve.AbstractSparseFactorization
    layout::L
end

function default_linsolve(u, cache, plan; segment_length::Int = 32)
    segment_length > 0 || throw(ArgumentError("segment_length must be positive."))
    layout = if cache.twopoint && cache.na + cache.nb == cache.n &&
            segment_length > 1 && cache.intervals > 1
        condensation_layout(cache, plan, segment_length)
    else
        nothing
    end
    return CachedFactorization(layout)
end

function BoundaryValueDiffEqShooting.__shooting_default_linsolve(u::CuArray, cache, plan)
    return default_linsolve(u, cache, plan)
end

function analyze(A, x, b)
    solver = CUDSS.CudssSolver(A, "G", 'F')
    CUDSS.cudss_set(solver, "reordering_alg", "algo1")
    CUDSS.cudss_set(solver, "pivot_type", 'R')
    CUDSS.cudss_set(solver, "ir_n_steps", 2)
    CUDSS.cudss("analysis", solver, x, b; asynchronous = false)
    return solver
end

mutable struct LinearStats
    analyses::Int
    factorizations::Int
    solves::Int
    fallbacks::Int
end
LinearStats() = LinearStats(0, 0, 0, 0)

# No cuDSS object is created here. In particular init_cacheval never factorizes
# NonlinearSolve's zero-valued placeholder Jacobian.
mutable struct DirectCache{F, A, R, C}
    factor::Union{Nothing, F}
    matrix::A
    rowptr::R
    colind::C
    dims::Tuple{Int, Int}
end
function DirectCache(A::CuCSR{T, I}) where {T, I}
    return DirectCache{CUDSS.CudssSolver{T, I}, typeof(A), typeof(A.rowPtr), typeof(A.colVal)}(
        nothing, A, copy(A.rowPtr), copy(A.colVal), size(A)
    )
end

function same_pattern(state::DirectCache, A)
    return state.dims == size(A) && length(state.colind) == nnz(A) &&
        all(state.rowptr .== A.rowPtr) && all(state.colind .== A.colVal)
end

function factorize!(state::DirectCache, A, x, b, stats)
    all(isfinite, nonzeros(A)) || return false
    if state.factor === nothing || !same_pattern(state, A)
        state.factor = nothing
        state.matrix = A
        state.rowptr = copy(A.rowPtr)
        state.colind = copy(A.colVal)
        state.dims = size(A)
        factor = analyze(A, x, b)
        stats.analyses += 1
        state.factor = factor
    else
        state.matrix = A # keep the arrays referenced by cuDSS alive
        CUDSS.cudss_update(state.factor, A)
    end
    # Reuse symbolic analysis, but permit new numerical pivots for new values.
    # Refactorization with fixed numerical pivots is not safe for arbitrary BVPs.
    CUDSS.cudss_set(state.factor, "info", 0)
    CUDSS.cudss("factorization", state.factor, x, b; asynchronous = false)
    stats.factorizations += 1
    return CUDSS.cudss_get(state.factor, "info") == 0
end

function backsolve!(x, state::DirectCache, b, stats)
    ldiv!(x, state.factor, b)
    stats.solves += 1
    return CUDSS.cudss_get(state.factor, "info") == 0
end

# Bounded segment condensation of A_i*x_i + D_i*x_(i+1) = b_i.
# D_i is diagonal (-I for analytic AD). Read its actual values so finite
# differences and scaling do not require an exactly represented -1.
# Structural flow components are independent inside a segment; boundary rows
# may still couple arbitrary components in the reduced global system.

function condensation_layout(cache, plan, segment_length)
    (; n, intervals, na, nb) = cache
    graph = plan.flow .| sparse(transpose(plan.flow))
    seen = falses(n)
    components = Vector{Int}[]
    for root in 1:n
        seen[root] && continue
        seen[root] = true
        group = [root]
        cursor = 1
        while cursor <= length(group)
            column = group[cursor]
            for k in SparseArrays.nzrange(graph, column)
                row = SparseArrays.rowvals(graph)[k]
                if !seen[row]
                    seen[row] = true
                    push!(group, row)
                end
            end
            cursor += 1
        end
        push!(components, sort!(group))
    end
    states = reduce(vcat, components)
    offsets = cumsum(vcat(1, length.(components)))
    dense_offsets = cumsum(vcat(0, length.(components) .^ 2))
    localpos = zeros(Int, n)
    for group in components, (j, state) in enumerate(group)
        localpos[state] = j
    end
    segments = cld(intervals, segment_length)
    rows, cols = Int[], Int[]
    for segment in 1:segments, group in components, row in group
        for col in group
            push!(rows, na + (segment - 1) * n + row)
            push!(cols, (segment - 1) * n + col)
        end
        push!(rows, na + (segment - 1) * n + row)
        push!(cols, segment * n + row)
    end
    boundary = Dict{Tuple{Int, Int}, Int}()
    for k in eachindex(plan.rows)
        row, col = plan.rows[k], plan.cols[k]
        if row <= na
            boundary[(row, col)] = k
            push!(rows, row); push!(cols, col)
        elseif row > na + intervals * n
            reduced_row = row - (intervals - segments) * n
            reduced_col = col - (intervals - segments) * n
            boundary[(reduced_row, reduced_col)] = k
            push!(rows, reduced_row); push!(cols, reduced_col)
        end
    end
    dimension = n * (segments + 1)
    pattern = sparse(rows, cols, trues(length(rows)), dimension, dimension)
    return (;
        n, intervals, na, nb, segments, segment_length,
        states, offsets, dense_offsets, localpos, pattern, boundary,
    )
end

function condensation_buffers(layout, u)
    platform = CUDA.CUDABackend()
    (; n, segments, states, offsets, dense_offsets, localpos) = layout
    storage = __device_sparse_matrix(u, layout.pattern)
    old_entries, new_entries = Int[], Int[]
    for k in eachindex(storage.rows)
        original = get(layout.boundary, (storage.rows[k], storage.cols[k]), 0)
        if original != 0
            push!(old_entries, original)
            push!(new_entries, k)
        end
    end
    return (;
        matrix = storage.matrix, n, segments, layout.intervals, layout.na, layout.nb,
        layout.segment_length, components = length(offsets) - 1,
        states = CuArray(states), offsets = CuArray(offsets), dense_offsets = CuArray(dense_offsets),
        localpos = CuArray(localpos), old_entries = CuArray(old_entries), new_entries = CuArray(new_entries),
        p0 = similar(u, segments, last(dense_offsets)), p1 = similar(u, segments, last(dense_offsets)),
        q0 = similar(u, segments, n), q1 = similar(u, segments, n),
        valid = similar(u, Bool, segments * (length(offsets) - 1)),
        b = similar(u, size(storage.matrix, 1)), x = similar(u, size(storage.matrix, 2)), platform,
    )
end

@kernel function condense_matrix_kernel!(
        reduced_values, reduced_rowptr, values, rowptr, colind,
        p0, p1, valid, states, offsets, dense_offsets, localpos,
        n, na, intervals, segments, segment_length, transfer_limit
    )
    task = @index(Global, Linear)
    segment = (task - 1) % segments + 1
    component = (task - 1) ÷ segments + 1
    @inbounds begin
        lo, hi = offsets[component], offsets[component + 1] - 1
        d = hi - lo + 1
        base = dense_offsets[component]
        for col in 1:d, row in 1:d
            p0[segment, base + (col - 1) * d + row] = row == col ? one(eltype(p0)) : zero(eltype(p0))
        end
        current, next = p0, p1
        first_interval = (segment - 1) * segment_length + 1
        last_interval = min(segment * segment_length, intervals)
        safe = true
        for interval in first_interval:last_interval
            for row in 1:d
                state = states[lo + row - 1]
                full_row = na + (interval - 1) * n + state
                start, last = rowptr[full_row], rowptr[full_row + 1] - 1
                diagonal = values[last]
                safe &= colind[last] == interval * n + state && isfinite(diagonal) && !iszero(diagonal)
                for col in 1:d
                    value = zero(eltype(p0))
                    for k in start:(last - 1)
                        input = colind[k] - (interval - 1) * n
                        if 1 <= input <= n
                            value -= values[k] * current[segment, base + (col - 1) * d + localpos[input]] / diagonal
                        else
                            safe = false
                        end
                    end
                    safe &= isfinite(value) && abs(value) <= transfer_limit
                    next[segment, base + (col - 1) * d + row] = value
                end
            end
            current, next = next, current
        end
        for row in 1:d
            state = states[lo + row - 1]
            reduced_row = na + (segment - 1) * n + state
            start = reduced_rowptr[reduced_row]
            for col in 1:d
                reduced_values[start + col - 1] = current[segment, base + (col - 1) * d + row]
            end
            reduced_values[start + d] = -one(eltype(reduced_values))
        end
        valid[task] = safe
    end
end

@kernel function copy_boundary_values!(out, values, old_entries, new_entries)
    i = @index(Global, Linear)
    @inbounds out[new_entries[i]] = values[old_entries[i]]
end

function condense_matrix!(work, A)
    (; matrix, platform, segments, components) = work
    condense_matrix_kernel!(platform, 64)(
        nonzeros(matrix), matrix.rowPtr, nonzeros(A), A.rowPtr, A.colVal,
        work.p0, work.p1, work.valid, work.states, work.offsets, work.dense_offsets, work.localpos,
        work.n, work.na, work.intervals, segments, work.segment_length, inv(sqrt(eps(eltype(A))));
        ndrange = segments * components
    )
    if !isempty(work.old_entries)
        copy_boundary_values!(platform)(nonzeros(matrix), nonzeros(A), work.old_entries, work.new_entries; ndrange = length(work.old_entries))
    end
    # Same-stream kernels are ordered; this reduction synchronizes once, after
    # all transfer computations, and returns just the validity decision.
    return all(work.valid)
end

@kernel function condense_rhs_kernel!(
        reduced_b, b, values, rowptr, colind, q0, q1, states, offsets,
        n, na, intervals, segments, segment_length
    )
    task = @index(Global, Linear)
    segment = (task - 1) % segments + 1
    component = (task - 1) ÷ segments + 1
    @inbounds begin
        lo, hi = offsets[component], offsets[component + 1] - 1
        for j in lo:hi
            q0[segment, states[j]] = zero(eltype(q0))
        end
        current, next = q0, q1
        for interval in ((segment - 1) * segment_length + 1):min(segment * segment_length, intervals)
            for j in lo:hi
                state = states[j]
                row = na + (interval - 1) * n + state
                start, last = rowptr[row], rowptr[row + 1] - 1
                value = b[row]
                for k in start:(last - 1)
                    input = colind[k] - (interval - 1) * n
                    value -= values[k] * current[segment, input]
                end
                next[segment, state] = value / values[last]
            end
            current, next = next, current
        end
        for j in lo:hi
            state = states[j]
            reduced_b[na + (segment - 1) * n + state] = -current[segment, state]
        end
    end
end

function condense_rhs!(work, A, b)
    (; n, na, nb, segments, intervals) = work
    condense_rhs_kernel!(work.platform, 64)(
        work.b, b, nonzeros(A), A.rowPtr, A.colVal, work.q0, work.q1, work.states, work.offsets,
        n, na, intervals, segments, work.segment_length; ndrange = segments * work.components
    )
    copyto!(work.b, 1, b, 1, na)
    copyto!(work.b, na + segments * n + 1, b, na + intervals * n + 1, nb)
    return work.b
end

@kernel function recover_kernel!(
        x, reduced_x, b, values, rowptr, colind, states, offsets,
        n, na, intervals, segments, segment_length
    )
    task = @index(Global, Linear)
    segment = (task - 1) % segments + 1
    component = (task - 1) ÷ segments + 1
    @inbounds begin
        lo, hi = offsets[component], offsets[component + 1] - 1
        first_interval = (segment - 1) * segment_length + 1
        last_interval = min(segment * segment_length, intervals)
        for j in lo:hi
            state = states[j]
            x[(first_interval - 1) * n + state] = reduced_x[(segment - 1) * n + state]
            if segment == segments
                x[intervals * n + state] = reduced_x[segments * n + state]
            end
        end
        # Endpoints are owned by the reduced solve. Reconstruct only interiors.
        for interval in first_interval:(last_interval - 1)
            for j in lo:hi
                state = states[j]
                row = na + (interval - 1) * n + state
                start, last = rowptr[row], rowptr[row + 1] - 1
                value = b[row]
                for k in start:(last - 1)
                    value -= values[k] * x[colind[k]]
                end
                x[interval * n + state] = value / values[last]
            end
        end
    end
end

function recover!(x, work, A, b)
    recover_kernel!(work.platform, 64)(
        x, work.x, b, nonzeros(A), A.rowPtr, A.colVal, work.states, work.offsets,
        work.n, work.na, work.intervals, work.segments, work.segment_length;
        ndrange = work.segments * work.components
    )
    return x
end

mutable struct ShootingLinearCache{D, W, V}
    full::D
    reduced::Union{Nothing, D}
    work::W
    active::Bool
    residual::V
    correction::V
    stats::LinearStats
end

function LinearSolve.init_cacheval(
        alg::CachedFactorization, A::CuCSR, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::LinearVerbosity, assumptions::LinearSolve.OperatorAssumptions
    )
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("The shooting cuDSS solver requires a square matrix."))
    eltype(A) <: Union{Float32, Float64} || throw(ArgumentError("The shooting cuDSS solver requires Float32 or Float64."))
    layout = alg.layout
    work = layout === nothing ? nothing : condensation_buffers(layout, u)
    full = DirectCache(A)
    reduced = work === nothing ? nothing : DirectCache(work.matrix)
    return ShootingLinearCache(
        full, reduced, work, work !== nothing, similar(u), similar(u), LinearStats()
    )
end

function disable_condensation!(state)
    state.active = false
    state.stats.fallbacks += 1
    return nothing
end

function linear_residual!(r, A, x, b)
    mul!(r, A, x)
    r .= b .- r
    return norm(r, Inf)
end

# A strict relative residual guard is useful for detecting unstable segment
# transfers. Bound the requested tolerance so loose Newton forcing cannot hide
# an inaccurate condensed correction. cuDSS also performs internal refinement.
function residual_tolerance(cache, b)
    T = eltype(b)
    relative = min(T(cache.reltol), sqrt(eps(T)))
    return max(T(cache.abstol), relative * norm(b, Inf))
end

function solve_condensed!(x, state, A, b)
    work = state.work
    condense_rhs!(work, A, b)
    backsolve!(work.x, state.reduced, work.b, state.stats) || return false
    recover!(x, work, A, b)
    return true
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CachedFactorization; kwargs...)
    state = cache.cacheval
    A, b, x = cache.A, cache.b, cache.u
    size(A, 1) == size(A, 2) && length(b) == size(A, 1) && length(x) == size(A, 2) ||
        throw(DimensionMismatch("The shooting cuDSS solver requires a square system with compatible vectors."))
    if length(state.residual) != length(b)
        state.residual = similar(b)
        state.correction = similar(x)
    end
    if cache.isfresh && state.active
        # A changed structure invalidates both the analysis and shooting layout.
        if !same_pattern(state.full, A) || !condense_matrix!(state.work, A)
            disable_condensation!(state)
        elseif !factorize!(state.reduced, state.work.matrix, state.work.x, state.work.b, state.stats)
            disable_condensation!(state)
        end
    end
    tolerance = residual_tolerance(cache, b)
    if state.active
        ok = solve_condensed!(x, state, A, b)
        # Refine against the ORIGINAL system, not just the reduced equations.
        for _ in 1:2
            ok || break
            residual = linear_residual!(state.residual, A, x, b)
            if isfinite(residual) && residual <= tolerance
                cache.isfresh = false
                return SciMLBase.build_linear_solution(alg, x, nothing, nothing; retcode = ReturnCode.Success)
            end
            isfinite(residual) || break
            ok = solve_condensed!(state.correction, state, A, state.residual)
            x .+= state.correction
        end
        if ok
            residual = linear_residual!(state.residual, A, x, b)
            if isfinite(residual) && residual <= tolerance
                cache.isfresh = false
                return SciMLBase.build_linear_solution(alg, x, nothing, nothing; retcode = ReturnCode.Success)
            end
        end
        disable_condensation!(state)
    end
    if cache.isfresh || state.full.factor === nothing
        if !factorize!(state.full, A, x, b, state.stats)
            return SciMLBase.build_linear_solution(alg, x, nothing, nothing; retcode = ReturnCode.Failure)
        end
    end
    ok = backsolve!(x, state.full, b, state.stats)
    residual = linear_residual!(state.residual, A, x, b)
    ok &= isfinite(residual) && residual <= tolerance
    cache.isfresh = !ok
    return SciMLBase.build_linear_solution(
        alg, x, nothing, nothing; retcode = ok ? ReturnCode.Success : ReturnCode.Failure
    )
end

end
