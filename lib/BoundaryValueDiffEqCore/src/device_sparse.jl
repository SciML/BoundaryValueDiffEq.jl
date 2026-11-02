"""
    __device_sparse_supported(template)

Whether the array backend provides sparse Jacobian storage. Device extensions
implement this hook together with [`__device_sparse_matrix`](@ref).
"""
__device_sparse_supported(template::AbstractArray) =
    parent(template) === template ? false : __device_sparse_supported(parent(template))
__device_sparse_supported(::Array) = true

"""
    __device_sparse_matrix(template, pattern::SparseMatrixCSC)

Allocate zero-valued sparse storage with the structure of `pattern`, using the
element type and device of `template`. Return `(; matrix, rows, cols)`, where the
host coordinates `rows` and `cols` follow the storage order of `nonzeros(matrix)`.
Keep structural zeros and do not alias or modify the input pattern.
"""
function __device_sparse_matrix(template, pattern)
    if template isa AbstractArray && parent(template) !== template
        return __device_sparse_matrix(parent(template), pattern)
    end
    throw(ArgumentError("No sparse-storage adapter is loaded for $(typeof(template)). Load CUDA for CuArray support, or implement BoundaryValueDiffEqCore.__device_sparse_matrix and __device_sparse_supported for the backend."))
end

function __device_sparse_matrix(template::Array, pattern::SparseMatrixCSC)
    matrix = SparseMatrixCSC(
        size(pattern)..., copy(pattern.colptr), copy(rowvals(pattern)),
        zeros(eltype(template), nnz(pattern))
    )
    rows, cols, _ = findnz(matrix)
    return (; matrix, rows, cols)
end

"""
    __device_residual!(residual, u, cache, boundary = true[, reuse])

Evaluate a resident collocation residual. Solvers dispatch on their cache type;
`boundary = false` skips boundary equations. The optional `reuse` flag lets nested
FIRK reuse primal stages and factorizations within a ForwardDiff color sweep.
"""
function __device_residual! end
__device_residual!(residual, u, cache, boundary, reuse) =
    __device_residual!(residual, u, cache, boundary)

"""
    __bvp_device_unknowns(cache)

Return the packed nonlinear unknowns. They can exclude internal stage values.
The Jacobian workspace is stored in `cache.device_cache`, separately from stage
buffers, and uses `__bvp_device_residual_prototype(cache)` for its backend and size.
"""
function __bvp_device_unknowns end

"""
    __bvp_device_residual_prototype(cache)

Return the resident residual buffer used to determine Jacobian workspace size,
element type and backend. Solvers with directly owned buffers override this accessor.
"""
__bvp_device_residual_prototype(cache) = cache.residual[]

"""
    __bvp_device_jacobian_plan(cache)

Return the resident sparse Jacobian plan, or `nothing` for dense differentiation.
Solvers override this accessor to match their cache storage.
"""
__bvp_device_jacobian_plan(cache) = cache.jacobian_cache[]

function __bvp_device_jacobian_buffers(cache, ::Type{T}) where {T}
    return get!(cache.device_cache, Tuple{T}) do
        residual = __bvp_device_residual_prototype(cache)
        (;
            input = similar(residual, T, length(__bvp_device_unknowns(cache))),
            plus = similar(residual, T), minus = similar(residual, T),
        )
    end
end
# Dense and colored Jacobians share residual evaluations and kernel launches.
# Only the column seeds and extraction layout depend on the storage strategy.
@inline __bvp_device_color(::Nothing, i) = i
@inline __bvp_device_color(colors::AbstractVector, i) = @inbounds colors[i]

@kernel function __bvp_device_seed_kernel!(dual, u, column, ::Val{C}, colors) where {C}
    i = @index(Global, Linear)
    color = __bvp_device_color(colors, i)
    @inbounds dual[i] = eltype(dual)(
        u[i], ForwardDiff.Partials(
            ntuple(j -> color == column + j - 1 ? one(eltype(u)) : zero(eltype(u)), Val(C))
        )
    )
end

@inline function __bvp_device_extract_jacobian!(
        J::AbstractMatrix, residual, column, firstrow, ::Val{C}, index,
        ::Nothing, rows, cols, colors
    ) where {C}
    row = firstrow + (index - 1) % (size(J, 1) - firstrow + 1)
    part = (index - 1) ÷ (size(J, 1) - firstrow + 1) + 1
    @inbounds J[row, column + part - 1] = ForwardDiff.partials(residual[row])[part]
    return nothing
end

@inline function __bvp_device_extract_jacobian!(
        values::AbstractVector, residual, column, firstrow, ::Val{C}, index,
        entries::AbstractVector, rows, cols, colors
    ) where {C}
    @inbounds begin
        part = colors[cols[index]] - column + 1
        if 1 <= part <= C
            values[entries[index]] = ForwardDiff.partials(residual[rows[index]])[part]
        end
    end
    return nothing
end

@kernel function __bvp_device_extract_jac_kernel!(
        J, residual, column, firstrow, chunk, entries, rows, cols, colors
    )
    index = @index(Global, Linear)
    __bvp_device_extract_jacobian!(
        J, residual, column, firstrow, chunk, index, entries, rows, cols, colors
    )
end

@inline __bvp_device_fd_step(value, relstep, absstep, direction) =
    max(abs(value) * relstep, absstep) * direction

@kernel function __bvp_device_perturb_kernel!(out, u, column, relstep, absstep, direction, colors)
    i = @index(Global, Linear)
    @inbounds out[i] = u[i] + (
        __bvp_device_color(colors, i) == column ?
            __bvp_device_fd_step(u[i], relstep, absstep, direction) : zero(eltype(u))
    )
end

@inline function __bvp_device_extract_jacobian!(
        J::AbstractMatrix, plus, minus, u, column, relstep, absstep, direction, central,
        row, ::Nothing, rows, cols, colors
    )
    @inbounds begin
        h = __bvp_device_fd_step(u[column], relstep, absstep, direction)
        J[row, column] = (plus[row] - minus[row]) / (central ? 2h : h)
    end
    return nothing
end

@inline function __bvp_device_extract_jacobian!(
        values::AbstractVector, plus, minus, u, column, relstep, absstep, direction, central,
        index, entries::AbstractVector, rows, cols, colors
    )
    @inbounds begin
        col = cols[index]
        if colors[col] == column
            h = __bvp_device_fd_step(u[col], relstep, absstep, direction)
            row = rows[index]
            values[entries[index]] = (plus[row] - minus[row]) / (central ? 2h : h)
        end
    end
    return nothing
end

@kernel function __bvp_device_fd_jac_kernel!(
        J, plus, minus, u, column, relstep, absstep, direction, central,
        entries, rows, cols, colors
    )
    index = @index(Global, Linear)
    __bvp_device_extract_jacobian!(
        J, plus, minus, u, column, relstep, absstep, direction, central,
        index, entries, rows, cols, colors
    )
end

__bvp_device_jacobian_view(J, residual, group) =
    __bvp_device_jacobian_view(J, residual, group, group.entries)

function __bvp_device_jacobian_view(J, residual, group, ::Nothing)
    return (;
        values = view(J, group.rows, :), residual = view(residual, group.rows),
        entries = nothing, rows = nothing, cols = nothing, colors = nothing,
    )
end

function __bvp_device_jacobian_view(J, residual, group, entries::AbstractVector)
    return (;
        values = SparseArrays.nonzeros(J), residual, entries,
        rows = group.rows, cols = group.cols, colors = group.colors,
    )
end

__bvp_device_jacobian_ndrange(J::AbstractMatrix, ::Nothing, count) = size(J, 1) * count
__bvp_device_jacobian_ndrange(values::AbstractVector, entries::AbstractVector, count) = length(entries)

"""
    __bvp_device_ad_jacobian!(J, u, cache, mode, rows, boundary)

Fill selected Jacobian rows with ForwardDiff or finite differences, optionally evaluating the boundary.
"""
function __bvp_device_ad_jacobian!(J, u, cache, mode, rows, boundary::Bool)
    group = (; rows, boundary, colors = nothing, entries = nothing, ncolors = length(u))
    return __bvp_device_ad_jacobian!(J, u, cache, mode, group)
end

function __bvp_device_ad_jacobian!(J, u, cache, mode::AutoForwardDiff{C}, group) where {C}
    group.ncolors == 0 && return J
    return __bvp_device_ad_jacobian!(
        J, u, cache, mode, group,
        Val(C === nothing ? min(group.ncolors, 8) : min(C, group.ncolors))
    )
end

function __bvp_device_ad_jacobian!(J, u, cache, mode, group, chunk::Val{C}) where {C}
    tag = mode.tag === nothing ? typeof(ForwardDiff.Tag(cache, eltype(u))) : typeof(mode.tag)
    D = ForwardDiff.Dual{tag, eltype(u), C}
    work = __bvp_device_jacobian_buffers(cache, D)
    target = __bvp_device_jacobian_view(J, work.plus, group)
    platform = cache.alg.platform
    for column in 1:C:group.ncolors
        __bvp_device_seed_kernel!(platform)(
            work.input, u, column, chunk, group.colors; ndrange = length(u)
        )
        synchronize(platform)
        __device_residual!(work.plus, work.input, cache, group.boundary, column > 1)
        __bvp_device_extract_jac_kernel!(platform)(
            target.values, target.residual, column, 1, chunk,
            target.entries, target.rows, target.cols, target.colors;
            ndrange = __bvp_device_jacobian_ndrange(
                target.values, target.entries, min(C, group.ncolors - column + 1)
            )
        )
        synchronize(platform)
    end
    return J
end

function __bvp_device_ad_jacobian!(J, u, cache, mode::AutoFiniteDiff, group)
    group.ncolors == 0 && return J
    T = eltype(u)
    central = mode.fdjtype isa Val{:central}
    default_step = central ? cbrt(eps(T)) : sqrt(eps(T))
    relstep = hasproperty(mode, :relstep) && mode.relstep !== nothing ? T(mode.relstep) : default_step
    absstep = hasproperty(mode, :absstep) && mode.absstep !== nothing ? T(mode.absstep) : relstep
    direction = hasproperty(mode, :dir) && !mode.dir ? -one(T) : one(T)
    work = __bvp_device_jacobian_buffers(cache, T)
    target = __bvp_device_jacobian_view(J, work.plus, group)
    minus = __bvp_device_jacobian_view(J, work.minus, group).residual
    platform = cache.alg.platform
    central || __device_residual!(work.minus, u, cache, group.boundary)
    for column in 1:group.ncolors
        __bvp_device_perturb_kernel!(platform)(
            work.input, u, column, relstep, absstep, direction, group.colors; ndrange = length(u)
        )
        synchronize(platform)
        __device_residual!(work.plus, work.input, cache, group.boundary)
        if central
            __bvp_device_perturb_kernel!(platform)(
                work.input, u, column, relstep, absstep, -direction, group.colors; ndrange = length(u)
            )
            synchronize(platform)
            __device_residual!(work.minus, work.input, cache, group.boundary)
        end
        __bvp_device_fd_jac_kernel!(platform)(
            target.values, target.residual, minus, u, column, relstep, absstep, direction, central,
            target.entries, target.rows, target.cols, target.colors;
            ndrange = __bvp_device_jacobian_ndrange(target.values, target.entries, 1)
        )
        synchronize(platform)
    end
    return J
end

"""
    __device_jacobian!(J, u, cache)

Fill a resident Jacobian using cache-dispatched residuals, dense or colored storage and per-group AD.
"""
__device_jacobian!(J, u, cache) = __device_jacobian!(J, u, cache, __bvp_device_jacobian_plan(cache))

function __device_jacobian!(J, u, cache, plan::SparseJacobianCache)
    for group in plan.groups
        isempty(group.entries) && continue
        __bvp_device_ad_jacobian!(J, u, cache, group.mode, group)
    end
    return J
end

function __device_jacobian!(J, u, cache, ::Nothing)
    jac_alg = cache.alg.jac_alg
    return __device_jacobian!(J, u, cache, cache.problem_type, jac_alg)
end

function __device_jacobian!(J, u, cache, ::Union{TwoPointBVProblem, TwoPointSecondOrderBVProblem}, jac_alg)
    return __bvp_device_ad_jacobian!(J, u, cache, get_dense_ad(jac_alg.diffmode), axes(J, 1), true)
end

function __device_jacobian!(J, u, cache, ::Union{StandardBVProblem, StandardSecondOrderBVProblem}, jac_alg)
    bc_mode, ode_mode = get_dense_ad(jac_alg.bc_diffmode), get_dense_ad(jac_alg.nonbc_diffmode)
    if bc_mode == ode_mode
        __bvp_device_ad_jacobian!(J, u, cache, bc_mode, axes(J, 1), true)
    else
        nbc = prod(cache.resid_size[1])
        __bvp_device_ad_jacobian!(J, u, cache, bc_mode, 1:nbc, true)
        __bvp_device_ad_jacobian!(J, u, cache, ode_mode, (nbc + 1):size(J, 1), false)
    end
    return J
end

"""
    __device_jacobian_products(cache)

Return in-place JVP and VJP callbacks backed by the sparse plan, or `nothing` callbacks for dense storage.
"""
__device_jacobian_products(cache) = __device_jacobian_products(cache, __bvp_device_jacobian_plan(cache))
__device_jacobian_products(cache, ::Nothing) = (; jvp = nothing, vjp = nothing)
function __device_jacobian_products(cache, ::SparseJacobianCache)
    # Explicit products prevent generic line searches from allocating a dense J.
    jvp = (out, v, u, p) -> __bvp_device_jacobian_product!(out, v, u, cache, Val(false))
    vjp = (out, v, u, p) -> __bvp_device_jacobian_product!(out, v, u, cache, Val(true))
    return (; jvp, vjp)
end
function __bvp_device_jacobian_product!(out, v, u, cache, transposed)
    plan = __bvp_device_jacobian_plan(cache)
    __device_jacobian!(plan.product, u, cache, plan)
    LinearAlgebra.mul!(vec(out), __bvp_device_jacobian_product(plan.product, transposed), vec(v))
    return nothing
end
__bvp_device_jacobian_product(J, ::Val{false}) = J
__bvp_device_jacobian_product(J, ::Val{true}) = adjoint(J)

"""
    __device_host_parameter(p)

Copy mutable device parameter arrays to host storage recursively for boundary sparsity tracing.
"""
__device_host_parameter(p) = p
__device_host_parameter(p::AbstractArray) = isbits(p) ? p : Array(p)
__device_host_parameter(p::Union{Tuple, NamedTuple}) =
    map(__device_host_parameter, p)

function __bvp_device_pattern_entries(pattern)
    # findnz includes explicitly stored zeros of a sparse user prototype. Those
    # entries describe structure and must not be discarded by numeric filtering.
    rows, columns, _ = SparseArrays.findnz(sparse(pattern))
    return rows, columns
end

"""
    __bvp_device_sparse_group(pattern, rows, mode, boundary)

Color the selected residual rows, compact active colors and omit seeds for structurally empty columns.
"""
function __bvp_device_sparse_group(pattern, rows, mode, boundary)
    subpattern = pattern[rows, :]
    coloring_algorithm = __default_coloring_algorithm(mode)
    # AutoSparse's constructor uses NoColoringAlgorithm when the user omits
    # this option. Use BVP's default coloring for the known device structure.
    if coloring_algorithm isa ADTypes.NoColoringAlgorithm
        coloring_algorithm = __default_coloring_algorithm(nothing)
    end
    colors = collect(Int, ADTypes.column_coloring(subpattern, coloring_algorithm))
    length(colors) == size(pattern, 2) ||
        throw(DimensionMismatch("Device BVP column coloring has the wrong length."))
    # Empty columns need no seed. Compact remaining color IDs so a boundary
    # condition involving only a few distant nodes launches only useful batches.
    for column in eachindex(colors)
        if isempty(SparseArrays.nzrange(subpattern, column))
            colors[column] = 0
        elseif colors[column] <= 0
            throw(ArgumentError("Device BVP column coloring must be positive for nonempty columns."))
        end
    end
    active_colors = sort!(unique(filter(>(0), colors)))
    color_ids = Dict(color => index for (index, color) in enumerate(active_colors))
    map!(color -> color == 0 ? 0 : color_ids[color], colors, colors)
    return (;
        mode = get_dense_ad(mode), rows, colors, ncolors = length(active_colors), boundary,
    )
end
function __bvp_device_collocation_pattern(y, nleft, nright, stage = 0)
    M, nodes = size(y)
    ncollocation, nunknowns = M * (nodes - 1), length(y)
    nresiduals = nleft + ncollocation + nright
    rows, columns = Int[], Int[]
    sizehint!(rows, 2M * ncollocation + M * (nleft + nright))
    sizehint!(columns, 2M * ncollocation + M * (nleft + nright))
    for interval in 1:((nodes - 1) ÷ (stage + 1))
        firstcol = (interval - 1) * (stage + 1) * M
        # Continuity and stage equations couple the interval's stages and endpoints.
        for column in (firstcol + 1):(firstcol + (stage + 2) * M)
            for row in (nleft + firstcol + 1):(nleft + firstcol + (stage + 1) * M)
                push!(rows, row)
                push!(columns, column)
            end
        end
    end
    return rows, columns, nresiduals, nunknowns
end
"""
    __device_boundary_pattern(make_boundary, mode, T, nbc, nunknowns)

Trace boundary dependencies on the host. Known sparsity bypasses callback
construction; unsupported tracing conservatively fills the entire boundary border.
Explicitly stored zeros in a supplied sparse pattern remain structural entries.
"""
function __device_boundary_pattern(make_boundary, mode, T, nbc, nunknowns)
    detector = mode isa AutoSparse ? mode.sparsity_detector : nothing
    return __device_boundary_pattern(make_boundary, detector, T, nbc, nunknowns, Val(:trace))
end
function __device_boundary_pattern(make_boundary, detector::ADTypes.KnownJacobianSparsityDetector, T, nbc, nunknowns, ::Val{:trace})
    pattern = ADTypes.jacobian_sparsity((r, x) -> nothing, zeros(T, nbc), zeros(T, nunknowns), detector)
    rows, columns = __bvp_device_pattern_entries(pattern)
    return sparse(rows, columns, trues(length(rows)), nbc, nunknowns), false
end
function __device_boundary_pattern(make_boundary, detector, T, nbc, nunknowns, ::Val{:trace})
    try
        pattern = ADTypes.jacobian_sparsity(
            make_boundary(), zeros(T, nbc), zeros(T, nunknowns),
            SparseConnectivityTracer.TracerSparsityDetector()
        )
        return pattern, false
    catch err
        err isa InterruptException && rethrow()
        rows = repeat(collect(1:nbc), nunknowns)
        columns = repeat(collect(1:nunknowns); inner = nbc)
        return sparse(rows, columns, trues(length(rows)), nbc, nunknowns), true
    end
end

"""
    __device_sparse_structure(problem_type, jac_alg, y, bc_sizes, stage, boundary)

Assemble collocation sparsity from adjacent packed blocks. `stage` is zero for
MIRK, MIRKN and nested FIRK, and the number of retained stages for expanded FIRK.
Problem-type dispatch selects a traced general boundary or dense endpoint blocks.
"""
function __device_sparse_structure(
        ::Union{StandardBVProblem, StandardSecondOrderBVProblem}, jac_alg, y,
        bc_sizes, stage, boundary
    )
    M, nleft = size(y, 1), prod(bc_sizes[1])
    rows, columns, nresiduals, nunknowns = __bvp_device_collocation_pattern(y, nleft, 0, stage)
    boundary_pattern, boundary_fallback = boundary()
    boundary_rows, boundary_columns = __bvp_device_pattern_entries(boundary_pattern)
    boundary_row_counts = zeros(Int, nleft)
    for row in boundary_rows
        boundary_row_counts[row] += 1
    end
    separate = boundary_fallback || any(>((stage + 2) * M), boundary_row_counts)
    append!(rows, boundary_rows)
    append!(columns, boundary_columns)
    pattern = sparse(rows, columns, ones(eltype(y), length(rows)), nresiduals, nunknowns)
    groups = if !separate && get_dense_ad(jac_alg.bc_diffmode) == get_dense_ad(jac_alg.nonbc_diffmode)
        (__bvp_device_sparse_group(pattern, 1:nresiduals, jac_alg.nonbc_diffmode, true),)
    else
        # Keep dense boundary colors separate from the bounded collocation band.
        (
            __bvp_device_sparse_group(pattern, 1:nleft, jac_alg.bc_diffmode, true),
            __bvp_device_sparse_group(pattern, (nleft + 1):nresiduals, jac_alg.nonbc_diffmode, false),
        )
    end
    return (; pattern, groups, boundary_fallback)
end
function __device_sparse_structure(
        ::Union{TwoPointBVProblem, TwoPointSecondOrderBVProblem}, jac_alg, y,
        bc_sizes, stage, boundary
    )
    M, nodes = size(y)
    nleft, nright = prod(bc_sizes[1]), prod(bc_sizes[2])
    rows, columns, nresiduals, nunknowns = __bvp_device_collocation_pattern(y, nleft, nright, stage)
    for column in 1:M, row in 1:nleft
        push!(rows, row)
        push!(columns, column)
    end
    for column in ((nodes - 1) * M + 1):nunknowns, row in (nresiduals - nright + 1):nresiduals
        push!(rows, row)
        push!(columns, column)
    end
    pattern = sparse(rows, columns, ones(eltype(y), length(rows)), nresiduals, nunknowns)
    groups = (__bvp_device_sparse_group(pattern, 1:nresiduals, jac_alg.diffmode, true),)
    return (; pattern, groups, boundary_fallback = false)
end

__bvp_device_bc_length(::Union{TwoPointBVProblem, TwoPointSecondOrderBVProblem}, bc_sizes) = sum(prod, bc_sizes)
__bvp_device_bc_length(::Union{StandardBVProblem, StandardSecondOrderBVProblem}, bc_sizes) = prod(first(bc_sizes))

"""
    __prepare_device_jacobian(make_structure, y, problem_type, bc_sizes)

Allocate sparse storage and coloring metadata when the backend supports it;
otherwise allocate a dense Jacobian without evaluating `make_structure`.
"""
function __prepare_device_jacobian(make_structure, y, problem_type, bc_sizes)
    nbc = __bvp_device_bc_length(problem_type, bc_sizes)
    return __prepare_device_jacobian(make_structure, y, nbc, Val(__device_sparse_supported(y)))
end
function __prepare_device_jacobian(make_structure, y, nbc, ::Val{false})
    return (; matrix = similar(y, size(y, 1) * (size(y, 2) - 1) + nbc, length(y)), plan = nothing)
end
function __prepare_device_jacobian(make_structure, y, nbc, ::Val{true})
    structure = make_structure()
    storage = __device_sparse_matrix(y, structure.pattern)
    platform = KernelAbstractions.get_backend(y)
    upload(x) = __device_parameter(platform, x)
    groups = collect(
        map(structure.groups) do group
            entries = findall(row -> row in group.rows, storage.rows)
            return (;
                mode = get_dense_ad(group.mode), colors = upload(group.colors),
                ncolors = group.ncolors, boundary = group.boundary,
                entries = upload(entries), rows = upload(storage.rows[entries]), cols = upload(storage.cols[entries]),
            )
        end
    )
    plan = SparseJacobianCache(groups, structure.boundary_fallback, structure.pattern, copy(storage.matrix))
    return (; matrix = storage.matrix, plan)
end
