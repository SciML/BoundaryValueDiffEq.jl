# The collocation rows depend only on one interval and its two differential
# endpoints. Build a conservative pattern, independent of the initial iterate
# and of which entries of the user's RHS Jacobian happen to vanish there.
function __ascher_collocation_pattern(cache, nrows, ncols)
    (; ncomp, ny, k, mesh) = cache
    ncy = ncomp + ny
    n = length(mesh) - 1
    nz = ncomp * (n + 1)
    rows, cols = Int[], Int[]
    entries = n * (k * ncy * (ncomp * (k + 1) + ny) + ncomp * (k + 2))
    sizehint!(rows, entries)
    sizehint!(cols, entries)
    for i in 1:n
        roffset = (i - 1) * (k * ncy + ncomp)
        zoffset = (i - 1) * ncomp
        soffset = nz + (i - 1) * k * ncy
        for col in (zoffset + 1):(zoffset + ncomp), row in 1:(k * ncy)
            push!(rows, roffset + row)
            push!(cols, col)
        end
        for l in 1:k, c in 1:ncomp, row in 1:(k * ncy)
            push!(rows, roffset + row)
            push!(cols, soffset + (l - 1) * ncy + c)
        end
        for j in 1:k, c in (ncomp + 1):ncy, r in 1:ncy
            push!(rows, roffset + (j - 1) * ncy + r)
            push!(cols, soffset + (j - 1) * ncy + c)
        end
        for c in 1:ncomp
            row = roffset + k * ncy + c
            for col in (zoffset + c, zoffset + ncomp + c)
                push!(rows, row)
                push!(cols, col)
            end
            for j in 1:k
                push!(rows, row)
                push!(cols, soffset + (j - 1) * ncy + c)
            end
        end
    end
    return sparse(rows, cols, ones(eltype(cache), length(rows)), nrows, ncols)
end

function __ascher_twopoint_pattern(cache, x, nres, nbc)
    pattern = __ascher_collocation_pattern(cache, nres, length(x))
    (; ncomp, ny, k, mesh) = cache
    n = length(mesh) - 1
    nz = ncomp * (n + 1)
    ncy = ncomp + ny
    nleft = length(first(cache.bcresid_prototype))
    # Endpoint interpolation uses the first/last interval polynomial, including
    # algebraic variables. Allow every variable in those local polynomials.
    for (interval, bcrows) in ((1, 1:nleft), (n, (nleft + 1):nbc))
        zoffset = (interval - 1) * ncomp
        soffset = nz + (interval - 1) * k * ncy
        for r in bcrows
            row = nres - nbc + r
            for c in (zoffset + 1):(zoffset + ncomp)
                pattern[row, c] = 1
            end
            for c in (soffset + 1):(soffset + k * ncy)
                pattern[row, c] = 1
            end
        end
    end
    return pattern
end

function __ascher_sparse_mode(mode, pattern)
    coloring = mode.coloring_algorithm === nothing ?
        __default_coloring_algorithm(get_dense_ad(mode)) : mode.coloring_algorithm
    return AutoSparse(
        get_dense_ad(mode);
        sparsity_detector = ADTypes.KnownJacobianSparsityDetector(pattern),
        coloring_algorithm = coloring
    )
end

function __ascher_global_jacobian(cache, x, residual, nbc, loss!)
    mode = cache.alg.jac_alg.diffmode
    if !(mode isa AutoSparse)
        prepared = DI.prepare_jacobian(loss!, residual, mode, x, Constant(cache.p); strict = Val(false))
        jac! = (J, x, p) -> DI.jacobian!(loss!, residual, J, prepared, mode, x, Constant(p))
        return zeros(eltype(x), length(residual), length(x)), jac!
    elseif cache.pt isa TwoPointBVProblem
        pattern = __ascher_twopoint_pattern(cache, x, length(residual), nbc)
        sparse_mode = __ascher_sparse_mode(mode, pattern)
        prepared = DI.prepare_jacobian(loss!, residual, sparse_mode, x, Constant(cache.p); strict = Val(false))
        jac! = (J, x, p) -> DI.jacobian!(loss!, residual, J, prepared, sparse_mode, x, Constant(p))
        fill!(nonzeros(pattern), zero(eltype(x)))
        return pattern, jac!
    end

    # General boundary functions can inspect arbitrary points or combine distant
    # intervals. Differentiate their few rows separately so they cannot force
    # the interval-local collocation rows to use one color per global unknown.
    ncoll = length(residual) - nbc
    coll_residual = similar(x, ncoll)
    bc_residual = similar(x, nbc)
    coll_loss! = (res, x, p) -> __ascher_collocation_loss!(res, x, p, cache)
    bc_loss! = (res, x, p) -> __ascher_boundary_loss!(res, x, p, cache)
    pattern = __ascher_collocation_pattern(cache, ncoll, length(x))
    coll_mode = __ascher_sparse_mode(mode, pattern)
    bc_mode = get_dense_ad(cache.alg.jac_alg.bc_diffmode)
    coll_prepared = DI.prepare_jacobian(coll_loss!, coll_residual, coll_mode, x, Constant(cache.p); strict = Val(false))
    bc_prepared = DI.prepare_jacobian(bc_loss!, bc_residual, bc_mode, x, Constant(cache.p); strict = Val(false))
    Jcoll = copy(pattern)
    Jbc = zeros(eltype(x), nbc, length(x))
    prototype = vcat(pattern, sparse(ones(eltype(x), nbc, length(x))))
    fill!(nonzeros(prototype), zero(eltype(x)))
    function jac!(J, x, p)
        DI.jacobian!(coll_loss!, coll_residual, Jcoll, coll_prepared, coll_mode, x, Constant(p))
        DI.jacobian!(bc_loss!, bc_residual, Jbc, bc_prepared, bc_mode, x, Constant(p))
        __ascher_copy_jacobian!(J, Jcoll, Jbc)
        return nothing
    end
    return prototype, jac!
end

function __ascher_copy_jacobian!(J::SparseMatrixCSC, Jcoll::SparseMatrixCSC, Jbc)
    values, coll_values = nonzeros(J), nonzeros(Jcoll)
    for col in axes(J, 2)
        dest = first(nzrange(J, col))
        for src in nzrange(Jcoll, col)
            values[dest] = coll_values[src]
            dest += 1
        end
        for row in axes(Jbc, 1)
            values[dest] = Jbc[row, col]
            dest += 1
        end
    end
    return nothing
end

# A nonlinear solver may switch to a dense workspace, for example when its
# trust-region fallback factors the Jacobian with QR. Preserve the sparse AD
# preparation while accepting the workspace selected by that solver.
function __ascher_copy_jacobian!(J::AbstractMatrix, Jcoll::SparseMatrixCSC, Jbc)
    ncoll = size(Jcoll, 1)
    @views J[1:ncoll, :] .= Jcoll
    @views J[(ncoll + 1):(ncoll + size(Jbc, 1)), :] .= Jbc
    return nothing
end
