# For FIRK Methods
"""
    __generate_sparse_jacobian_prototype(::FIRKCacheNested, ::StandardBVProblem, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::FIRKCacheNested, ::TwoPointBVProblem, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::FIRKCacheExpand, ::StandardBVProblem, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::FIRKCacheExpand, ::TwoPointBVProblem, ya, yb, M, N)

Generate a prototype of the sparse Jacobian matrix for the BVP problem with row and column
coloring.

If the problem is a TwoPointBVProblem, then this is the complete Jacobian, else it only
computes the sparse part excluding the contributions from the boundary conditions.
"""
function __generate_sparse_jacobian_prototype(
        ::FIRKCacheNested, ::StandardBVProblem, ya, yb, M, N)
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J_c = BandedMatrix(Ones{eltype(ya)}(M * (N - 1), M * N), (1, 2M - 1))
    return ColoredMatrix(J_c, matrix_colors(J_c'), matrix_colors(J_c))
end

function __generate_sparse_jacobian_prototype(
        ::FIRKCacheNested, ::TwoPointBVProblem, ya, yb, M, N)
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J₁ = length(ya) + length(yb) + M * (N - 1)
    J₂ = M * N
    J = BandedMatrix(Ones{eltype(ya)}(J₁, J₂), (M + 1, M + 1))
    # for underdetermined systems we don't have banded qr implemented. use sparse
    J₁ < J₂ && return ColoredMatrix(sparse(J), matrix_colors(J'), matrix_colors(J))
    return ColoredMatrix(J, matrix_colors(J'), matrix_colors(J))
end

function __generate_sparse_jacobian_prototype(
        cache::FIRKCacheExpand, ::StandardBVProblem, ya, yb, M, N)
    (; stage) = cache

    # Get number of nonzeros
    block_size = M * (stage + 1) * M * (stage + 2)
    l = (N - 1) * block_size
    # Initialize Is and Js
    Is = Vector{Int}(undef, l)
    Js = Vector{Int}(undef, l)

    # Fill Is and Js
    row_size = M * (stage + 1) * (N - 1)

    idx = 1
    i_start = 0
    j_start = 0
    i_step = M * (stage + 1)
    j_step = M * (stage + 2)
    for k in 1:(N - 1)
        for i in 1:i_step
            for j in 1:j_step
                Is[idx] = i + i_start
                Js[idx] = j + j_start
                idx += 1
            end
        end
        i_start += i_step
        j_start += i_step
    end

    # Create sparse matrix from Is and Js
    J_c = _sparse_like(Is, Js, ya, row_size, row_size + M)

    col_colorvec = Vector{Int}(undef, size(J_c, 2))
    for i in eachindex(col_colorvec)
        col_colorvec[i] = mod1(i, (2 * M * (stage + 1)) + M)
    end
    row_colorvec = Vector{Int}(undef, size(J_c, 1))
    for i in eachindex(row_colorvec)
        row_colorvec[i] = mod1(i, (2 * M * (stage + 1)) + M)
    end

    return ColoredMatrix(J_c, row_colorvec, col_colorvec)
end

function __generate_sparse_jacobian_prototype(
        cache::FIRKCacheExpand, ::TwoPointBVProblem, ya, yb, M, N)
    (; stage) = cache

    # Get number of nonzeros
    block_size = M * (stage + 1) * M * (stage + 2)
    l = (N - 1) * block_size + M * (stage + 2) * (length(ya) + length(yb))
    # Initialize Is and Js
    Is = Vector{Int}(undef, l)
    Js = Vector{Int}(undef, l)

    # Fill Is and Js
    row_size = M * (stage + 1) * (N - 1)
    idx = 1
    i_start = 0
    j_start = 0
    i_step = M * (stage + 1)
    j_step = M * (stage + 2)

    # Fill first rows
    for i in 1:length(ya)
        for j in 1:j_step
            Is[idx] = i
            Js[idx] = j
            idx += 1
        end
    end
    i_start += length(ya)

    for k in 1:(N - 1)
        for i in 1:i_step
            for j in 1:j_step
                Is[idx] = i + i_start
                Js[idx] = j + j_start
                idx += 1
            end
        end
        i_start += i_step
        j_start += i_step
    end
    j_start -= i_step
    #Fill last rows
    for i in 1:length(yb)
        for j in 1:j_step
            Is[idx] = i + i_start
            Js[idx] = j + j_start
            idx += 1
        end
    end

    # Create sparse matrix from Is and Js
    J = _sparse_like(Is, Js, ya, row_size + length(ya) + length(yb), row_size + M)
    return ColoredMatrix(J, matrix_colors(J'), matrix_colors(J))
end
