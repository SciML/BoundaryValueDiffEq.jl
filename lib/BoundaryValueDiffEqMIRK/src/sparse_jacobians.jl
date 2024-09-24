# For MIRK Methods
"""
    __generate_sparse_jacobian_prototype(::MIRKCache, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::MIRKCache, _, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::MIRKCache, ::TwoPointBVProblem, ya, yb, M, N)

Generate a prototype of the sparse Jacobian matrix for the BVP problem with row and column
coloring.

If the problem is a TwoPointBVProblem, then this is the complete Jacobian, else it only
computes the sparse part excluding the contributions from the boundary conditions.
"""
function __generate_sparse_jacobian_prototype(cache::MIRKCache, ya, yb, M, N)
    return __generate_sparse_jacobian_prototype(cache, cache.problem_type, ya, yb, M, N)
end

function __generate_sparse_jacobian_prototype(
        ::MIRKCache, ::StandardBVProblem, ya, yb, M, N)
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J_c = BandedMatrix(Ones{eltype(ya)}(M * (N - 1), M * N), (1, 2M - 1))
    return ColoredMatrix(J_c, matrix_colors(J_c'), matrix_colors(J_c))
end

function __generate_sparse_jacobian_prototype(
        ::MIRKCache, ::TwoPointBVProblem, ya, yb, M, N)
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J₁ = length(ya) + length(yb) + M * (N - 1)
    J₂ = M * N
    J = BandedMatrix(Ones{eltype(ya)}(J₁, J₂), (M + 1, M + 1))
    # for underdetermined systems we don't have banded qr implemented. use sparse
    J₁ < J₂ && return ColoredMatrix(sparse(J), matrix_colors(J'), matrix_colors(J))
    return ColoredMatrix(J, matrix_colors(J'), matrix_colors(J))
end
