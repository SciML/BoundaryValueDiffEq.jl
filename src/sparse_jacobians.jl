# This file defines several common patterns of sparse Jacobians we see in the BVP solvers.
function _sparse_like(I, J, x::AbstractArray, m = maximum(I), n = maximum(J))
    I′ = adapt(parameterless_type(x), I)
    J′ = adapt(parameterless_type(x), J)
    V = __ones_like(x, length(I))
    return sparse(I′, J′, V, m, n)
end

# NOTE: We don't retain the Banded Structure in non-TwoPoint BVP cases since vcat/hcat makes
# it into a dense array. Instead we can atleast exploit sparsity!

# FIXME: Fix the cases where fast_scalar_indexing is not possible

# Helpers for IIP/OOP functions
function __sparse_jacobian_cache(::Val{iip}, ad, sd, fn, fx, y) where {iip}
    if iip
        sparse_jacobian_cache(ad, sd, fn, fx, y)
    else
        sparse_jacobian_cache(ad, sd, fn, y; fx)
    end
end

@concrete struct ColoredMatrix
    M
    row_colorvec
    col_colorvec
end

Base.size(M::ColoredMatrix, args...) = size(M.M, args...)
Base.eltype(M::ColoredMatrix) = eltype(M.M)

ColoredMatrix() = ColoredMatrix(nothing, nothing, nothing)

function __sparsity_detection_alg(M::ColoredMatrix)
    return PrecomputedJacobianColorvec(; jac_prototype = M.M, M.row_colorvec,
        M.col_colorvec)
end
__sparsity_detection_alg(::ColoredMatrix{Nothing}) = NoSparsityDetection()

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

function __generate_sparse_jacobian_prototype(::MIRKCache, ::StandardBVProblem, ya, yb, M,
        N)
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J_c = BandedMatrix(Ones{eltype(ya)}(M * (N - 1), M * N), (1, 2M - 1))
    return ColoredMatrix(J_c, matrix_colors(J_c'), matrix_colors(J_c))
end

function __generate_sparse_jacobian_prototype(::MIRKCache, ::TwoPointBVProblem,
        ya, yb, M, N)
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J₁ = length(ya) + length(yb) + M * (N - 1)
    J₂ = M * N
    J = BandedMatrix(Ones{eltype(ya)}(J₁, J₂), (M + 1, M + 1))
    # for underdetermined systems we don't have banded qr implemented. use sparse
    J₁ < J₂ && return ColoredMatrix(sparse(J), matrix_colors(J'), matrix_colors(J))
    return ColoredMatrix(J, matrix_colors(J'), matrix_colors(J))
end

# For Multiple Shooting
"""
    __generate_sparse_jacobian_prototype(::MultipleShooting, ::StandardBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int)
    __generate_sparse_jacobian_prototype(::MultipleShooting, ::TwoPointBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int)

Returns a 3-Tuple:

  - Entire Jacobian Prototype (if Two-Point Problem) else `nothing`.
  - Sparse Non-BC Part Jacobian Prototype along with the column and row color vectors.
  - Sparse BC Part Jacobian Prototype along with the column and row color vectors (if
    Two-Point Problem) else `nothing`.
"""
function __generate_sparse_jacobian_prototype(::MultipleShooting, ::StandardBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int)
    fast_scalar_indexing(u0) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J₁ = nshoots * N
    J₂ = (nshoots + 1) * N
    J = BandedMatrix(Ones{eltype(u0)}(J₁, J₂), (N - 1, N + 1))

    return ColoredMatrix(sparse(J), matrix_colors(J'), matrix_colors(J))
end

function __generate_sparse_jacobian_prototype(::MultipleShooting, ::TwoPointBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int)
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
    J₁ < J₂ && return ColoredMatrix(sparse(J), matrix_colors(J'), matrix_colors(J))
    return ColoredMatrix(J, matrix_colors(J'), matrix_colors(J))
end
