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
