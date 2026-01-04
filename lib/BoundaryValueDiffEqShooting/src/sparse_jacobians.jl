# For Multiple Shooting
"""
    __generate_sparse_jacobian_prototype(::MultipleShooting, ::StandardBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int)
    __generate_sparse_jacobian_prototype(::MultipleShooting, ::TwoPointBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int)

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
