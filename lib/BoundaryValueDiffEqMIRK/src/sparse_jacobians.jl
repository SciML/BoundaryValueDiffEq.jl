# For MIRK Methods
"""
    __generate_sparse_jacobian_prototype(::MIRKCache, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::MIRKCache, _, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::MIRKCache, ::TwoPointBVProblem, ya, yb, M, N)

Generate a prototype of the sparse Jacobian matrix for the BVP problem.

If the problem is a TwoPointBVProblem, then this is the complete Jacobian, else it only
computes the sparse part excluding the contributions from the boundary conditions.
"""
function __generate_sparse_jacobian_prototype(cache::MIRKCache, ya, yb, M, N)
    return __generate_sparse_jacobian_prototype(cache, cache.problem_type, ya, yb, M, N)
end

function __generate_sparse_jacobian_prototype(
        ::MIRKCache, ::StandardBVProblem, ya, yb, M, N
    )
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J_c = BandedMatrix(Ones{eltype(ya)}(M * (N - 1), M * N), (1, 2M - 1))
    return J_c
end

function __generate_sparse_jacobian_prototype(
        ::MIRKCache, ::TwoPointBVProblem, ya, yb, M, N
    )
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J₁ = length(ya) + length(yb) + M * (N - 1)
    J₂ = M * N
    J = BandedMatrix(Ones{eltype(ya)}(J₁, J₂), (M + 1, M + 1))
    # for underdetermined systems we don't have banded qr implemented. use sparse
    J₁ < J₂ && return sparse(J)
    return J
end

"""
    __generate_control_jacobian_prototype(::MIRKCache, ::StandardBVProblem, y, M, N, L_f)
    __generate_control_jacobian_prototype(::MIRKCache, ::TwoPointBVProblem, y, M, N, L_f, L_a, L_b)

Structural Jacobian pattern for problems whose `M` unknowns per mesh node include controls,
so each interval contributes only `L_f` collocation residuals. The residuals of interval `i`
depend on the unknowns of nodes `i` and `i + 1`. For a `TwoPointBVProblem` the pattern also
covers the `L_a` and `L_b` boundary residuals, which depend on the first and last node.
"""
function __generate_control_jacobian_prototype(
        ::MIRKCache, ::StandardBVProblem, y, M, N, L_f
    )
    return __control_jacobian_pattern(y, M, N, L_f, 0, 0)
end

function __generate_control_jacobian_prototype(
        ::MIRKCache, ::TwoPointBVProblem, y, M, N, L_f, L_a, L_b
    )
    return __control_jacobian_pattern(y, M, N, L_f, L_a, L_b)
end

function __control_jacobian_pattern(y, M, N, L_f, L_a, L_b)
    n_rows = L_a + L_f * (N - 1) + L_b
    rows, cols = Int[], Int[]
    for r in 1:L_a, c in 1:M
        push!(rows, r)
        push!(cols, c)
    end
    for i in 1:(N - 1), r in 1:L_f, c in 1:(2M)
        push!(rows, L_a + (i - 1) * L_f + r)
        push!(cols, (i - 1) * M + c)
    end
    for r in 1:L_b, c in 1:M
        push!(rows, n_rows - L_b + r)
        push!(cols, (N - 1) * M + c)
    end
    return _sparse_like(rows, cols, y, n_rows, M * N)
end
