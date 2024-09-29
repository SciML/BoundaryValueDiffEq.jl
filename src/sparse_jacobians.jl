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
    return PrecomputedJacobianColorvec(;
        jac_prototype = M.M, M.row_colorvec, M.col_colorvec)
end
__sparsity_detection_alg(::ColoredMatrix{Nothing}) = NoSparsityDetection()
