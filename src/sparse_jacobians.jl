# This file defines several common patterns of sparse Jacobians we see in the BVP solvers.
function _sparse_like(I, J, x::AbstractArray, m = maximum(I), n = maximum(J))
    I′ = adapt(parameterless_type(x), I)
    J′ = adapt(parameterless_type(x), J)
    V = similar(x, length(I))
    return sparse(I′, J′, V, m, n)
end

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

function SparseDiffTools.PrecomputedJacobianColorvec(M::ColoredMatrix)
    return PrecomputedJacobianColorvec(; jac_prototype = M.M, M.row_colorvec,
        M.col_colorvec)
end

# For MIRK Methods
"""
    __generate_sparse_jacobian_prototype(::MIRKCache, y, M, N)
    __generate_sparse_jacobian_prototype(::MIRKCache, _, y, M, N)
    __generate_sparse_jacobian_prototype(::MIRKCache, ::TwoPointBVProblem, y, M, N)

Generate a prototype of the sparse Jacobian matrix for the BVP problem with row and column
coloring.

If the problem is a TwoPointBVProblem, then this is the complete Jacobian, else it only
computes the sparse part excluding the contributions from the boundary conditions.
"""
function __generate_sparse_jacobian_prototype(cache::MIRKCache, y, M, N)
    return __generate_sparse_jacobian_prototype(cache, cache.problem_type, y, M, N)
end

function __generate_sparse_jacobian_prototype(::MIRKCache, _, y, M, N)
    l = sum(i -> min(2M + i, M * N) - max(1, i - 1) + 1, 1:(M * (N - 1)))
    Is = Vector{Int}(undef, l)
    Js = Vector{Int}(undef, l)
    idx = 1
    for i in 1:(M * (N - 1)), j in max(1, i - 1):min(2M + i, M * N)
        Is[idx] = i
        Js[idx] = j
        idx += 1
    end

    J_c = _sparse_like(Is, Js, y, M * (N - 1), M * N)

    col_colorvec = Vector{Int}(undef, size(J_c, 2))
    for i in eachindex(col_colorvec)
        col_colorvec[i] = mod1(i, min(2M + 1, M * N) + 1)
    end
    row_colorvec = Vector{Int}(undef, size(J_c, 1))
    for i in eachindex(row_colorvec)
        row_colorvec[i] = mod1(i, min(2M + 1, M * N) + 1)
    end

    return ColoredMatrix(J_c, row_colorvec, col_colorvec)
end

function __generate_sparse_jacobian_prototype(::MIRKCache, ::TwoPointBVProblem,
    y::ArrayPartition, M, N)
    resida, residb = y.x

    l = sum(i -> min(2M + i, M * N) - max(1, i - 1) + 1, 1:(M * (N - 1)))
    l_top = M * length(resida)
    l_bot = M * length(residb)

    Is = Vector{Int}(undef, l + l_top + l_bot)
    Js = Vector{Int}(undef, l + l_top + l_bot)

    idx = 1
    for i in 1:length(resida), j in 1:M
        Is[idx] = i
        Js[idx] = j
        idx += 1
    end
    for i in 1:(M * (N - 1)), j in max(1, i - 1):min(2M + i, M * N)
        Is[idx] = i + length(resida)
        Js[idx] = j
        idx += 1
    end
    for i in 1:length(residb), j in 1:M
        Is[idx] = i + length(resida) + M * (N - 1)
        Js[idx] = j + M * (N - 1)
        idx += 1
    end

    J = _sparse_like(Is, Js, y, M * N, M * N)

    col_colorvec = Vector{Int}(undef, size(J, 2))
    for i in eachindex(col_colorvec)
        col_colorvec[i] = mod1(i, min(2M + 1, M * N) + 1)
    end
    row_colorvec = Vector{Int}(undef, size(J, 1))
    for i in eachindex(row_colorvec)
        row_colorvec[i] = mod1(i, min(2M + 1, M * N) + 1)
    end

    return ColoredMatrix(J, row_colorvec, col_colorvec)
end

# For Multiple Shooting