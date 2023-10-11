# This file defines several common patterns of sparse Jacobians we see in the BVP solvers.
function _sparse_like(I, J, x::AbstractArray, m = maximum(I), n = maximum(J))
    I′ = adapt(parameterless_type(x), I)
    J′ = adapt(parameterless_type(x), J)
    V = __ones_like(x, length(I))
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

Base.size(M::ColoredMatrix, args...) = size(M.M, args...)
Base.eltype(M::ColoredMatrix) = eltype(M.M)

ColoredMatrix() = ColoredMatrix(nothing, nothing, nothing)

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
"""
    __generate_sparse_jacobian_prototype(::MultipleShooting, ::StandardBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int)
    __generate_sparse_jacobian_prototype(::MultipleShooting, ::TwoPointBVProblem,
        bcresid_prototype, u0, N::Int, nshoots::Int)

Returns a 3-Tuple:

* Entire Jacobian Prototype (if Two-Point Problem) else `nothing`.
* Sparse Non-BC Part Jacobian Prototype along with the column and row color vectors.
* Sparse BC Part Jacobian Prototype along with the column and row color vectors (if
  Two-Point Problem) else `nothing`.
"""
function __generate_sparse_jacobian_prototype(::MultipleShooting, ::StandardBVProblem,
    bcresid_prototype, u0, N::Int, nshoots::Int)
    Is = Vector{Int}(undef, (N^2 + N) * nshoots)
    Js = Vector{Int}(undef, (N^2 + N) * nshoots)

    idx = 1
    for i in 1:nshoots
        for (i₁, i₂) in Iterators.product(1:N, 1:N)
            Is[idx] = i₁ + ((i - 1) * N)
            Js[idx] = i₂ + ((i - 1) * N)
            idx += 1
        end
        Is[idx:(idx + N - 1)] .= (1:N) .+ ((i - 1) * N)
        Js[idx:(idx + N - 1)] .= (1:N) .+ (i * N)
        idx += N
    end

    J_c = _sparse_like(Is, Js, u0)

    col_colorvec = Vector{Int}(undef, size(J_c, 2))
    for i in eachindex(col_colorvec)
        col_colorvec[i] = mod1(i, 2N)
    end
    row_colorvec = Vector{Int}(undef, size(J_c, 1))
    for i in eachindex(row_colorvec)
        row_colorvec[i] = mod1(i, 2N)
    end

    return nothing, ColoredMatrix(J_c, row_colorvec, col_colorvec), nothing
end

function __generate_sparse_jacobian_prototype(alg::MultipleShooting, ::TwoPointBVProblem,
    bcresid_prototype::ArrayPartition, u0, N::Int, nshoots::Int)
    resida, residb = bcresid_prototype.x
    L₁, L₂ = length(resida), length(residb)

    _, J_c, _ = __generate_sparse_jacobian_prototype(alg, StandardBVProblem(),
        bcresid_prototype, u0, N, nshoots)

    Is_bc = Vector{Int}(undef, (L₁ + L₂) * N)
    Js_bc = Vector{Int}(undef, (L₁ + L₂) * N)
    idx = 1
    for i in 1:L₁, j in 1:N
        Is_bc[idx] = i
        Js_bc[idx] = j
        idx += 1
    end
    for i in 1:L₂, j in 1:N
        Is_bc[idx] = i + L₁
        Js_bc[idx] = j + N
        idx += 1
    end

    col_colorvec_bc = Vector{Int}(undef, 2N)
    row_colorvec_bc = Vector{Int}(undef, L₁ + L₂)
    col_colorvec_bc[1:N] .= 1:N
    col_colorvec_bc[(N + 1):end] .= 1:N
    for i in 1:max(L₁, L₂)
        i ≤ L₁ && (row_colorvec_bc[i] = i)
        i ≤ L₂ && (row_colorvec_bc[i + L₁] = i)
    end

    J_bc = ColoredMatrix(_sparse_like(Is_bc, Js_bc, bcresid_prototype), row_colorvec_bc,
        col_colorvec_bc)

    J_full = _sparse_like(Int[], Int[], u0, size(J_bc, 1) + size(J_c, 1),
        size(J_c, 2))

    J_full[(L₁ + L₂ + 1):end, :] .= J_c.M
    J_full[1:L₁, 1:N] .= J_bc.M[1:L₁, 1:N]
    J_full[(L₁ + 1):(L₁ + L₂), (end - 2N + 1):(end - N)] .= J_bc.M[(L₁ + 1):(L₁ + L₂),
        (N + 1):(2N)]

    return J_full, J_c, J_bc
end
