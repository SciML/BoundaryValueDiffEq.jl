module BoundaryValueDiffEqCoreCUDAExt

using BoundaryValueDiffEqCore: BoundaryValueDiffEqCore
using CUDA: CUDA, CuArray
using SparseArrays: SparseMatrixCSC, nnz, nzrange, rowvals, sparse

@static if isdefined(CUDA, :cuSPARSE)
    using CUDA: cuSPARSE
else
    using CUDA: CUSPARSE as cuSPARSE
end

BoundaryValueDiffEqCore.__device_sparse_supported(::CuArray) = true

function BoundaryValueDiffEqCore.__device_sparse_matrix(
        template::CuArray, pattern::SparseMatrixCSC
    )
    # CSC storage of the transpose gives the CSR structure directly, without
    # constructing a dense matrix or converting sparse storage on the device.
    pattern_t = sparse(transpose(pattern))
    cols = Vector{Int}(rowvals(pattern_t))
    rows = Vector{Int}(undef, nnz(pattern_t))
    for row in axes(pattern, 1), k in nzrange(pattern_t, row)
        rows[k] = row
    end

    matrix = CUDA.device!(CUDA.device(template)) do
        rowptr = similar(template, Int32, length(pattern_t.colptr))
        colind = similar(template, Int32, length(cols))
        values = similar(template, eltype(template), length(cols))
        copyto!(rowptr, Int32.(pattern_t.colptr))
        copyto!(colind, Int32.(cols))
        fill!(values, zero(eltype(template)))
        cuSPARSE.CuSparseMatrixCSR(rowptr, colind, values, size(pattern))
    end

    # The coordinates match nonzeros(matrix), which is updated by the solvers'
    # colored differentiation kernels without sparse setindex! or host copies.
    return (; matrix, rows, cols)
end

end
