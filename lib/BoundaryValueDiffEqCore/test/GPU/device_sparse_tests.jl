using BoundaryValueDiffEqCore: __device_sparse_matrix, __device_sparse_supported
using SparseArrays
using Test

function test_device_sparse_storage(adaptor)
    # Rectangular, nonsymmetric, with an empty row/column and an explicit zero.
    # Boolean patterns are used by Shooting; values must follow the template type.
    patterns = (
        sparse([1, 4, 1, 3], [1, 1, 3, 5], [true, true, false, true], 4, 6),
        sparse([1, 4, 1, 3], [1, 1, 3, 5], [2.0, 3.0, 0.0, 5.0], 4, 6),
        spzeros(Bool, 4, 6),
    )
    for T in (Float32, Float64), pattern in patterns
        original = copy(pattern)
        template = adaptor(zeros(T, 2, 3))
        @test __device_sparse_supported(template)
        (; matrix, rows, cols) = __device_sparse_matrix(template, pattern)
        @test eltype(matrix) === T
        @test size(matrix) == size(pattern)
        @test nnz(matrix) == nnz(pattern)
        @test rows isa Vector{Int}
        @test cols isa Vector{Int}
        @test length(rows) == length(cols) == nnz(pattern)
        @test all(iszero, nonzeros(matrix))

        # Writing unique values through the returned coordinates must agree with
        # writing them directly into the backend's nonzero storage (CSC or CSR).
        values = T.(1:nnz(pattern))
        copyto!(nonzeros(matrix), values)
        expected = sparse(rows, cols, values, size(pattern)...)
        @test Array(matrix) == Matrix(expected)
        pattern_rows, pattern_cols, _ = findnz(pattern)
        @test Set(zip(rows, cols)) == Set(zip(pattern_rows, pattern_cols))
        @test pattern.colptr == original.colptr
        @test rowvals(pattern) == rowvals(original)
        @test nonzeros(pattern) == nonzeros(original)
    end
    return nothing
end
