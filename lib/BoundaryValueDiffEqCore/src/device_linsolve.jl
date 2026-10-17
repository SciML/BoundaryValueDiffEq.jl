"""
    __device_sparse_linsolve(matrix)

Select a sparse direct solver for device collocation systems, or return `nothing`
to use the nonlinear solver's default. The CUDSS extension supplies a cached
factorization with numerical matching for CUDA CSR matrices.
"""
__device_sparse_linsolve(matrix) = nothing

"""
    __default_linsolve(u)

Select the portable linear solver for a resident square system. Host arrays use
LinearSolve's default; other backends use GMRES without requiring a factorization.
Solver-specific extensions may override this choice at their call site.
"""
__default_linsolve(u) = KrylovJL_GMRES()
__default_linsolve(::Array) = nothing

"""
    __default_sparse_linsolve(matrix)

Select a linear solver for a resident sparse system. Float64 host CSC matrices use
UMFPACK; Float32 host CSC matrices use GMRES to preserve Float32 storage and avoid
LinearSolve 4.2's default LU path, which passes Float32 directly to an unsupported
UMFPACK constructor. Other matrices use the optional backend adapter in
`__device_sparse_linsolve`.
"""
__default_sparse_linsolve(matrix) = __device_sparse_linsolve(matrix)
__default_sparse_linsolve(::SparseMatrixCSC{Float64}) = UMFPACKFactorization()
__default_sparse_linsolve(::SparseMatrixCSC{Float32}) = KrylovJL_GMRES()
