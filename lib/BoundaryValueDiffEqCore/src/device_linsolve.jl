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

Select a sparse direct solver for a resident system. Float64 host CSC matrices use
UMFPACK; other matrices use the optional backend adapter in `__device_sparse_linsolve`.
"""
__default_sparse_linsolve(matrix) = __device_sparse_linsolve(matrix)
__default_sparse_linsolve(::SparseMatrixCSC{Float64}) = UMFPACKFactorization()
