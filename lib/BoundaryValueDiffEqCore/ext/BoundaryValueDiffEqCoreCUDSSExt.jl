module BoundaryValueDiffEqCoreCUDSSExt

using BoundaryValueDiffEqCore: BoundaryValueDiffEqCore
using CUDA: CUDA
using CUDSS: CUDSS
using LinearAlgebra: LinearAlgebra
using LinearSolve: LinearSolve, GenericFactorization

@static if isdefined(CUDA, :cuSPARSE)
    const CuCSR = CUDA.cuSPARSE.CuSparseMatrixCSR
else
    const CuCSR = CUDA.CUSPARSE.CuSparseMatrixCSR
end

const LinearVerbosity = @static if isdefined(LinearSolve, :LinearVerbosity)
    Union{LinearSolve.LinearVerbosity, Bool}
else
    Bool
end

# One matched factorization per nonlinear solver and mesh. In particular, do
# not analyze the zero-valued Jacobian placeholder created by NonlinearSolve.
mutable struct MatchedCUDSSFactorization{T, I}
    solver::Union{Nothing, CUDSS.CudssSolver{T, I}}
end

function LinearSolve.init_cacheval(
        ::GenericFactorization{<:MatchedCUDSSFactorization}, A::CuCSR,
        b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::LinearVerbosity, assumptions::LinearSolve.OperatorAssumptions
    )
    return CUDSS.CudssSolver(A, "G", 'F')
end

function (factorization::MatchedCUDSSFactorization{T, I})(A) where {T, I}
    solver = factorization.solver
    if solver === nothing
        x = CUDSS.CudssMatrix(T, size(A, 1))
        b = CUDSS.CudssMatrix(T, size(A, 1))
        solver = CUDSS.CudssSolver(A, "G", 'F')
        if pkgversion(CUDSS) < v"0.8"
            CUDSS.cudss_set(solver, "use_matching", 1)
        end
        CUDSS.cudss_set(solver, "matching_alg", "algo2")
        CUDSS.cudss("analysis", solver, x, b; asynchronous = false)
        CUDSS.cudss("factorization", solver, x, b; asynchronous = false)
        factorization.solver = solver
    else
        LinearAlgebra.lu!(solver, A)
    end
    return solver
end

function BoundaryValueDiffEqCore.__device_sparse_linsolve(::CuCSR{T, I}) where {T, I}
    return GenericFactorization(; fact_alg = MatchedCUDSSFactorization{T, I}(nothing))
end

end
