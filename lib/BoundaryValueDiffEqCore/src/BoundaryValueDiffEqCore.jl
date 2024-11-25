module BoundaryValueDiffEqCore

using ADTypes, Adapt, ArrayInterface, ForwardDiff, LinearAlgebra, LineSearch,
      NonlinearSolveFirstOrder, RecursiveArrayTools, Reexport, SciMLBase, Setfield,
      SparseDiffTools

using PreallocationTools: PreallocationTools, DiffCache

# Special Matrix Types
using SparseArrays

using ADTypes: AbstractADType
using ArrayInterface: matrix_colors, parameterless_type, fast_scalar_indexing
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase, solve
using ForwardDiff: ForwardDiff, pickchunksize
using Logging
using NonlinearSolveFirstOrder: NonlinearSolvePolyAlgorithm
using LineSearch: BackTracking
using RecursiveArrayTools: VectorOfArray, DiffEqArray
using SciMLBase: SciMLBase, AbstractBVProblem, AbstractDiffEqInterpolation,
                 StandardBVProblem, StandardSecondOrderBVProblem, __solve, _unwrap_val

@reexport using NonlinearSolveFirstOrder, SparseDiffTools, SciMLBase

include("types.jl")
include("utils.jl")
include("algorithms.jl")
include("alg_utils.jl")
include("default_nlsolve.jl")
include("sparse_jacobians.jl")

function SciMLBase.__solve(
        prob::AbstractBVProblem, alg::BoundaryValueDiffEqAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

export BVPJacobianAlgorithm

end
