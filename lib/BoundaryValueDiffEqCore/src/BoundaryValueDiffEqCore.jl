module BoundaryValueDiffEqCore

using Adapt: adapt
using ADTypes: ADTypes, AbstractADType, AutoSparse, AutoForwardDiff, AutoFiniteDiff,
               NoSparsityDetector, KnownJacobianSparsityDetector
using ArrayInterface: matrix_colors, parameterless_type, fast_scalar_indexing
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase, solve
using ForwardDiff: ForwardDiff, pickchunksize
using Logging
using LinearAlgebra
using LineSearch: BackTracking
using NonlinearSolveFirstOrder: NonlinearSolvePolyAlgorithm
using PreallocationTools: PreallocationTools, DiffCache
using RecursiveArrayTools: AbstractVectorOfArray, VectorOfArray, DiffEqArray
using Reexport: @reexport
using SciMLBase: SciMLBase, AbstractBVProblem, AbstractDiffEqInterpolation,
                 StandardBVProblem, StandardSecondOrderBVProblem, __solve, _unwrap_val
using Setfield: @set!, @set
using SparseArrays: sparse
using SparseMatrixColorings: GreedyColoringAlgorithm

@reexport using NonlinearSolveFirstOrder, SciMLBase

include("types.jl")
include("utils.jl")
include("algorithms.jl")
include("alg_utils.jl")
include("default_nlsolve.jl")
include("misc_utils.jl")

function SciMLBase.__solve(
        prob::AbstractBVProblem, alg::BoundaryValueDiffEqAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

export BVPJacobianAlgorithm

end
