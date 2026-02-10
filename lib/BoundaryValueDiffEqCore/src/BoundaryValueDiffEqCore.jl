module BoundaryValueDiffEqCore

using Adapt: adapt
using ADTypes: ADTypes, AbstractADType, AutoSparse, AutoForwardDiff, AutoFiniteDiff,
    NoSparsityDetector, KnownJacobianSparsityDetector, AutoPolyesterForwardDiff
using ArrayInterface: parameterless_type, fast_scalar_indexing
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase, solve
using ForwardDiff: ForwardDiff, pickchunksize
using Integrals: Integrals, IntegralProblem
using Logging: Logging
using LinearAlgebra
using LineSearch: BackTracking
using NonlinearSolveFirstOrder: NonlinearSolvePolyAlgorithm
using OptimizationBase: OptimizationBase
using PreallocationTools: PreallocationTools, DiffCache
using RecursiveArrayTools: AbstractVectorOfArray, VectorOfArray, DiffEqArray
using Reexport: @reexport
using SciMLBase: SciMLBase, AbstractBVProblem, AbstractDiffEqInterpolation,
    StandardBVProblem, StandardSecondOrderBVProblem, __solve, _unwrap_val
using SciMLLogging: SciMLLogging, @SciMLMessage, @verbosity_specifier
using Setfield: @set!, @set
using SparseArrays: sparse
using SparseConnectivityTracer: SparseConnectivityTracer, TracerLocalSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using SciMLStructures: SciMLStructures

@reexport using NonlinearSolveFirstOrder, SciMLBase

include("types.jl")
include("solution_utils.jl")
include("utils.jl")
include("internal_problems.jl")
include("algorithms.jl")
include("abstract_types.jl")
include("alg_utils.jl")
include("default_internal_solve.jl")
include("calc_errors.jl")
include("verbosity.jl")

function SciMLBase.__solve(
        prob::AbstractBVProblem,
        alg::AbstractBoundaryValueDiffEqAlgorithm, args...; kwargs...
    )
    cache = SciMLBase.__init(prob, alg, args...; kwargs...)
    return SciMLBase.solve!(cache)
end

export AbstractBoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm
export DefectControl, GlobalErrorControl, SequentialErrorControl, HybridErrorControl,
    NoErrorControl
export HOErrorControl, REErrorControl
export integral
export BVPVerbosity, _process_verbose_param, DEFAULT_VERBOSE

end
