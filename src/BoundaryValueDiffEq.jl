module BoundaryValueDiffEq

using Adapt, BandedMatrices, ForwardDiff, LinearAlgebra, PreallocationTools,
    RecursiveArrayTools, Reexport, Setfield, SparseArrays
@reexport using ADTypes, DiffEqBase, NonlinearSolve, SparseDiffTools, SciMLBase

import ADTypes: AbstractADType
import ArrayInterface: matrix_colors, parameterless_type, undefmatrix, fast_scalar_indexing
import ConcreteStructs: @concrete
import DiffEqBase: solve
import ForwardDiff: pickchunksize
import RecursiveArrayTools: ArrayPartition, DiffEqArray
import SciMLBase: AbstractDiffEqInterpolation, StandardBVProblem, __solve, _unwrap_val
import SparseDiffTools: AbstractSparseADType
import TruncatedStacktraces: @truncate_stacktrace
import UnPack: @unpack

include("types.jl")
include("utils.jl")
include("algorithms.jl")
include("alg_utils.jl")

include("mirk_tableaus.jl")

include("solve/single_shooting.jl")
include("solve/multiple_shooting.jl")
include("solve/mirk.jl")

include("collocation.jl")
include("sparse_jacobians.jl")

include("adaptivity.jl")
include("interpolation.jl")

function __solve(prob::BVProblem, alg::BoundaryValueDiffEqAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

export Shooting, MultipleShooting
export MIRK2, MIRK3, MIRK4, MIRK5, MIRK6
export MIRKJacobianComputationAlgorithm, BVPJacobianAlgorithm
# From ODEInterface.jl
export BVPM2, BVPSOL

end
