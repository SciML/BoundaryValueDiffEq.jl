module BoundaryValueDiffEq

using Adapt, LinearAlgebra, PreallocationTools, Reexport, Setfield, SparseArrays, SciMLBase,
    RecursiveArrayTools
@reexport using ADTypes, DiffEqBase, NonlinearSolve, SparseDiffTools, SciMLBase

import ADTypes: AbstractADType
import ArrayInterface: matrix_colors, parameterless_type
import ConcreteStructs: @concrete
import DiffEqBase: solve
import ForwardDiff: pickchunksize
import RecursiveArrayTools: ArrayPartition, DiffEqArray
import SciMLBase: AbstractDiffEqInterpolation
import SparseDiffTools: AbstractSparseADType
import TruncatedStacktraces: @truncate_stacktrace
import UnPack: @unpack

include("types.jl")
include("utils.jl")
include("algorithms.jl")
include("alg_utils.jl")
include("mirk_tableaus.jl")
include("cache.jl")
include("collocation.jl")
include("nlprob.jl")
include("solve/single_shooting.jl")
include("solve/mirk.jl")
include("adaptivity.jl")
include("lobatto_tableaus.jl")
include("interpolation.jl")

function SciMLBase.__solve(prob::BVProblem, alg::BoundaryValueDiffEqAlgorithm, args...;
    kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end
include("lobatto_tableaus.jl")

export Shooting
export MIRK2, MIRK3, MIRK4, MIRK5, MIRK6
export LobattoIIIb2, LobattoIIIb3, LobattoIIIb4, LobattoIIIb5
export MIRKJacobianComputationAlgorithm
# From ODEInterface.jl
export BVPM2, BVPSOL

end
