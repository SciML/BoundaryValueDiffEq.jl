module BoundaryValueDiffEq

using Adapt, LinearAlgebra, PreallocationTools, Reexport, Setfield, SparseArrays, SciMLBase,
    Static, RecursiveArrayTools, ForwardDiff
@reexport using ADTypes, DiffEqBase, NonlinearSolve, SparseDiffTools, SciMLBase

import ADTypes: AbstractADType
import ArrayInterface: matrix_colors, parameterless_type, undefmatrix
import ConcreteStructs: @concrete
import DiffEqBase: solve
import ForwardDiff: pickchunksize
import RecursiveArrayTools: ArrayPartition, DiffEqArray
import SciMLBase: AbstractDiffEqInterpolation, StandardBVProblem, __solve
import RecursiveArrayTools: ArrayPartition
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
include("lobatto_tableaus.jl")
include("radau_tableaus.jl")
include("interpolation.jl")

function __solve(prob::BVProblem, alg::BoundaryValueDiffEqAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

export Shooting, MultipleShooting
export MIRK2, MIRK3, MIRK4, MIRK5, MIRK6
export RadauIIa1, RadauIIa3, RadauIIa5,RadauIIa9,RadauIIa13
export LobattoIIIa2, LobattoIIIa3, LobattoIIIa4, LobattoIIIa5
export LobattoIIIb2, LobattoIIIb3, LobattoIIIb4, LobattoIIIb5
export MIRKJacobianComputationAlgorithm, BVPJacobianAlgorithm
# From ODEInterface.jl
export BVPM2, BVPSOL

end
