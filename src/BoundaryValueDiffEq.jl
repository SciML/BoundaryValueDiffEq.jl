module BoundaryValueDiffEq

import PrecompileTools: @compile_workload, @setup_workload

using ADTypes, Adapt, ArrayInterface, DiffEqBase, ForwardDiff, LinearAlgebra,
      NonlinearSolve, OrdinaryDiffEq, Preferences, RecursiveArrayTools, Reexport, SciMLBase,
      Setfield, SparseDiffTools

using PreallocationTools: PreallocationTools, DiffCache

# Special Matrix Types
using BandedMatrices, FastAlmostBandedMatrices, SparseArrays

import ADTypes: AbstractADType
import ArrayInterface: matrix_colors, parameterless_type, undefmatrix, fast_scalar_indexing
import ConcreteStructs: @concrete
import DiffEqBase: solve
import FastClosures: @closure
import ForwardDiff: ForwardDiff, pickchunksize
import Logging
import RecursiveArrayTools: ArrayPartition, DiffEqArray
import SciMLBase: AbstractDiffEqInterpolation, StandardBVProblem, __solve, _unwrap_val

@reexport using ADTypes, DiffEqBase, NonlinearSolve, OrdinaryDiffEq, SparseDiffTools,
                SciMLBase

include("types.jl")
include("utils.jl")
include("algorithms.jl")
include("alg_utils.jl")

include("mirk_tableaus.jl")
include("lobatto_tableaus.jl")
include("radau_tableaus.jl")

include("solve/single_shooting.jl")
include("solve/multiple_shooting.jl")
include("solve/firk.jl")
include("solve/mirk.jl")

include("collocation.jl")
include("sparse_jacobians.jl")

include("adaptivity.jl")
include("interpolation.jl")

include("default_nlsolve.jl")

function __solve(prob::BVProblem, alg::BoundaryValueDiffEqAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

export Shooting, MultipleShooting
export MIRK2, MIRK3, MIRK4, MIRK5, MIRK6
export BVPM2, BVPSOL, COLNEW # From ODEInterface.jl

export RadauIIa1, RadauIIa2, RadauIIa3, RadauIIa5, RadauIIa7
export LobattoIIIa2, LobattoIIIa3, LobattoIIIa4, LobattoIIIa5
export LobattoIIIb2, LobattoIIIb3, LobattoIIIb4, LobattoIIIb5
export LobattoIIIc2, LobattoIIIc3, LobattoIIIc4, LobattoIIIc5
export MIRKJacobianComputationAlgorithm, BVPJacobianAlgorithm

end
