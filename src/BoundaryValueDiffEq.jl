module BoundaryValueDiffEq

using Adapt, LinearAlgebra, PreallocationTools, Reexport, Setfield, SparseArrays
@reexport using ADTypes, DiffEqBase, NonlinearSolve, SparseDiffTools

import ADTypes: AbstractADType
import ArrayInterface: matrix_colors, parameterless_type
import BandedMatrices: BandedMatrix
import ConcreteStructs: @concrete
import DiffEqBase: solve
import ForwardDiff: pickchunksize
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
include("solve.jl")
include("adaptivity.jl")

export Shooting
export MIRK2, MIRK3, MIRK4, MIRK5, MIRK6
export MIRKJacobianComputationAlgorithm

end
